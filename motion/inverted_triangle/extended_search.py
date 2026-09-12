"""Parallel CPU physics search with a body shift, independent landing, and CAD cost.

Candidate-selection measurements are coarse; only simulate.run can accept a flip.
No controller consumes simulated base state or contact forces.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .core import Robot, LEGS, PROX, HUB, duration, smooth, provenance, versions
from .clearance import CADClearance
from .simulate import trajectory, run
from training.wandb_logging import ExperimentLogger

_WORKER = None


def initialize(start, leg, direction):
    global _WORKER
    r = Robot()
    home = r.restore(start)
    cad = CADClearance(r)
    # Include motor/body pairs: a shin-only cost missed a third-flip motor collision.
    _WORKER = (r, home, start, leg, direction, cad)


def decode(x, home, leg, direction):
    pre = home.copy()
    pre[[0, 3, 6, 9]] += np.array([1, -1, 1, -1]) * x[0]
    pre[[1, 4, 7, 10]] = x[1:5]
    lift = home.copy()
    lift[PROX] = x[5:13]
    land = lift.copy()
    land[HUB[leg]] += direction * np.pi
    land[PROX] = x[13:21]
    return pre, lift, land


def evaluate(x):
    r, home, start, leg, direction, cad = _WORKER
    r.restore(start)
    pre, lift, land = decode(x, home, leg, direction)
    phases = trajectory(home, lift, leg, direction=direction,
                        pre_shift_pose=pre, landing_pose=land)
    limits = r.m.jnt_range[1:][PROX]
    if any(np.any((q[PROX] < limits[:, 0]) | (q[PROX] > limits[:, 1])) for _, q, _ in phases):
        return dict(cost=1e8, environment_steps=0, x=np.asarray(x).tolist(), invalid_target=True)
    previous = home.copy()
    gap = 1.; support = 1e6; contact = 0.; floor = 0.; tilt = 0.; torque = 0.
    cad_penalty = 0.; cad_min = 1.; samples = 0; steps_total = 0
    descent = 0.; last_bottom = None; last_time = None; terminated = None
    for phase, target, seconds in phases:
        steps = int(np.ceil(duration(previous, target, seconds) / r.m.opt.timestep))
        for k in range(steps):
            tau = r.tick(previous + smooth((k + 1) / steps) * (target - previous))
            steps_total += 1
            torque = max(torque, float(np.abs(tau).max()))
            if steps_total % 26:
                continue
            tilt = max(tilt, r.tilt()); floor = max(floor, r.unintended_floor_force())
            bottom = float(r.bottoms()[leg])
            if phase in ('rotate', 'angle_hold'):
                f = r.contacts_precise()
                gap = min(gap, bottom); support = min(support, float(np.delete(f, leg).min()))
                contact = max(contact, float(f[leg]))
            if phase == 'land' and last_bottom is not None and bottom < .008:
                descent = max(descent, (last_bottom - bottom) / (r.d.time - last_time))
            last_bottom = bottom; last_time = float(r.d.time)
            if steps_total % 130 == 0:
                clearance = cad.measure()['minimum_m']
                cad_min = min(cad_min, clearance)
                cad_penalty += max(.0025 - clearance, 0.) ** 2 * 1e7
                samples += 1
            if tilt > .6 or r.d.qpos[2] < .045 or not np.isfinite(r.d.qpos).all():
                terminated = phase
                break
        if terminated:
            break
        previous = target.copy()
    f = r.contacts_precise()
    tracking = abs(r.d.qpos[7 + HUB[leg]] - (home[HUB[leg]] + direction * np.pi))
    tip = abs(float(r.tip_bottoms()[leg])); speed = float(np.abs(r.d.qvel[6:]).max())
    cost = 1000. * bool(terminated)
    cost += max(.01-gap, 0)**2*80000 + max(2.-support, 0)**2*8
    cost += max(contact-.05, 0)**2*5 + max(floor-.02, 0)**2*20
    cost += np.maximum(2.5-f, 0).dot(np.maximum(2.5-f, 0))*8
    cost += max(tilt-np.deg2rad(4), 0)**2*500 + max(torque-2.5, 0)**2*8
    cost += max(tracking-.02, 0)**2*20 + max(tip-.001, 0)**2*80000
    cost += max(descent-.015, 0)**2*80000 + max(speed-.05, 0)**2*20
    cost += cad_penalty/max(samples, 1)
    cost += float(np.sum((lift[PROX]-home[PROX])**2))*.005
    return dict(cost=float(cost), x=np.asarray(x).tolist(), environment_steps=steps_total,
                minimum_rotation_floor_m=gap, minimum_support_force_N=support,
                maximum_rotation_contact_N=contact, maximum_unintended_floor_force_N=floor,
                max_tilt_deg=float(np.rad2deg(tilt)), peak_requested_torque_Nm=torque,
                minimum_search_cad_gap_m=cad_min, final_normal_force_N=f.tolist(),
                final_tip_error_m=tip, final_hub_error_rad=float(tracking),
                peak_landing_descent_m_s=descent, final_speed_rad_s=speed,
                early_termination=terminated)


def optimize(output, start, leg, direction=-1, seed=0, candidate=None,
             generations=30, population=96, workers=24):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    r = Robot(); home = r.restore(start); index = LEGS.index(leg)
    lift = home[PROX].copy(); lift[2*index+1] = 1.15 if index % 2 == 0 else -1.15
    # Move support hips rearward when lifting a front limb.
    if index < 2:
        lift[::2] += np.array([1,-1,1,-1])*.4
    land = lift.copy(); land[2*index+1] -= (1 if index % 2 == 0 else -1)*.4
    x = np.r_[0., home[[1,4,7,10]], lift, land]
    if candidate:
        c = json.loads(Path(candidate).read_text())
        if c['leg'] != leg or c['model_sha256'] != r.manifest['model_sha256']:
            raise ValueError('Candidate model/leg mismatch')
        if 'search_vector' in c:
            x = np.asarray(c['search_vector'])
        else:
            x[5:13] = c['proximal_pose']; x[13:21] = c['proximal_pose']
            x[13+2*index+1] -= (1 if index % 2 == 0 else -1)*c['landing_delta']
    limits = r.m.jnt_range[1:][PROX]
    bounds = [(-.25,.7)] + [(max(limits[2*i+1,0],home[3*i+1]-.25),
                                  min(limits[2*i+1,1],home[3*i+1]+.25)) for i in range(4)]
    for offset in (5,13):
        for j in range(8):
            radius = .65 if j%2 == 0 else .4
            bounds.append((max(limits[j,0],x[offset+j]-radius), min(limits[j,1],x[offset+j]+radius)))
    bounds = np.asarray(bounds)
    x = np.clip(x, bounds[:,0], bounds[:,1])
    rng = np.random.default_rng(seed)
    pop = np.clip(x + rng.normal(size=(population,21))*(bounds[:,1]-bounds[:,0])*.18,
                  bounds[:,0], bounds[:,1]); pop[0] = x
    total = 0; count = 0; best = None; begin = time.monotonic()
    config = dict(source_hashes=provenance(), environment=versions(), leg=leg,
                  direction=direction, seed=seed, generations=generations,
                  population=population, workers=workers, bounds=bounds.tolist(),
                  start_state_sha256=hashlib.sha256(json.dumps(start,sort_keys=True).encode()).hexdigest(),
                  search_cad_period_s=.25, search_cad_scope='all detailed shin/motor/body pairs',
                  simulation_only=True, hardware_validated=False)
    (output/'config.json').write_text(json.dumps(config,indent=2))
    def record(items):
        nonlocal total, count, best
        costs = []
        with (output/'evaluations.jsonl').open('a') as f:
            for item in items:
                count += 1; total += item['environment_steps']; item['evaluation'] = count
                f.write(json.dumps(item)+'\n'); costs.append(item['cost'])
                if best is None or item['cost'] < best['cost']:
                    best = item
                    pre, lift, land = decode(np.asarray(item['x']),home,index,direction)
                    c = dict(status='UNACCEPTED_EXTENDED_PHYSICS_CANDIDATE', leg=leg,
                             model_sha256=r.manifest['model_sha256'], full_pose=lift.tolist(),
                             proximal_pose=lift[PROX].tolist(), pre_shift_pose=pre.tolist(),
                             landing_pose=land.tolist(), direction=direction, landing_delta=0.,
                             search_vector=item['x'], start_state=start, measurements=item)
                    (output/'best_flip.json').write_text(json.dumps(c,indent=2))
                    with (output/'improvements.jsonl').open('a') as out:out.write(json.dumps(item)+'\n')
                    print(json.dumps({k:v for k,v in item.items() if k != 'x'}),flush=True)
        return costs
    with mp.get_context('spawn').Pool(workers, initializer=initialize,
                                    initargs=(start,index,direction)) as pool:
        def mapper(function, values):
            return record(pool.map(evaluate,list(values)))
        result = differential_evolution(lambda v: 0., bounds, init=pop, seed=seed,
                                        maxiter=generations, polish=False, workers=mapper,
                                        updating='deferred', tol=1e-4, mutation=(.4,1.), recombination=.8)
    initialize(start,index,direction)
    def local(v):return record([evaluate(v)])[0]
    result_local = minimize(local, np.asarray(best['x']), method='Powell', bounds=bounds,
                            options=dict(maxiter=4,maxfev=700,xtol=.004,ftol=.001))
    report = dict(environment_steps=total,evaluations=count,seconds=time.monotonic()-begin,
                  best=best,solver_success=bool(result.success),message=str(result.message),
                  polish_message=str(result_local.message))
    (output/'search.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=Path,required=True)
    p.add_argument('--leg',choices=LEGS,required=True);p.add_argument('--direction',type=int,choices=[-1,1],default=-1)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--candidate',type=Path)
    p.add_argument('--generations',type=int,default=30);p.add_argument('--population',type=int,default=96);p.add_argument('--workers',type=int,default=24)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    config=dict(simulation_only=True,source_hashes=provenance(),environment=versions(),
                arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    logger=ExperimentLogger(a.output,config);exit_code=0
    try:
        start=json.loads(a.start.read_text())
        report=optimize(a.output/'search',start,a.leg,a.direction,a.seed,a.candidate,a.generations,a.population,a.workers)
        candidate=a.output/'search/best_flip.json'
        audit=run(candidate,a.output/'replay',video=True,direction=a.direction,start_override=start)
        steps=report['environment_steps']+audit['environment_steps']
        logger.metrics(steps,{k:v for k,v in audit.items() if isinstance(v,(int,float,bool,str))})
        logger.video(a.output/'replay/rollout.mp4',steps,key='trajectory/'+a.leg,
                     caption=f'SIMULATION extended candidate {audit["candidate_sha256"]} | seed {a.seed} | {audit["status"]} | early termination {audit["early_termination"]}; no trained checkpoint')
        logger.run.summary.update(dict(simulation_audit_status=audit['status'],gates=audit['gates'],hardware_validated=False))
    except BaseException:
        import traceback
        (a.output/'error.txt').write_text(traceback.format_exc());exit_code=1
        raise
    finally:
        artifact=logger.sdk.Artifact('inverted-extended-'+logger.state['id'],type='trajectory-run')
        for path in sorted(a.output.rglob('*')):
            if path.is_file() and path.suffix in ['.json','.jsonl','.txt'] and 'wandb' not in path.relative_to(a.output).parts:
                artifact.add_file(str(path),name=path.relative_to(a.output).as_posix())
        logger.run.log_artifact(artifact);logger.finish(exit_code)
        (a.output/'logging_status.json').write_text(json.dumps(dict(url=logger.state['url'],status='SDK finish completed; cloud verification pending',exit_code=exit_code),indent=2))

if __name__ == '__main__':main()
