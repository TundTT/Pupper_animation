"""CPU sensitivity optimization; coarse selection is never an acceptance audit.

Search only consumes simulation truth in the objective, not in its controller.
Each context integrates the complete preceding plan, without nominal resets.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import mujoco as mj
import numpy as np
from scipy.optimize import differential_evolution, minimize
from .core import Robot, LEGS, HUB, PROX, duration, smooth, provenance, versions
from .simulate import trajectory, run
from .clearance import CADClearance
from .robustness import conditions
from training.wandb_logging import ExperimentLogger

W = None

def phases(home, c):
    return trajectory(home, np.asarray(c['full_pose']), LEGS.index(c['leg']),
                      c['landing_delta'], c['direction'],
                      pre_shift_pose=c.get('pre_shift_pose'), landing_pose=c.get('landing_pose'),
                      touchdown_pose=c.get('touchdown_pose'))

def integrate(r, previous, phase_list):
    count = 0
    for phase, target, seconds in phase_list:
        n = int(np.ceil(duration(previous, target, seconds) / r.m.opt.timestep))
        for k in range(n):
            r.tick(previous + smooth((k+1)/n)*(target-previous))
        count += n
        previous = target.copy()
    return previous, count

def contexts(plan, stage, landing):
    # Fixed training sensitivities; held-out cases are defined separately.
    names = ['mu-0.5-seed-1', 'mu-0.8-seed-1', 'mu-1.0-seed-1',
             'mass-low', 'mass-high', 'gains-low', 'gains-high', 'combined-seed-22']
    jobs = [j for j in conditions() if j['name'] in names]
    data = json.loads(plan.read_text()); candidates=[]
    for entry in data['stages']:
        p=plan.parent/entry['candidate']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['candidate_sha256']:
            raise ValueError('Plan hash mismatch')
        candidates.append(json.loads(p.read_text()))
    states=[];steps=0
    for job in jobs:
        r=Robot(job['friction'],job['scenario'].get('dynamics'))
        home=r.initial[7:].copy()
        r.d.qpos[7:][PROX]+=np.random.default_rng(job['seed']).uniform(-.01,.01,8)
        offset=job['scenario'].get('initial_offset',{})
        roll=offset.get('roll_rad',0.);pitch=offset.get('pitch_rad',0.)
        r.d.qpos[2]+=offset.get('height_m',0.)
        quat=np.array([np.cos(roll/2)*np.cos(pitch/2),np.sin(roll/2)*np.cos(pitch/2),np.cos(roll/2)*np.sin(pitch/2),-np.sin(roll/2)*np.sin(pitch/2)])
        q=np.empty(4);mj.mju_mulQuat(q,r.d.qpos[3:7].copy(),quat);r.d.qpos[3:7]=q
        mj.mj_forward(r.m,r.d)
        for c in candidates[:stage]:
            home,n=integrate(r,home,phases(home,c));steps+=n
        start=r.snapshot(home)
        if landing:
            prefix=[]
            for item in phases(home,candidates[stage]):
                if item[0]=='land':break
                prefix.append(item)
            home,n=integrate(r,home,prefix);steps+=n
        states.append(dict(job=job,start=start,state=r.snapshot(home)))
    return candidates[stage],states,steps

def decode(x, candidate, mode):
    c=dict(candidate);leg=LEGS.index(c['leg']);lift=np.array(c['full_pose'])
    if mode=='lift':
        # Keep the parent's final command exactly; change support transfer/lift only.
        if 'landing_pose' not in c:
            land=lift.copy();land[HUB[leg]]+=c['direction']*np.pi
            land[3*leg+1]-=(1 if leg%2==0 else -1)*c['landing_delta']
            c['landing_pose']=land.tolist()
        lift[PROX]=x;c['full_pose']=lift.tolist();c['proximal_pose']=lift[PROX].tolist()
    else:
        touch=lift.copy();touch[HUB[leg]]+=c['direction']*np.pi
        touch[3*leg:3*leg+2]=x[:2]
        land=np.array(c['landing_pose']);land[PROX]=x[2:]
        c['touchdown_pose']=touch.tolist();c['landing_pose']=land.tolist()
    c['status']='UNACCEPTED_SENSITIVITY_CANDIDATE'
    c.pop('start_state',None)
    return c

def initialize(states,candidate,mode):
    global W
    rows=[]
    for state in states:
        r=Robot(state['job']['friction'],state['job']['scenario'].get('dynamics'))
        rows.append((r,state,CADClearance(r)))
    W=(rows,candidate,mode)

def evaluate(x):
    rows,candidate,mode=W;c=decode(x,candidate,mode);leg=LEGS.index(c['leg'])
    results=[];total=0
    for r,context,cad in rows:
        home=r.restore(context['state']);previous=home.copy()
        ps=phases(np.array(context['start']['command']),c)
        if mode=='landing':ps=ps[next(i for i,p in enumerate(ps) if p[0]=='land'):]
        floor=tilt=torque=contact=descent=0.;gap=cad_gap=1.;support=1e6
        support_loss=0.;samples=0;last=None;last_time=None;tick=0;early=False
        for phase,target,seconds in ps:
            n=int(np.ceil(duration(previous,target,seconds)/r.m.opt.timestep))
            for k in range(n):
                tau=r.tick(previous+smooth((k+1)/n)*(target-previous));tick+=1
                torque=max(torque,float(np.abs(tau).max()))
                if tick%13:continue
                f=r.contacts_precise();b=float(r.bottoms()[leg]);now=float(r.d.time)
                floor=max(floor,r.unintended_floor_force());tilt=max(tilt,float(np.rad2deg(r.tilt())))
                if phase in ['rotate','angle_hold']:
                    gap=min(gap,b);support=min(support,float(np.delete(f,leg).min()));contact=max(contact,float(f[leg]))
                if phase=='land':
                    if last is not None and b<.008:descent=max(descent,(last-b)/(now-last_time))
                    # Penalize loss of a support before touchdown, preventing diagonal tipping.
                    if f[leg]<2.:support_loss+=float(np.maximum(1.5-np.delete(f,leg),0).sum());samples+=1
                last=b;last_time=now
                if tick%130==0:cad_gap=min(cad_gap,cad.measure()['minimum_m'])
                if tilt>20 or r.d.qpos[2]<.045:early=True;break
            previous=target.copy()
            if early:break
        total+=tick;f=r.contacts_precise();tip=abs(float(r.tip_bottoms()[leg]))
        error=abs(float(r.d.qpos[7+HUB[leg]]-previous[HUB[leg]]));speed=float(np.abs(r.d.qvel[6:]).max())
        cost=1000.*early+max(.008-gap,0)**2*2e6+max(1.5-support,0)**2*20
        cost+=max(contact-.05,0)**2*20+max(floor-.01,0)**2*100
        cost+=max(.004-cad_gap,0)**2*3e6+max(tilt-6.,0)**2*5
        cost+=max(descent-.018,0)**2*3e5+support_loss/max(samples,1)*10
        cost+=float(np.maximum(2.-f,0).dot(np.maximum(2.-f,0)))*30
        cost+=max(error-.022,0)**2*5e4+max(tip-.001,0)**2*2e5+max(speed-.04,0)**2*50+max(torque-2.8,0)**2*20
        results.append(dict(name=context['job']['name'],cost=float(cost),rotation_floor_m=gap,support_N=support,floor_N=floor,cad_gap_m=cad_gap,descent_m_s=descent,pre_touchdown_support_deficit=support_loss/max(samples,1),tilt_deg=tilt,hub_error_rad=error,final_force_N=f.tolist(),tip_m=tip,speed_rad_s=speed,early_termination=early))
    costs=[r['cost'] for r in results]
    return dict(x=np.asarray(x).tolist(),cost=float(max(costs)+np.mean(costs)),conditions=results,environment_steps=total)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--stage',type=int,required=True,choices=range(1,5));p.add_argument('--mode',choices=['lift','landing'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--generations',type=int,default=12);p.add_argument('--population',type=int,default=48);p.add_argument('--workers',type=int,default=24);p.add_argument('--seed',type=int,default=41);p.add_argument('--radius',type=float,default=.08)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    c,states,steps=contexts(a.plan,a.stage-1,a.mode=='landing')
    config=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},source_hashes=provenance(),environment=versions(),simulation_only=True,parent_plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),selection_only=True,search_cad_period_s=.25,search_physics_period_s=.025)
    (out/'config.json').write_text(json.dumps(config,indent=2));(out/'contexts.json').write_text(json.dumps(states,indent=2))
    logger=ExperimentLogger(out,config)
    limits=Robot().m.jnt_range[1:][PROX];leg=LEGS.index(c['leg'])
    if a.mode=='lift':
        x=np.asarray(c['full_pose'])[PROX];bounds=np.c_[np.maximum(limits[:,0],x-a.radius),np.minimum(limits[:,1],x+a.radius)]
    else:
        x=np.r_[np.asarray(c['full_pose'])[3*leg:3*leg+2],np.asarray(c['landing_pose'])[PROX]]
        x[1]+=(1 if leg%2 else -1)*.3
        lim=np.r_[limits[2*leg:2*leg+2],limits]
        radius=np.r_[.25,.3,np.repeat(a.radius,8)]
        bounds=np.c_[np.maximum(lim[:,0],x-radius),np.minimum(lim[:,1],x+radius)]
    config['bounds']=bounds.tolist();(out/'config.json').write_text(json.dumps(config,indent=2))
    rng=np.random.default_rng(a.seed);pop=np.clip(x+rng.normal(size=(a.population,len(x)))*(bounds[:,1]-bounds[:,0])*.18,bounds[:,0],bounds[:,1]);pop[0]=x
    best=None;count=0;begin=time.monotonic()
    def record(items):
        nonlocal best,count,steps
        values=[]
        with (out/'evaluations.jsonl').open('a') as f:
            for item in items:
                count+=1;steps+=item['environment_steps'];item['evaluation']=count;f.write(json.dumps(item)+'\n');values.append(item['cost'])
                if best is None or item['cost']<best['cost']:
                    best=item;candidate=decode(np.asarray(item['x']),c,a.mode);candidate['search_measurements']=item
                    (out/'candidate.json').write_text(json.dumps(candidate,indent=2));print(json.dumps(item),flush=True)
        return values
    with mp.get_context('spawn').Pool(a.workers,initializer=initialize,initargs=(states,c,a.mode)) as pool:
        def mapper(function,values):return record(pool.map(evaluate,list(values)))
        differential_evolution(lambda x:0,bounds,init=pop,seed=a.seed,maxiter=a.generations,workers=mapper,updating='deferred',polish=False,tol=1e-5)
    initialize(states,c,a.mode)
    minimize(lambda x:record([evaluate(x)])[0],np.asarray(best['x']),method='Powell',bounds=bounds,options=dict(maxiter=2,maxfev=100,xtol=.001,ftol=.001))
    (out/'search.json').write_text(json.dumps(dict(evaluations=count,environment_steps=steps,seconds=time.monotonic()-begin,best=best),indent=2))
    # Independent full-stage audit with detailed CAD and an actual physics video.
    context=next(s for s in states if s['job']['name']=='mu-0.8-seed-1')
    audit=run(out/'candidate.json',out/'replay',video=True,landing_delta=c['landing_delta'],direction=c['direction'],start_override=context['start'])
    steps+=audit['environment_steps'];logger.metrics(steps,{k:v for k,v in audit.items() if isinstance(v,(int,float,bool,str))})
    logger.video(out/'replay/rollout.mp4',steps,key='trajectory/'+c['leg'],caption=f'SIMULATION sensitivity candidate | {audit["status"]} | {audit["candidate_sha256"]} | actual integrated stage from continuous training prefix; not a full sequence claim')
    logger.run.summary.update(dict(simulation_audit_status=audit['status'],gates=audit['gates'],hardware_validated=False))
    artifact=logger.sdk.Artifact('sensitivity-search-'+logger.state['id'],type='trajectory-run')
    for file in out.rglob('*'):
        if file.is_file() and file.suffix in ['.json','.jsonl'] and 'wandb' not in file.relative_to(out).parts:artifact.add_file(str(file),name=file.relative_to(out).as_posix())
    logger.run.log_artifact(artifact);logger.finish()
    (out/'logging_status.json').write_text(json.dumps(dict(url=logger.state['url'],status='SDK finish completed; cloud verification pending'),indent=2))

if __name__=='__main__':main()
