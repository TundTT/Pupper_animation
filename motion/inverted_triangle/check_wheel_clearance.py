"""Detailed wheel/shin pair distances, with refinement near closest approaches.

This replays torque-limited dynamics. It does not enable self-contact forces or
claim hardware clearance. Original nonconvex visual CAD is checked with FCL.
"""
import argparse,csv,hashlib,itertools,json,subprocess
from pathlib import Path
import fcl
import mujoco as mj
import numpy as np
from .core import HERE,Robot,LEGS,provenance,versions,duration,smooth
from .clearance import CADClearance
from .simulate import trajectory


def run(candidate,output):
    candidate=Path(candidate);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=json.loads(candidate.read_text());r=Robot();home=r.initial[7:].copy()
    if source['model_sha256']!=r.manifest['model_sha256']:raise ValueError('Candidate model hash mismatch')
    if 'start_state' in source:home=r.restore(source['start_state'])
    leg=LEGS.index(source['leg']);lift=np.array(source['full_pose'])
    cad=CADClearance(r)
    pairs=list(itertools.combinations(LEGS,2))
    best={a+'__'+b:dict(minimum_m=float('inf'),intersection_samples=0,phase_minima_m={}) for a,b in pairs}
    rows=[];tick=0;states=[];checks=0

    def check(phase):
        nonlocal checks
        checks+=1
        # mj_step advances qpos; refresh transforms to match the recorded time.
        # This changes neither qpos/qvel nor the integration warm start.
        mj.mj_kinematics(r.m,r.d)
        for name in LEGS:
            obj=cad.objects['leg_'+name+'_3'];bid=cad.body_ids['leg_'+name+'_3']
            obj.setTransform(fcl.Transform(r.d.xmat[bid].reshape(3,3),r.d.xpos[bid]))
        values=[]
        for a,b in pairs:
            key=a+'__'+b;record=best[key]
            oa=cad.objects['leg_'+a+'_3'];ob=cad.objects['leg_'+b+'_3']
            hit=fcl.collide(oa,ob,fcl.CollisionRequest(num_max_contacts=1),fcl.CollisionResult())>0
            result=fcl.DistanceResult()
            gap=0. if hit else float(fcl.distance(oa,ob,fcl.DistanceRequest(enable_nearest_points=True),result))
            if not np.isfinite(gap) or gap<0:raise ValueError('Invalid CAD distance')
            record['intersection_samples']+=int(hit)
            record['phase_minima_m'][phase]=min(record['phase_minima_m'].get(phase,float('inf')),gap)
            if gap<record['minimum_m']:
                points=None if hit else [np.asarray(p).tolist() for p in result.nearest_points]
                if points is not None and abs(np.linalg.norm(np.array(points[0])-points[1])-gap)>1e-7:
                    raise ValueError('Nearest-point distance inconsistency')
                record.update(minimum_m=gap,time_s=float(r.d.time),phase=phase,nearest_points_world_m=points,qpos=r.d.qpos.tolist())
            values.append(gap)
        if tick%10==0:rows.append([float(r.d.time),phase]+values)

    check('initial')
    previous=home.copy()
    for phase,target,seconds in trajectory(home,lift,leg,source.get('landing_delta',.3),source.get('direction',-1)):
        steps=int(np.ceil(duration(previous,target,seconds)/r.m.opt.timestep))
        for k in range(steps):
            r.tick(previous+smooth((k+1)/steps)*(target-previous));tick+=1
            states.append((tick,float(r.d.time),phase,r.d.qpos.copy()))
            if tick%13==0:check(phase)
        print(f'{phase}: t={r.d.time:.3f}s, right front/rear={best["front_r__back_r"]["minimum_m"]*1000:.3f} mm, left front/rear={best["front_l__back_l"]["minimum_m"]*1000:.3f} mm',flush=True)
        previous=target.copy()
    check('planted_hold');final_state=r.snapshot(previous);physics_steps=tick
    # Refine actual integrated poses around both same-side front/rear minima.
    # Assigning recorded qpos here is geometry analysis, not dynamic validation.
    centers=[best[key]['time_s'] for key in ['front_r__back_r','front_l__back_l']]
    for frame,time_s,phase,qpos in states:
        if frame%13 and any(abs(time_s-center)<=.125 for center in centers):
            tick=frame;r.d.qpos[:]=qpos;r.d.time=time_s;check(phase)
    report=dict(scope='All six pairs of complete rigid wheel/shin CAD meshes during the saved single flip; not motor assemblies or hardware',simulation_only=True,hardware_validated=False,model_sha256=r.manifest['model_sha256'],additional_gap_m=r.manifest['gap_m'],candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parents[1]).decode().strip(),source_hashes=provenance(),environment=versions(),physics_steps=physics_steps,samples_per_pair=checks,coarse_sample_period_s=13*float(r.m.opt.timestep),refined_sample_period_s=float(r.m.opt.timestep),refinement_half_window_s=.125,refinement_centers_s=centers,duration_s=final_state['state'][0],intersection_samples=sum(v['intersection_samples'] for v in best.values()),pairs=best,final_state=final_state,limitations='Discrete 40 Hz CAD sampling, refined at 520 Hz around both same-side front/rear closest approaches, using recorded nominal rigid dynamics. Not a continuous-time collision proof or a hardware tolerance/compliance audit.',wandb_upload_status='not attempted; focused local geometry audit, not training')
    (output/'audit.json').write_text(json.dumps(report,indent=2))
    with (output/'distances.csv').open('w',newline='') as out:
        writer=csv.writer(out);writer.writerow(['time_s','phase']+[a+'__'+b+'_m' for a,b in pairs]);writer.writerows(sorted(rows,key=lambda row:row[0]))
    print(json.dumps({key:{k:v for k,v in value.items() if k not in ['qpos','nearest_points_world_m','phase_minima_m']} for key,value in best.items()},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,default=HERE/'results/nominal_rear/candidate.json');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.candidate,a.output)
