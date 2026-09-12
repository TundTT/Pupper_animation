"""Refine CAD gaps and render multiple views of recorded nominal physics.

Pose assignments below are geometry/visual review only. Before using them, a
fresh integrated replay must match the saved end state, including velocities.
"""
import argparse
import hashlib
import json
from pathlib import Path

import fcl
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

from .core import Robot, LEGS, provenance, duration, smooth
from .clearance import CADClearance
from .simulate import trajectory


def review(directory, output):
    directory=Path(directory);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    audit=json.loads((directory/'audit.json').read_text())
    candidate=json.loads((directory/'candidate.json').read_text())
    start=json.loads((directory/'start_state.json').read_text())
    expected=json.loads((directory/'end_state.json').read_text())
    r=Robot(friction=audit['friction'],dynamics=audit.get('effective_dynamics'))
    home=r.restore(start);leg=LEGS.index(candidate['leg']);cad=CADClearance(r)
    states=[(float(r.d.time),'initial',r.d.qpos.copy())];phase_min={};pair_min={};coarse=[]
    def measure(frame):
        t,phase,q=frame;r.d.qpos[:]=q;r.d.time=t;mj.mj_kinematics(r.m,r.d)
        c=cad.measure();c.update(time_s=t,phase=phase)
        for side in ['r','l']:
            a='leg_front_'+side+'_3';b='leg_back_'+side+'_3'
            hit=any(set(pair)=={a,b} for pair in c['intersections'])
            gap=0. if hit else float(fcl.distance(cad.objects[a],cad.objects[b],fcl.DistanceRequest(),fcl.DistanceResult()))
            if gap<pair_min.get(side,{}).get('minimum_m',1.):pair_min[side]=dict(minimum_m=gap,time_s=t,phase=phase)
        return c
    previous=home.copy();tick=0;max_lift=None;half=None
    phases=trajectory(home,np.asarray(candidate['full_pose']),leg,audit['landing_delta'],audit['direction'],pre_shift_pose=candidate.get('pre_shift_pose'),landing_pose=candidate.get('landing_pose'),touchdown_pose=candidate.get('touchdown_pose'))
    for phase,target,seconds in phases:
        steps=int(np.ceil(duration(previous,target,seconds)/r.m.opt.timestep))
        for k in range(steps):
            r.tick(previous+smooth((k+1)/steps)*(target-previous));tick+=1
            frame=(float(r.d.time),phase,r.d.qpos.copy());states.append(frame)
            if phase in ['lift','clearance_hold']:
                bottom=float(r.bottoms()[leg])
                if max_lift is None or bottom>max_lift[0]:max_lift=(bottom,frame)
            if phase=='rotate' and k==steps//2:half=frame
            if tick%52==0:
                c=measure(frame);coarse.append(c)
                if c['minimum_m']<phase_min.get(phase,{}).get('minimum_m',1.):phase_min[phase]=c
        previous=target.copy()
    actual=r.snapshot(previous)
    error=float(np.max(np.abs(np.asarray(actual['state'])-expected['state'])))
    if error>1e-7:raise ValueError(f'Review replay diverged from saved integration state: {error}')
    centers=[v['time_s'] for v in phase_min.values()]+[v['time_s'] for v in pair_min.values()]
    refined=[]
    for frame in states:
        if any(abs(frame[0]-t)<=.125 for t in centers):refined.append(measure(frame))
    best=min(coarse+refined,key=lambda c:c['minimum_m'])
    nearest=None
    frame=min(states,key=lambda f:abs(f[0]-best['time_s']));measure(frame)
    if best['minimum_m']>0:
        a,b=best['nearest_pair'];result=fcl.DistanceResult()
        gap=fcl.distance(cad.objects[a],cad.objects[b],fcl.DistanceRequest(enable_nearest_points=True),result)
        nearest=[np.asarray(p).tolist() for p in result.nearest_points]
        if abs(np.linalg.norm(np.asarray(nearest[0])-nearest[1])-gap)>1e-7:raise ValueError('Nearest points inconsistent')
    frames=[states[0],max_lift[1],half,states[-1]]
    r.m.vis.global_.offwidth=640;r.m.vis.global_.offheight=420
    renderer=mj.Renderer(r.m,height=420,width=640);sheet=Image.new('RGB',(1920,1680))
    for row,frame in enumerate(frames):
        t,phase,q=frame;r.d.qpos[:]=q;mj.mj_forward(r.m,r.d)
        for col,azimuth in enumerate([40,140,230]):
            camera=mj.MjvCamera();camera.lookat[:]=r.d.qpos[:3];camera.distance=.7;camera.azimuth=azimuth;camera.elevation=-20
            renderer.update_scene(r.d,camera=camera);img=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(img);draw.rectangle((0,0,640,38),fill='black')
            draw.text((8,6),f'RECORDED SIMULATION | {candidate["leg"]} | {phase} | t={t:.3f}s',fill='white')
            draw.text((8,21),f'view {azimuth} deg | no hardware validation',fill='white');sheet.paste(img,(640*col,420*row))
    renderer.close();sheet.save(output/'views.png')
    result=dict(simulation_only=True,hardware_validated=False,source_hashes=provenance(),
                model_sha256=audit['model_sha256'],candidate_sha256=audit['candidate_sha256'],
                replay_audit_sha256=hashlib.sha256((directory/'audit.json').read_bytes()).hexdigest(),
                integrated_steps=tick,end_state_max_absolute_error=error,
                coarse_sample_period_s=.1,refined_sample_period_s=float(r.m.opt.timestep),
                refinement_half_window_s=.125,refinement_centers_s=centers,
                coarse_samples=len(coarse),refined_samples=len(refined),minimum=best,
                nearest_points_world_m=nearest,same_side_front_rear=pair_min,
                phase_coarse_minima=phase_min,
                limitations='Discrete CAD samples around phase minima and front/rear minima; no continuous-time collision proof. Rigid nominal CAD omits polymer compliance and spacer mass.')
    (output/'cad_review.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(stage=candidate['leg'],minimum=best,front_rear=pair_min)),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    review(a.replay,a.output)

if __name__=='__main__':main()
