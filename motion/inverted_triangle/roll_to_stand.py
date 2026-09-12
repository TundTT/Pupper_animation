"""Simulation-only coordinated roll feasibility probes. Never connects to hardware.

These are new experiments on the backpack model, not replays of the old accepted
four-flip plan. A probe pass is not walking-policy handoff validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco as mj
import numpy as np

from .core import Robot, HUB, PROX, LEGS, KP, KD, SPEED, ACCEL, smooth, HERE, provenance, versions
from .clearance import CADClearance

ROOT = HERE.parents[1]
POLICY = ROOT/'ros2_ws/src/neural_controller/launch/policy_walk_v2.json'


def initialize(direction='forward'):
    r = Robot()
    goal = np.array(json.loads(POLICY.read_text())['default_joint_pos'])
    # World hub axes are mirrored. Choose a fixed initial winding that reaches
    # the policy's UNWRAPPED default; never wrap/teleport any running trajectory.
    signs = np.array([-1., 1., -1., 1.])
    if direction == 'backward': signs *= -1
    home = r.initial[7:].copy()
    home[HUB] = goal[HUB] - signs*np.pi
    before = r.d.xpos.copy()
    r.d.qpos[7:] = home
    mj.mj_forward(r.m, r.d)
    assert np.max(abs(before-r.d.xpos)) < 1e-10
    r.initial = r.d.qpos.copy()
    return r, home, goal


def commands(home, goal, seconds, shoulder_bump=0., splay_power=1.):
    count = int(np.ceil(seconds*520))
    u = np.arange(1,count+1)/count
    s = smooth(u)
    out = home + s[:,None]*(goal-home)
    for leg in range(4):
        side = 1 if leg%2 == 0 else -1
        out[:,leg*3] += side*shoulder_bump*64*u**3*(1-u)**3
        out[:,leg*3+1] = home[leg*3+1] + smooth(u**splay_power)*(goal[leg*3+1]-home[leg*3+1])
    return np.concatenate([np.tile(home,(1040,1)),out,np.tile(goal,(2600,1))])


def render_states(r, states, output, title):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    r.m.vis.global_.offwidth=960; r.m.vis.global_.offheight=640
    # Preserve exact CAD. Highlight terminal shins and tip sites, rather than
    # replacing the accepted mesh with an invented triangular visual shape.
    colors=((.95,.45,.12,1),(.2,.65,.95,1),(.95,.7,.15,1),(.3,.8,.6,1))
    for leg,color in zip(LEGS,colors): r.m.geom_rgba[r.m.geom(leg+'_shin_visual').id]=color
    renderer=mj.Renderer(r.m,height=640,width=960)
    camera=mj.MjvCamera(); camera.distance=.7; camera.azimuth=100; camera.elevation=-18
    try:
        with imageio.get_writer(str(output),fps=20,macro_block_size=16) as writer:
            for t,q in states:
                r.d.qpos[:]=q; mj.mj_forward(r.m,r.d)
                camera.lookat[:]=r.d.qpos[:3]; camera.lookat[2]=.1
                renderer.update_scene(r.d,camera=camera)
                im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,960,43),fill='black')
                draw.text((8,5),f'SIMULATION ONLY | {title} | t={t:.2f}s',fill='white')
                draw.text((8,23),'9 mm gap + backpack | exact CAD highlighted | no hardware or walking handoff',fill='white')
                writer.append_data(np.asarray(im))
    finally: renderer.close()


def probe(config, output, video=False, cad=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    r,home,goal=initialize(config['direction'])
    targets=commands(home,goal,config['seconds'],config['shoulder_bump'],config['splay_power'])
    velocity=np.diff(np.vstack([home,targets]),axis=0)*520
    acceleration=np.diff(np.vstack([np.zeros(12),velocity]),axis=0)*520
    limits=r.m.jnt_range[1:][PROX]
    assert np.all((targets[:,PROX]>=limits[:,0])&(targets[:,PROX]<=limits[:,1])), 'command limits'
    report=dict(config=config,simulation_only=True,hardware_validated=False,walking_handoff_validated=False,
        model_sha256=r.manifest['model_sha256'],policy_sha256=hashlib.sha256(POLICY.read_bytes()).hexdigest(),
        goal_joint_positions=goal.tolist(),initial_joint_positions=home.tolist(),
        initial_winding='Constant whole turns selected before simulation, same physical point-up pose; no runtime wrapping',
        max_command_speed_ratio=float(np.max(abs(velocity)/SPEED)),
        max_command_acceleration_ratio=float(np.max(abs(acceleration)/ACCEL)),
        max_tilt_deg=0.,peak_requested_torque_Nm=0.,max_unintended_floor_force_N=0.,
        max_measured_joint_speed=0.,minimum_cad_gap_m=None,first_cad_intersections=None,
        cad_sample_period_s=.1 if cad else None,early_termination=None)
    checker=CADClearance(r) if cad else None
    trace=[];states=[];hold=[]
    for step,target in enumerate(targets):
        tau=r.tick(target)
        report['peak_requested_torque_Nm']=max(report['peak_requested_torque_Nm'],float(abs(tau).max()))
        report['max_measured_joint_speed']=max(report['max_measured_joint_speed'],float(abs(r.d.qvel[6:]).max()))
        if step%13==0:
            tilt=float(np.rad2deg(r.tilt()));floor=r.unintended_floor_force()
            report['max_tilt_deg']=max(report['max_tilt_deg'],tilt)
            report['max_unintended_floor_force_N']=max(report['max_unintended_floor_force_N'],floor)
            row=dict(time=float(r.d.time),tilt_deg=tilt,base_height_m=float(r.d.qpos[2]),
                policy_pose_error_rad=float(abs(r.d.qpos[7:]-goal).max()),
                floor_force_N=floor,tip_bottom_m=r.tip_bottoms().tolist(),forces_N=r.contacts_precise().tolist(),
                joint_speed=float(abs(r.d.qvel[6:]).max()),base_speed=float(np.linalg.norm(r.d.qvel[:3])))
            trace.append(row)
            if step>len(targets)-1560:hold.append(row)
            if cad and step%52==0:
                c=checker.measure()
                previous_gap=report['minimum_cad_gap_m']
                report['minimum_cad_gap_m']=c['minimum_m'] if previous_gap is None else min(previous_gap,c['minimum_m'])
                if c['intersections'] and report['first_cad_intersections'] is None:
                    report['first_cad_intersections']={'time':float(r.d.time),'pairs':c['intersections']}
            if tilt>35 or r.d.qpos[2]<.045 or not np.isfinite(r.d.qpos).all():
                report['early_termination']='large tilt, collapse, or nonfinite state'
                states.append((float(r.d.time),r.d.qpos.copy())); break
        if step%26==0:states.append((float(r.d.time),r.d.qpos.copy()))
    report['environment_steps']=step+1
    report['simulated_seconds']=float(r.d.time)
    report['final']=trace[-1]
    report['final_joint_positions']=r.d.qpos[7:].tolist()
    gates={'completed':step+1==len(targets),
        'command_rate':report['max_command_speed_ratio']<=1.000001 and report['max_command_acceleration_ratio']<=1.000001,
        'tilt':report['max_tilt_deg']<=8.,'torque':report['peak_requested_torque_Nm']<=3.,
        'motor_body_floor':report['max_unintended_floor_force_N']<=.01,
        'measured_speed':report['max_measured_joint_speed']<=2.,
        'three_second_walk_pose':len(hold)>=119 and max(x['policy_pose_error_rad'] for x in hold)<=.1,
        'three_second_tip_support':len(hold)>=119 and all(max(map(abs,x['tip_bottom_m']))<=.003 and min(x['forces_N'])>=1 for x in hold),
        'three_second_settled':len(hold)>=119 and all(x['joint_speed']<=.1 and x['base_speed']<=.03 for x in hold),
        'sampled_cad':None if not cad else report['minimum_cad_gap_m']>=.001 and report['first_cad_intersections'] is None}
    report['gates']=gates
    report['status']='PROBE_PASS_REQUIRES_DENSE_AUDIT_AND_POLICY_HANDOFF' if all(v is True for v in gates.values()) else 'FAILED_OR_INCOMPLETE'
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'trace.json').write_text(json.dumps(trace)+'\n')
    np.savez_compressed(output/'states.npz',times=[s[0] for s in states],qpos=[s[1] for s in states])
    if video:render_states(r,states,output/'rollout.mp4',config['name']+' | '+report['status'])
    print(json.dumps({k:report[k] for k in ['config','status','max_tilt_deg','max_unintended_floor_force_N','simulated_seconds','gates']}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--video',action='store_true');p.add_argument('--cad',action='store_true')
    p.add_argument('--wandb',choices=['online','offline','disabled'],default='online')
    p.add_argument('--config',type=Path)
    a=p.parse_args()
    if a.wandb!='disabled' and not a.video:p.error('Logged probes require actual video')
    configs=json.loads(a.config.read_text()) if a.config else [
        dict(name='all_forward_direct',direction='forward',seconds=12.,shoulder_bump=0.,splay_power=1.),
        dict(name='all_backward_direct',direction='backward',seconds=12.,shoulder_bump=0.,splay_power=1.),
        dict(name='all_forward_raise_early',direction='forward',seconds=12.,shoulder_bump=-.3,splay_power=.7),
        dict(name='all_forward_raise_late',direction='forward',seconds=12.,shoulder_bump=.3,splay_power=1.5)]
    a.output.mkdir(parents=True,exist_ok=False)
    setup=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
        source_hashes=provenance(),environment=versions(),configs=configs,
        task='New coordinated ground roll to leg policy default; exploratory feasibility, not training')
    (a.output/'provenance.json').write_text(json.dumps(setup,indent=2)+'\n')
    logger=None
    if a.wandb!='disabled':
        from training.wandb_logging import ExperimentLogger
        logger=ExperimentLogger(a.output,setup,mode=a.wandb,name=a.output.name)
    reports=[];steps=0
    try:
        for config in configs:
            path=a.output/config['name'];report=probe(config,path,a.video,a.cad);reports.append(report)
            steps+=report['environment_steps']
            if logger:
                logger.metrics(steps,{'probe/'+config['name']+'/status':report['status']})
                logger.video(path/'rollout.mp4',steps,key='trajectory/'+config['name'],caption=f'SIMULATION probe {config}; {report["status"]}; early termination {report["early_termination"]}; no checkpoint or hardware')
        (a.output/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
        if logger:
            artifact=logger.sdk.Artifact('roll-to-stand-'+logger.state['id'],type='trajectory-probe')
            for path in a.output.rglob('*'):
                if path.is_file() and path.suffix in ('.json','.npz') and 'wandb' not in path.parts:
                    artifact.add_file(str(path),name=path.relative_to(a.output).as_posix())
            logger.run.log_artifact(artifact)
    finally:
        if logger:logger.finish()


if __name__=='__main__':main()
