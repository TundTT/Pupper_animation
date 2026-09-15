"""Native MuJoCo evaluation: four-leg cycles, 60 s holds, and mixed-stance sweeps.

Reports strict physical gates, independently of training reward. The unchanged
capsule/ring is a reach proxy; shortening does not simulate thermal deformation.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import jax
from jax import numpy as jp
import mujoco
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks
from brax.training.acme import running_statistics
from workspace import walk_geometry as geom
from workspace.leg_to_wheel_config import get_config, COMMAND_TO_FOOT, COMMAND_STATES
from workspace.leg_to_wheel_env import PupperLegToWheelEnv
from workspace.leg_to_wheel_sequencer import PolicySequencer
from workspace.leg_to_wheel_motion import limit_lowering_targets
from workspace.leg_to_wheel_randomize import domain_randomize, LEG_BODY_IDS


def native_model(env, randomized=None):
    m=copy.copy(env.mj_model)
    if randomized is not None:
        randomized,axes=randomized
        for name in ('geom_friction','actuator_gainprm','actuator_biasprm','body_mass','body_inertia',
                     'body_ipos','body_pos','geom_solref','geom_solimp','actuator_forcerange','dof_armature'):
            getattr(m,name)[:]=np.asarray(getattr(randomized,name))[0]
        mujoco.mj_setConst(m,mujoco.MjData(m))
    return m


def contacts(m,d,foot_ids,floor):
    pairs=d.contact.geom
    active=((pairs==floor).any(axis=1) & (d.contact.dist<=0) & (d.contact.efc_address>=0))
    hits=(pairs[:,None,:]==np.asarray(foot_ids)[None,:,None]).any(axis=-1)
    return (active[:,None] & hits).any(axis=0)


def frame(env,d,command,last_action,rng):
    rot=d.xmat[env.torso].reshape(3,3)
    result=np.concatenate([rot.T@d.cvel[env.torso,:3],rot.T@np.array([0.,0.,-1.]),
                           np.eye(5)[command],d.qpos[7:]-np.asarray(env.home),last_action])
    amplitude=np.array([env.config.sensor_noise]*6+[0.]*5+[env.config.sensor_noise]*12+[0.]*12)
    return result+rng.uniform(-1,1,35)*amplitude


def annotate(rgb, command, row):
    """Label the command and measured geometry directly on the rollout video."""
    from PIL import Image, ImageDraw, ImageFont
    canvas=Image.fromarray(rgb)
    draw=ImageDraw.Draw(canvas)
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:
        font=ImageFont.load_default()
    draw.rectangle((8,8,632,60),fill=(17,24,39))
    label=f'{COMMAND_STATES[command]}: lift / hold' if command else 'STAND / LOWER'
    draw.text((18,12),f"{label}   |   {row['time_s']:.1f} s",font=font,fill=(215,245,170))
    text=f"Tilt {row['tilt_deg']:.1f} deg   Height {row['height_m']*1000:.0f} mm"
    if command:text+=f"   Capsule gap {row['clearance_m']*1000:.0f} mm"
    draw.text((18,35),text,font=font,fill='white')
    return np.asarray(canvas)


def rollout(env,policy,seed,schedule,render=False,render_every=2,randomized=None,
            converted=(),shortening=.0075,progressive=False,diagnostics=False,lowering_joint_speed=0.,lowering_ease_seconds=.25):
    """Run continuously without episode resets. Never continue a fallen rollout.

    Command holds are external and may exceed training episode length. Progressive
    reach changes happen after the unloaded hold and before commanding touchdown.
    """
    if not np.isfinite(lowering_ease_seconds) or lowering_ease_seconds<0:
        raise ValueError('Lowering ease duration must be finite and nonnegative')
    c=env.config; m=native_model(env,randomized); d=mujoco.MjData(m)
    original_pos=m.body_pos[LEG_BODY_IDS].copy()
    def shorten(legs):
        for leg in legs:
            vec=original_pos[leg]
            m.body_pos[LEG_BODY_IDS[leg]]=vec*(1-shortening/np.linalg.norm(vec))
        mujoco.mj_setConst(m,mujoco.MjData(m))
    shorten(converted)
    mujoco.mj_resetDataKeyframe(m,d,0)
    rng=np.random.default_rng(seed)
    d.qpos[7:]+=rng.uniform(-c.reset_joint_noise,c.reset_joint_noise,12)
    d.qpos[:2]+=rng.uniform(-c.reset_xy_noise,c.reset_xy_noise,2)
    yaw=rng.uniform(-c.reset_yaw_noise,c.reset_yaw_noise)
    quaternion=np.empty(4)
    mujoco.mju_mulQuat(quaternion,np.array([np.cos(yaw/2),0.,0.,np.sin(yaw/2)]),d.qpos[3:7])
    d.qpos[3:7]=quaternion
    mujoco.mj_forward(m,d)
    start_xy=d.qpos[:2].copy(); previous=np.zeros(12); previous_target=np.zeros(12)
    obs=np.tile(frame(env,d,schedule[0][0],previous,rng),c.observation_history)
    renderer=mujoco.Renderer(m,height=480,width=640) if render else None
    cam=mujoco.MjvCamera();cam.distance=.65;cam.azimuth=200;cam.elevation=-22
    records=[]; frames=[]; elapsed=0; completed=set(converted); previous_command=0
    initial_height=None; segment_reports=[]; fell=False; stopped_reason=None
    sequencer=PolicySequencer(clearance_gate=c.lift_clearance_gate,converted=sum(1<<leg for leg in converted)) if progressive else None
    policy=jax.jit(policy)
    policy_key=jax.random.PRNGKey(seed)
    try:
        for segment,(command,seconds) in enumerate(schedule):
            if segment==1 and schedule[0][0]==0:
                start_xy=d.qpos[:2].copy()  # maneuver starts from the settled stance
            if progressive and previous_command and command==0:
                last=segment_reports[-1]
                if last['min_held_clearance_m']<c.lift_clearance_gate or last['max_held_stance_missing']:
                    stopped_reason='Unloaded clearance/support gate did not pass; conversion withheld'
                    break
                foot=COMMAND_TO_FOOT[previous_command]
                sequencer.update(0.,float(points[foot,2]),bool(contact[foot]),heating_confirmed=True)
                if sequencer.command!=0:
                    stopped_reason='Heating confirmation withheld: clearance was not stable'
                    break
                completed.add(foot);shorten(completed)
                mujoco.mj_forward(m,d)
            if sequencer is not None:
                if command and not sequencer.request(command):
                    stopped_reason='Previous leg did not settle before the next request'
                    break
                command=sequencer.command
            lowering_foot=COMMAND_TO_FOOT[previous_command] if command==0 else -1
            obs[6:11]=np.eye(5)[command]
            local=[]
            for tick in range(round(seconds/env.dt)):
                policy_key,key=jax.random.split(policy_key)
                action=np.clip(np.asarray(policy(jp.asarray(obs),key)[0]),-1.,1.)
                phase=min(tick*env.dt/lowering_ease_seconds,1.) if lowering_ease_seconds>0 else 0.
                strength=1.-3.*phase**2+2.*phase**3  # smooth release back to the policy
                if lowering_foot>=0:
                    current_bottom=geom.capsule_bottom(d.geom_xpos[env.foot_ids],d.geom_xmat[env.foot_ids],m.geom_size[env.foot_ids])[lowering_foot,2]
                    # Release fully within 20 mm so the policy owns touchdown.
                    strength*=np.clip((current_bottom-.02)/.02,0.,1.)
                applied=previous if rng.random()<c.latency_probability else action
                applied=limit_lowering_targets(applied,previous_target,lowering_foot,env.dt,env.action_scale,lowering_joint_speed,strength)
                # Smoothing belongs to actuation; the observation still records
                # the raw policy action, preserving the trained history layout.
                previous_target=applied
                d.ctrl[:]=np.clip(np.asarray(env.home)+applied*np.asarray(env.action_scale),
                                  m.jnt_range[1:,0],m.jnt_range[1:,1])-np.asarray(env.home)
                if rng.random()<c.push_probability:d.qvel[:2]+=rng.uniform(-c.push_velocity,c.push_velocity,2)
                max_impact=0.; max_descent=0.
                for _ in range(env.frames):
                    before=contacts(m,d,env.foot_ids,env.floor)
                    points=geom.capsule_bottom(d.geom_xpos[env.foot_ids],d.geom_xmat[env.foot_ids],m.geom_size[env.foot_ids])
                    velocity=geom.point_velocities(points[:,None,:],d.subtree_com[env.root_ids],d.cvel[env.body_ids])[:,0]
                    if lowering_foot>=0 and not before[lowering_foot]:
                        max_descent=max(max_descent,float(max(-velocity[lowering_foot,2],0.)))
                    mujoco.mj_step(m,d)
                    mujoco.mj_forward(m,d)
                    after=contacts(m,d,env.foot_ids,env.floor)
                    max_impact=max(max_impact,float(np.max(np.maximum(-velocity[:,2],0)*(after & ~before))))
                points=geom.capsule_bottom(d.geom_xpos[env.foot_ids],d.geom_xmat[env.foot_ids],m.geom_size[env.foot_ids])
                contact=contacts(m,d,env.foot_ids,env.floor)
                tilt=float(np.degrees(np.arccos(np.clip(d.xmat[env.torso].reshape(3,3)[2,2],-1.,1.))))
                foot=COMMAND_TO_FOOT[command]
                stance=np.arange(4)!=foot
                row=dict(time_s=(elapsed+1)*env.dt,command=command,tilt_deg=tilt,height_m=float(d.qpos[2]),
                         clearance_m=float(points[foot,2]) if command else 0.,
                         stance_missing=int(np.sum(~contact & stance)),touchdown_speed_mps=max_impact,
                         lowering_speed_mps=max_descent,
                         lowering_contact=bool(contact[lowering_foot]) if lowering_foot>=0 else False,
                         drift_m=float(np.linalg.norm(d.qpos[:2]-start_xy)))
                local.append(row);records.append(row)
                if sequencer is not None and sequencer.active:
                    active_foot=COMMAND_TO_FOOT[sequencer.active]
                    sequencer.update(env.dt,float(points[active_foot,2]),bool(contact[active_foot]))
                fell=bool(tilt>np.degrees(c.terminal_tilt) or d.qpos[2]<c.terminal_height or not np.isfinite(d.qpos).all())
                previous=action;obs=np.roll(obs,35);obs[:35]=frame(env,d,command,action,rng)
                if renderer is not None and elapsed%render_every==0:
                    cam.lookat[:]=d.xpos[env.torso]
                    renderer.update_scene(d,camera=cam)
                    frames.append(annotate(renderer.render().copy(),command,row))
                elapsed+=1
                if fell:break
            # Allow one second for command transitions before checking held clearance.
            held=local[min(round(1./env.dt),len(local)-1):]
            if initial_height is None:initial_height=float(np.mean([r['height_m'] for r in held]))
            segment_reports.append(dict(command=COMMAND_STATES[command],duration_s=len(local)*env.dt,
                peak_tilt_deg=max(r['tilt_deg'] for r in local),
                peak_touchdown_speed_mps=max(r['touchdown_speed_mps'] for r in local),
                peak_lowering_speed_mps=max(r['lowering_speed_mps'] for r in local),
                first_lowering_contact_s=next(((i+1)*env.dt for i,r in enumerate(local) if r['lowering_contact']),None),
                min_held_clearance_m=min(r['clearance_m'] for r in held) if command else None,
                max_held_stance_missing=max(r['stance_missing'] for r in held),
                mean_height_m=float(np.mean([r['height_m'] for r in held]))))
            if diagnostics:
                segment_reports[-1]['final_action']=previous.tolist()
                segment_reports[-1]['joint_offsets']=(d.qpos[7:]-np.asarray(env.home)).tolist()
            previous_command=command
            if fell:break
    finally:
        if renderer is not None:renderer.close()
    # A shortened model can spawn with feet above the plane. Its initial free
    # fall is setup, not an operator-commanded landing. Keep both measurements.
    settle_steps=round(schedule[0][1]/env.dt) if schedule[0][0]==0 and len(schedule)>1 else 0
    setup_records=records[:settle_steps]
    task_records=records[settle_steps:] or records
    front=[r['tilt_deg'] for r in task_records if r['command'] in (1,2)]
    held_lifts=[s['min_held_clearance_m'] for s in segment_reports if s['min_held_clearance_m'] is not None]
    sag=max(0.,initial_height-min(r['height_m'] for r in task_records))
    max_touch=max(r['touchdown_speed_mps'] for r in task_records)
    result=dict(seed=seed,duration_s=elapsed*env.dt,fell=fell,segments=segment_reports,
                front_peak_tilt_deg=max(front,default=0.),peak_tilt_deg=max(r['tilt_deg'] for r in task_records),
                height_sag_m=sag,final_height_m=records[-1]['height_m'],
                peak_touchdown_speed_mps=max_touch,
                peak_lowering_speed_mps=max(r['lowering_speed_mps'] for r in task_records),
                lowering_joint_speed_rad_s=lowering_joint_speed,lowering_ease_seconds=lowering_ease_seconds,
                lowering_filter='hip_descent_v1',lowering_clearance_fade_m=[.02,.04],
                peak_touchdown_time_s=max(task_records,key=lambda r:r['touchdown_speed_mps'])['time_s'],
                setup_peak_touchdown_speed_mps=max((r['touchdown_speed_mps'] for r in setup_records),default=0.),
                settling_seconds=settle_steps*env.dt,max_drift_m=max(r['drift_m'] for r in task_records),
                converted=list(converted),shortening_m=shortening,progressive=progressive,stopped_reason=stopped_reason,
                converted_mask=sequencer.converted if sequencer else sum(1<<leg for leg in converted))
    result['gates']=dict(no_fall=not fell,completed=len(segment_reports)==len(schedule) and not fell and (sequencer is None or sequencer.phase=='stand'),
        front_tilt=result['front_peak_tilt_deg']<10.,height_sag=sag<.010,
        clearance=bool(held_lifts) and min(held_lifts)>=c.lift_clearance_gate,
        stance=all(s['max_held_stance_missing']==0 for s in segment_reports),
        soft_touchdown=max_touch<c.soft_touchdown_speed,drift=result['max_drift_m']<.06)
    result['passed']=all(result['gates'].values())
    return result,frames


def load_policy(path,model_path=None):
    metadata=json.loads((Path(path).parent/'run.json').read_text())
    c=get_config();c.update(metadata['config'])
    env=PupperLegToWheelEnv(c,model_path or metadata['model_path'])
    for file,key in ((Path(env.model_path),'model_sha256'),(Path(__file__).with_name('ring_outline.json'),'ring_sha256')):
        if hashlib.sha256(file.read_bytes()).hexdigest()!=metadata[key]:raise ValueError(f'{file} differs from training')
    preprocess=running_statistics.normalize if c.ppo.normalize_observations else lambda x,y:x
    net=networks.make_ppo_networks(env.observation_size,12,preprocess_observations_fn=preprocess,
                                  policy_hidden_layer_sizes=tuple(c.hidden_layer_sizes),activation=jax.nn.elu)
    policy=networks.make_inference_fn(net)(model.load_params(path),deterministic=True)
    return env,policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',required=True)
    parser.add_argument('--model_path')
    parser.add_argument('--output',required=True)
    parser.add_argument('--video')
    parser.add_argument('--lowering_ease_seconds',type=float,default=.25,help='Fade the initial speed limit over this duration; 0 applies it throughout lowering')
    parser.add_argument('--lowering_joint_speed',type=float,default=0.,help='Initial downward hip target limit in rad/s; 0 disables it')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--suite',action='store_true',help='Add 60-second holds and every mixed-stance subset')
    parser.add_argument('--randomized',action='store_true',help='Randomized physical parameters')
    parser.add_argument('--disturbances',action='store_true',help='Also apply training sensor noise, latency, and repeated pushes')
    args=parser.parse_args()
    env,policy=load_policy(args.params,args.model_path)
    # Heating acceptance uses stationary conditions; repeated pushes are an
    # explicitly labeled stress protocol, independently of model randomization.
    if not args.disturbances:
        env.config.sensor_noise=env.config.latency_probability=env.config.push_probability=0.
        # Retain small seeded reset perturbations even in nominal dynamics.
    sequence=[(0,2.)]+[(cmd,3.) for leg in (1,2,3,4) for cmd in (leg,0)]
    report=dict(params=str(Path(args.params).resolve()),randomized=args.randomized,
                disturbances=args.disturbances,lowering_joint_speed_rad_s=args.lowering_joint_speed,lowering_ease_seconds=args.lowering_ease_seconds,
                thresholds=dict(front_tilt_deg=10.,height_sag_m=.010,
                    clearance_m=env.config.lift_clearance_gate,
                    touchdown_speed_mps=env.config.soft_touchdown_speed,drift_m=.06),runs=[])
    def record(result):
        report['runs'].append(result)
        Path(args.output).write_text(json.dumps(report,indent=2))
        if result['case']=='long_hold':
            print(f"Hold {result['command']} seed {result['seed']}: duration {result['duration_s']:.1f}s, fell={result['fell']}, gates={result['gates']}",flush=True)
    # Conversion reach is set explicitly per case; do not shorten twice.
    randomizer=jax.jit(lambda sys,keys:domain_randomize(sys,keys,shortening_range=(0.,0.)))
    for seed in args.seeds:
        randomized=randomizer(env.sys,jax.random.split(jax.random.PRNGKey(seed),1)) if args.randomized else None
        result,frames=rollout(env,policy,seed,sequence,render=bool(args.video and seed==args.seeds[0]),randomized=randomized,progressive=True,lowering_joint_speed=args.lowering_joint_speed,lowering_ease_seconds=args.lowering_ease_seconds)
        result['case']='four_leg_sequence'; record(result)
        if frames:
            import mediapy
            mediapy.write_video(args.video,frames,fps=round(1/env.dt/2))
        if args.suite:
            for command in (1,2,3,4):
                result,_=rollout(env,policy,seed,[(0,2.),(command,60.),(0,3.)],randomized=randomized,lowering_joint_speed=args.lowering_joint_speed,lowering_ease_seconds=args.lowering_ease_seconds)
                result.update(case='long_hold',command=COMMAND_STATES[command]);record(result)
                others=[i for i in range(4) if i!=COMMAND_TO_FOOT[command]]
                for count in range(4):
                    for converted in itertools.combinations(others,count):
                        for shortening in ((.005,.010) if converted else (0.,)):
                            result,_=rollout(env,policy,seed,[(0,2.),(command,3.),(0,3.)],
                                             randomized=randomized,converted=converted,shortening=shortening,lowering_joint_speed=args.lowering_joint_speed,lowering_ease_seconds=args.lowering_ease_seconds)
                            result.update(case='mixed_stance',command=COMMAND_STATES[command]);record(result)
        Path(args.output).write_text(json.dumps(report,indent=2))
        print(f'Seed {seed}: {sum(r["passed"] for r in report["runs"])}/{len(report["runs"])} cases passed',flush=True)
    report['passed']=all(r['passed'] for r in report['runs'])
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
