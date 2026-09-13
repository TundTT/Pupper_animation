"""Post-cooling entry -> roll -> settle, continuous native simulation.

An alignment terminal record is the preferred input. --synthetic is an explicit
local preflight, never evidence of a real alignment-controller handoff.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import mujoco as mj
from .core import Robot,HERE,HUB,PROX,KP,KD,provenance,versions
from .handoff_entry import EntryRoll,NEUTRAL,SIGNS,LOW,HIGH
from .handoff_shape import sample
from .balanced_support import BalancedSupport
from .roll_audit import RollRecorder
from .clearance import CADClearance

def initialize(config,terminal=None):
    allowed={'seed','entry_seconds','roll_seconds','settle_seconds','pose_noise_deg','friction','dynamics','formation','entry_integral_gain','entry_mode','entry_splay'}
    if set(config)-allowed:raise ValueError('Unknown handoff configuration fields')
    r=Robot(friction=config.get('friction',.8),dynamics=config.get('dynamics'),formation=config.get('formation'))
    if terminal is None:
        q=NEUTRAL.copy();q[HUB]-=SIGNS*np.pi
        r.d.qpos[7:]=q;r.d.qvel[:]=0
        boundary=dict(source_kind='synthetic_alignment_neutral',actual_alignment_replay=False)
    else:
        names=['leg_'+leg+'_'+str(j) for leg in ('front_r','front_l','back_r','back_l') for j in (1,2,3)]
        if terminal.get('schema_version')!=1 or terminal.get('joint_names')!=names:
            raise ValueError('Unsupported alignment terminal schema/joint order')
        if terminal.get('completed_mask')!=15 or terminal.get('failed_mask')!=0 or not terminal.get('lowering_and_settling_finished') or not terminal.get('alignment_passed'):
            raise ValueError('Alignment must finish all four wheels and lower/settle successfully')
        if not terminal.get('proximal_frame_verified') or not terminal.get('source_commit') or not terminal.get('model_sha256'):
            raise ValueError('Source provenance and proximal frame verification required')
        if terminal.get('target_model_sha256')!=r.manifest.get('nominal_model_sha256',r.manifest['model_sha256']):
            raise ValueError('Alignment frame verification must name the current nominal triangle model')
        source_q=np.array(terminal['qpos'],float);source_v=np.array(terminal['qvel'],float)
        home=np.array(terminal['wheel_home'],float)
        if source_q.shape!=(19,) or source_v.shape!=(18,) or home.shape!=(4,) or not np.isfinite(np.r_[source_q,source_v,home]).all():
            raise ValueError('Invalid terminal arrays')
        if abs(np.linalg.norm(source_q[3:7])-1)>.001:raise ValueError('Invalid terminal quaternion')
        r.d.qpos[:]=source_q;r.d.qvel[:]=source_v
        # A fixed frame conversion at the declared wheel -> cold-CAD boundary.
        # Left sides use -2pi so the future forward target stays near +1 rad.
        offset=NEUTRAL[HUB]-home+np.array([0.,-2*np.pi,0.,-2*np.pi])
        r.d.qpos[7+HUB]+=offset
        boundary=dict(source_kind=terminal['source_kind'],actual_alignment_replay=True,
            source_commit=terminal['source_commit'],source_model_sha256=terminal['model_sha256'],
            hub_frame_offset_rad=offset.tolist(),source_terminal=terminal)
    rng=np.random.default_rng(config.get('seed',0))
    noise=np.deg2rad(config.get('pose_noise_deg',0.))
    perturb=rng.uniform(-noise,noise,12);r.d.qpos[7:]+=perturb
    if np.any(r.d.qpos[7:]<LOW) or np.any(r.d.qpos[7:]>HIGH):raise ValueError('Perturbed start outside joint limits')
    mj.mj_forward(r.m,r.d)
    old_z=float(r.d.qpos[2]);r.d.qpos[2]-=float(r.bottoms().min())
    mj.mj_forward(r.m,r.d)
    boundary.update(pose_perturbation_rad=perturb.tolist(),root_height_adjustment_m=float(r.d.qpos[2])-old_z,
        boundary='Post-cooling geometry initialization with lowest intended foot at floor; heating/deformation dynamics NOT simulated. No subsequent state resets.')
    r.initial=r.d.qpos.copy();command=r.initial[7:].copy()
    if terminal is not None:
        proximal=np.asarray(terminal['last_proximal_position_target'],float)
        if proximal.shape!=(8,) or not np.isfinite(proximal).all():raise ValueError('Last proximal command required')
        command[PROX]=proximal
    boundary['entry_initial_command']=command.tolist()
    boundary['command_boundary']='Preserve last proximal position targets; switch hubs from velocity/torque control to measured-angle position hold. No live encoder reset.'
    r.command_history=[command.copy() for _ in range(r.dynamics['delay_steps'])]
    return r,boundary

def probe(config,output,terminal=None,video=True,cad=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=dict(source_hashes=provenance(),versions=versions(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()))
    r,boundary=initialize(config,terminal)
    from .cold_boundary import inspect
    boundary["cold_geometry"]=inspect(r)
    motion=EntryRoll(r.d.qpos[7:],previous_command=boundary['entry_initial_command'],entry_seconds=config.get('entry_seconds',4.),
        roll_seconds=config.get('roll_seconds',12.),settle_seconds=config.get('settle_seconds',32.),entry_integral_gain=config.get('entry_integral_gain',0.),entry_mode=config.get('entry_mode','measured'),entry_splay=config.get('entry_splay',.29))
    feedback=BalancedSupport(motion.goal,dict(estimate_mode='hybrid',norm_delay_s=2.,force_gain=.003,reference_cap=.075,integral_gain=.25))
    recorder=RollRecorder(r,motion.command);checker=CADClearance(r) if cad else None
    states=[];trace=[];gaps=[];hold=[];reason=None;peak_torque=0.;peak_speed=0.;correction=np.zeros(12)
    for step in range(int(520*(motion.entry_seconds+motion.roll_seconds+motion.settle_seconds+10))):
        if motion.phase=='settle' and step%10==0:
            correction=feedback.update(r.d.qpos[3:7],r.d.qpos[7:],r.d.qvel[6:],motion.command,10/520)
        target=motion.step(r.d.qpos[7:],r.d.qvel[6:],1/520,correction)
        if motion.phase=='failed':reason=motion.failure;break
        damping=KD*(1 if motion.phase in ('settle','done') else 2)
        tau=r.tick(target,kd=damping);recorder.add(r,target,kd=damping)
        peak_torque=max(peak_torque,float(np.max(abs(tau))))
        peak_speed=max(peak_speed,float(np.max(abs(r.d.qvel[6:]))))
        if step%13==0:
            row=dict(time=float(r.d.time),phase=motion.phase,qpos=r.d.qpos.tolist(),qvel=r.d.qvel.tolist(),
                tilt_deg=float(np.rad2deg(r.tilt())),body_height_m=float(r.d.qpos[2]),
                pose_error_rad=float(np.max(abs(r.d.qpos[7:]-motion.goal))),
                tip_bottom_m=r.tip_bottoms().tolist(),normal_force_N=r.contacts_precise().tolist(),
                unintended_floor_force_N=r.unintended_floor_force(),joint_speed=float(np.max(abs(r.d.qvel[6:]))),
                base_speed=float(np.linalg.norm(r.d.qvel[:3])))
            trace.append(row)
            if motion.phase=='settle' and motion.elapsed>=motion.settle_seconds-3:hold.append(row)
            if row['tilt_deg']>8 or peak_speed>2:reason='tilt_or_speed';break
        if step%26==0:states.append((float(r.d.time),r.d.qpos.copy()))
        if checker and step%52==0:
            gap=checker.measure();gaps.append(dict(step=step,**gap))
            if gap['intersections']:reason='CAD_intersection';break
        if motion.phase=='done':break
    dense=recorder.finish(r,output,cad=cad) if recorder.commands else None
    all_hold=lambda predicate:len(hold)>=119 and all(predicate(x) for x in hold)
    gates=dict(cold_boundary=boundary['cold_geometry']['contact_free_at_fixed_measured_posture'],completed=motion.phase=='done' and reason is None,
        command_rate=motion.peak_speed_ratio<=1.000001 and motion.peak_accel_ratio<=1.000001,
        tilt=recorder.tilt_peak<=8,torque=peak_torque<=r.dynamics['torque_limit_Nm'],
        motor_body_floor=recorder.floor_peak<=.01,measured_speed=peak_speed<=2,
        three_second_walk_pose=all_hold(lambda x:x['pose_error_rad']<=.1),
        three_second_tip_support=all_hold(lambda x:max(map(abs,x['tip_bottom_m']))<=.003 and min(x['normal_force_N'])>=1),
        three_second_settled=all_hold(lambda x:x['joint_speed']<=.1 and x['base_speed']<=.03),
        dense_cad=None if not cad or not dense else bool(dense['cad_pass']))
    floor_rows=[x for x in trace if x['unintended_floor_force_N']>.01]
    floor_summary=None if not floor_rows else dict(first_s=floor_rows[0]['time'],last_s=floor_rows[-1]['time'],
        peak_sample_N=max(x['unintended_floor_force_N'] for x in floor_rows),phases=sorted(set(x['phase'] for x in floor_rows)))
    report=dict(config=config,boundary=boundary,model_manifest=r.manifest,simulation_only=True,
        hardware_validated=False,walking_handoff_validated=False,optimizer_updates=0,
        engineering_gates=gates,engineering_pass=all(v is True for v in gates.values()),
        acceptance_complete=False,remaining_acceptance=['actual alignment endpoint distribution','base contact convergence','fresh held-out perturbations','walking handoff'],
        final_phase=motion.phase,early_termination=reason,environment_steps=len(recorder.commands),
        max_command_speed_ratio=motion.peak_speed_ratio,max_command_acceleration_ratio=motion.peak_accel_ratio,
        peak_requested_torque_Nm=peak_torque,dense_audit=dense,initial_q=r.initial.tolist(),final_q=r.d.qpos.tolist(),
        sampled_floor_contact=floor_summary,**source)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');(output/'trace.json').write_text(json.dumps(trace)+'\n')
    (output/'final_state.json').write_text(json.dumps(r.snapshot(motion.command))+'\n')
    if video and states:
        from .roll_to_stand import render_states
        render_states(r,states,output/'rollout.mp4',f'ENTRY + ROLL | {boundary["source_kind"]} | {reason or motion.phase}')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--terminal',type=Path);source.add_argument('--synthetic',action='store_true')
    p.add_argument('--config',type=Path,help='Explicit motion/formation/dynamics JSON; command-line values override it')
    p.add_argument('--seed',type=int);p.add_argument('--undercompressed',action='store_true')
    p.add_argument('--pose-noise-deg',type=float);p.add_argument('--entry-seconds',type=float)
    p.add_argument('--no-cad',action='store_true');p.add_argument('--no-video',action='store_true')
    p.add_argument('--wandb',choices=['online','offline','disabled'],default='online');args=p.parse_args()
    config=dict(seed=0,entry_seconds=4.,pose_noise_deg=0.)
    if args.config:config.update(json.loads(args.config.read_text()))
    for key in ('seed','entry_seconds','pose_noise_deg'):
        value=getattr(args,key)
        if value is not None:config[key]=value
    if args.undercompressed:config['formation']=sample(config['seed'])[0]
    terminal=json.loads(args.terminal.read_text()) if args.terminal else None
    report=probe(config,args.output,terminal,video=not args.no_video,cad=not args.no_cad)
    if args.wandb!='disabled':
        from training.wandb_logging import ExperimentLogger
        log=ExperimentLogger(args.output,dict(config,source_hashes=report['source_hashes'],boundary=report['boundary']),mode=args.wandb)
        log.metrics(report['environment_steps'],{'handoff/engineering_pass':report['engineering_pass'],'handoff/acceptance_complete':False})
        if not args.no_video:log.video(args.output/'rollout.mp4',report['environment_steps'],key='handoff/continuous_motion',
            caption=f'SIMULATION entry + roll; seed {config["seed"]}; {report["boundary"]["source_kind"]}; termination {report["early_termination"]}; no policy checkpoint')
        artifact=log.sdk.Artifact('alignment-roll-handoff-'+log.state['id'],type='motion-audit')
        for file in args.output.iterdir():
            if file.suffix in ('.json','.npz'):artifact.add_file(str(file))
        log.run.log_artifact(artifact);log.run.finish();report['wandb_url']=log.state.get('url')
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('engineering_gates','early_termination','environment_steps','acceptance_complete')},indent=2))
    raise SystemExit(0 if report['engineering_pass'] else 1)

if __name__=='__main__':main()
