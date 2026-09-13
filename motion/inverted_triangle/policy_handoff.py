"""Continue actual roll physics into the pinned walking export; no hardware code."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import mujoco as mj
from .core import Robot,provenance,versions,HERE
from .roll_to_stand import probe,render_states
from .walking_policy import WalkingPolicy,EXPORT_SHA256
from .sensors import JointImuSensors
from .clearance import CADClearance
from training.wandb_logging import ExperimentLogger

def handoff(config,transition,output,video=True,cad=True,contact_model='cad'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);scenario=config.get('scenario',{})
    start=json.loads((Path(transition)/'final_state.json').read_text());old_audit=json.loads((Path(transition)/'audit.json').read_text())
    r=Robot(config.get('friction',.8),scenario.get('dynamics'),config.get('formation'),contact_model=contact_model)
    if contact_model=='cad':home=r.restore(start);boundary=dict(model_changed=False,state_exact=True)
    else:
        original=Robot(config.get('friction',.8),scenario.get('dynamics'),config.get('formation'));home=original.restore(start)
        # Explicit matched-state counterfactual; no relabelled snapshot hash.
        assert r.m.nq==original.m.nq and r.m.nv==original.m.nv and r.m.nu==original.m.nu
        np.testing.assert_allclose(r.m.body_mass,original.m.body_mass,atol=0,rtol=0)
        mj.mj_setState(r.m,r.d,np.asarray(start['state']),mj.mjtState.mjSTATE_INTEGRATION);mj.mj_forward(r.m,r.d)
        r.command_history=[np.array(q) for q in start['command_history']]
        np.testing.assert_array_equal(r.d.qpos,original.d.qpos);np.testing.assert_array_equal(r.d.qvel,original.d.qvel)
        np.testing.assert_allclose(r.d.xpos,original.d.xpos,atol=1e-12)
        boundary=dict(model_changed=True,old_model_sha256=start['model_sha256'],new_model_sha256=r.manifest['model_sha256'],state_exact=True,reason='Explicit contact-model counterfactual; primary acceptance uses unchanged CAD-floor physics')
    actual_start=r.snapshot(home);(output/'start_state.json').write_text(json.dumps(actual_start,indent=2))
    policy=WalkingPolicy(r.d.qpos[7:].copy());sensor=JointImuSensors(scenario.get('sensors'),config.get('seed',0));checker=CADClearance(r) if cad else None
    zero_seconds=12.;ramp_seconds=1.;forward_seconds=7.;steps=round((zero_seconds+ramp_seconds+forward_seconds)*520)
    from .roll_audit import RollRecorder
    recorder=RollRecorder(r,home)
    rows=[];states=[];zero=[];forward=[];target=home.copy();start_xy=r.d.qpos[:2].copy();forward_start=None
    report=dict(config=config,boundary=boundary,model_manifest=r.manifest,export_sha256=EXPORT_SHA256,source_hashes=provenance(),environment=versions(),simulation_only=True,hardware_validated=False,contact_model=contact_model,transition_gates=old_audit['gates'],transition_pass=all(v is True for v in old_audit['gates'].values()),peak_requested_torque_Nm=0.,max_tilt_deg=0.,max_motor_body_floor_N=0.,max_measured_joint_speed=0.,minimum_cad_gap_m=1.,first_cad_intersections=None,early_termination=None,physics_hz=520,policy_hz=52,training_policy_hz=50,init_seconds=2.,fade_seconds=2.)
    for k in range(steps):
        elapsed=k/520;measurement=sensor.read(r)
        command=np.array([.1*np.clip((elapsed-zero_seconds)/ramp_seconds,0,1),0,0])
        if k<1040 or (k-1040)%10==0:target=policy.target(elapsed,measurement['omega'],measurement['gravity'],measurement['q'],command)
        tau=r.tick(target,kp=np.full(12,5.),kd=np.full(12,.25))
        recorder.add(r,target)
        report['peak_requested_torque_Nm']=max(report['peak_requested_torque_Nm'],float(np.abs(tau).max()))
        report['max_measured_joint_speed']=max(report['max_measured_joint_speed'],float(np.abs(r.d.qvel[6:]).max()))
        if k%13==0:
            R=r.d.xmat[r.base].reshape(3,3);tilt=float(np.rad2deg(r.tilt()));floor=r.unintended_floor_force()
            report['max_tilt_deg']=max(report['max_tilt_deg'],tilt);report['max_motor_body_floor_N']=max(report['max_motor_body_floor_N'],floor)
            row=dict(time_s=float(r.d.time),elapsed_s=elapsed,command=command.tolist(),qpos=r.d.qpos.tolist(),qvel=r.d.qvel.tolist(),target=target.tolist(),normal_force_N=r.contacts_precise().tolist(),tip_bottom_m=r.tip_bottoms().tolist(),tilt_deg=tilt,floor_force_N=floor,base_speed_m_s=float(np.linalg.norm(r.d.qvel[:3])),body_velocity_m_s=(R.T@r.d.qvel[:3]).tolist(),pose_error_rad=float(np.abs(r.d.qpos[7:]-policy.home).max()),joint_speed_rad_s=float(np.abs(r.d.qvel[6:]).max()))
            rows.append(row)
            if zero_seconds-3<=elapsed<zero_seconds:zero.append(row)
            if elapsed>=zero_seconds+ramp_seconds:
                if forward_start is None:forward_start=r.d.qpos[:2].copy()
                forward.append(row)
            if checker and k%52==0:
                c=checker.measure();report['minimum_cad_gap_m']=min(report['minimum_cad_gap_m'],c['minimum_m'])
                if c['intersections'] and report['first_cad_intersections'] is None:report['first_cad_intersections']=dict(time_s=float(r.d.time),pairs=c['intersections'])
            if tilt>35 or r.d.qpos[2]<.045 or not np.isfinite(r.d.qpos).all():report['early_termination']=dict(elapsed_s=elapsed,tilt_deg=tilt,height_m=float(r.d.qpos[2]));break
        if k%26==0:states.append((float(r.d.time),r.d.qpos.copy()))
    report['dense_dynamics_audit']=recorder.finish(r,output,cad,kp=np.full(12,5.),kd=np.full(12,.25))
    report['max_motor_body_floor_N']=max(report['max_motor_body_floor_N'],recorder.floor_peak)
    report['max_tilt_deg']=max(report['max_tilt_deg'],recorder.tilt_peak)
    report['environment_steps']=k+1;report['zero_samples']=len(zero);report['forward_samples']=len(forward)
    report['forward_displacement_m']=None if forward_start is None else (r.d.qpos[:2]-forward_start).tolist()
    report['forward_mean_body_velocity_m_s']=np.mean([x['body_velocity_m_s'] for x in forward],axis=0).tolist() if forward else None
    gates=dict(dense_cad=bool(cad and report['dense_dynamics_audit']['cad_pass']),completed=k+1==steps,transition_pass=report['transition_pass'],same_physics=contact_model=='cad',tilt=report['max_tilt_deg']<=8.,torque=report['peak_requested_torque_Nm']<=3.,motor_body_floor=report['max_motor_body_floor_N']<=.01,sampled_cad=bool(cad and report['minimum_cad_gap_m']>=.001 and report['first_cad_intersections'] is None),zero_stable=len(zero)>=119 and all(x['base_speed_m_s']<=.03 and x['joint_speed_rad_s']<=.1 for x in zero),zero_four_tip_support=len(zero)>=119 and all(min(x['normal_force_N'])>=1 and max(map(abs,x['tip_bottom_m']))<=.003 for x in zero),zero_pose=len(zero)>=119 and max(x['pose_error_rad'] for x in zero)<=.1,forward_tracking=len(forward)>=279 and abs(report['forward_mean_body_velocity_m_s'][0]-.1)<=.05 and abs(report['forward_mean_body_velocity_m_s'][1])<=.04)
    report['gates']=gates;report['status']='PASS_CONTINUOUS_ROLL_STAND_WALK' if all(gates.values()) else 'FAILED_OR_COUNTERFACTUAL'
    (output/'audit.json').write_text(json.dumps(report,indent=2));(output/'trace.json').write_text(json.dumps(rows));(output/'final_state.json').write_text(json.dumps(r.snapshot(target),indent=2));np.savez_compressed(output/'states.npz',times=[t for t,q in states],qpos=[q for t,q in states])
    if video:
        previous=np.load(Path(transition)/'states.npz');full=list(zip(previous['times'],previous['qpos']))+states
        render_states(r,full,output/'continuous_rollout.mp4',config['name']+' | roll + walking export '+EXPORT_SHA256[:8]+' | '+report['status'])
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--contact-model',choices=['cad','rigid_flush'],default='cad');a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    config=json.loads(a.config.read_text());setup=dict(config=config,source_hashes=provenance(),environment=versions(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),simulation_only=True,contact_model=a.contact_model)
    (a.output/'provenance.json').write_text(json.dumps(setup,indent=2));logger=ExperimentLogger(a.output,setup)
    try:
        initial=probe(config,a.output/'roll',video=True,cad=True);report=handoff(config,a.output/'roll',a.output/'handoff',contact_model=a.contact_model)
        steps=initial['environment_steps']+report['environment_steps'];logger.metrics(steps,dict(status=report['status'],max_tilt_deg=report['max_tilt_deg']));logger.video(a.output/'handoff/continuous_rollout.mp4',steps,key='trajectory/continuous_roll_stand_walk',caption=f'SIMULATION actual continuous states | selected export {EXPORT_SHA256} | {report["status"]} | {a.contact_model} contacts | early termination {report["early_termination"]}; no hardware')
        logger.run.summary.update(dict(simulation_audit_status=report['status'],gates=report['gates'],hardware_validated=False));artifact=logger.sdk.Artifact('roll-walking-'+logger.state['id'],type='simulation-handoff')
        for f in a.output.rglob('*'):
            if f.is_file() and f.suffix in ['.json','.npz'] and 'wandb' not in f.relative_to(a.output).parts:artifact.add_file(str(f),name=f.relative_to(a.output).as_posix())
        logger.run.log_artifact(artifact)
    finally:logger.finish()
if __name__=='__main__':main()
