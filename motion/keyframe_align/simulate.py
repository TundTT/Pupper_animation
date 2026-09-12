"""Native CPU keyframe audit/video; zero network inference or optimizer updates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import shutil
import numpy as np
import mujoco
from .native import Controller, CONFIG, PHASES
from training.wheel_align import configs as model_config, geometry

POS=np.array([0,1,3,4,6,7,9,10]); WHEEL=np.array([2,5,8,11])
DT=1/520

def run(config=None, *, friction=0., tilt=(0.,0.), interrupt=False, seed=0, video=None, interrupt_phase=None,payload_scale=1.,added_wheel_mass=0.):
    model=mujoco.MjModel.from_xml_path(str(model_config.MODEL_PATH));data=mujoco.MjData(model)
    if payload_scale<=0 or added_wheel_mass<0:raise ValueError('Invalid mass sensitivity case')
    heater=model.body('heating_module').id;model.body_mass[heater]*=payload_scale;model.body_inertia[heater]*=payload_scale
    for leg in model_config.LEGS:
        body=model.body(f'leg_{leg}_3').id;ratio=(model.body_mass[body]+added_wheel_mass)/model.body_mass[body]
        model.body_mass[body]*=ratio;model.body_inertia[body]*=ratio
    mujoco.mj_setConst(model,data)
    cfg=json.loads(CONFIG.read_text()) if config is None else config
    assert np.allclose(model.opt.timestep,DT)
    assert [model.joint(i).name for i in range(1,13)]==list(model_config.JOINT_NAMES)
    kp=np.full(12,5.);kd=np.full(12,.25)
    kp[WHEEL]=cfg['wheel_position_kp'];kd[WHEEL]=cfg['wheel_position_kd']
    def torque(target,effort=None):
        result=kp*(target-data.qpos[7:])-kd*data.qvel[6:]
        if effort is not None:result[WHEEL]+=effort
        return np.clip(result,-3.,3.)
    # Synthetic joint friction is a stress test, not a measurement of the hardware.
    model.dof_frictionloss[6+WHEEL]=friction
    mujoco.mj_resetDataKeyframe(model,data,0)
    rng=np.random.default_rng(seed);data.qpos[7+WHEEL]=rng.uniform(-math.pi,math.pi,4)
    initial=data.qpos[7:].copy()
    for _ in range(1560):
        data.ctrl[:]=torque(initial);mujoco.mj_step(model,data)
    controller=Controller(config=config);home=data.qpos[7+WHEEL].copy()
    if seed:home-=rng.uniform(-math.pi,math.pi,4) # Driving after calibration: varied target errors.
    controller.reset(data.qpos[7:],home)
    reports=[];trace=[];renderer=None;writer=None
    if video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(video,fps=520/34,macro_block_size=16)
        renderer=mujoco.Renderer(model,height=352,width=480)
        camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.12];camera.distance=.75;camera.azimuth=130;camera.elevation=-22
    wheel_geoms=[model.geom(n).id for n in model_config.WHEEL_COLLISION_GEOM_NAMES]
    floor_geom=int(np.flatnonzero(model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)[0])
    def state():
        base_body=int(model.jnt_bodyid[0]);r=data.xmat[base_body].reshape(3,3)
        gravity=r.T@np.array([0.,0.,-1.])
        # Bias the IMU input only: exposes estimator sensitivity without teleporting the body.
        rx,ry=tilt;cx,sx,cy,sy=np.cos(rx),np.sin(rx),np.cos(ry),np.sin(ry)
        bias=np.array([[cy,sy*sx,sy*cx],[0,cx,-sx],[-sy,cy*sx,cy*cx]])
        velocity=np.empty(6)
        # XBODY uses the regular body frame, not the principal inertia frame.
        # https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-objectvelocity
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_XBODY,base_body,velocity,1)
        return velocity[:3],bias.T@gravity
    def tick(request):
        nonlocal renderer
        angular,gravity=state();o=controller.step(DT,request,data.qpos[7:],data.qvel[6:],angular,gravity)
        target=np.zeros(12);target[POS]=o[:8];target[WHEEL]=o[25:29]
        data.ctrl[:]=torque(target,o[29:33]) if o[23] else 0.
        if not o[23]:return o,gravity
        mujoco.mj_step(model,data)
        if renderer and int(round(data.time/DT))%34==0:
            renderer.update_scene(data,camera=camera)
            from PIL import Image,ImageDraw
            image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,480,42),fill='black');draw.text((8,5),f'SIM | keyframes | seed {seed} | {PHASES[int(o[12])]}',fill='white')
            draw.text((8,23),f'command {int(o[13])} | angle error {math.degrees(o[20]):.1f} deg | gate mask {int(o[15])}',fill='white')
            writer.append_data(np.asarray(image))
        return o,gravity
    for _ in range(2080):tick(0)
    for request,leg in [(1,1),(2,0),(3,2),(4,3)]:
        start=data.time;phase_times=np.zeros(8);min_gaps=np.full(3,np.inf);actual_rotation_clearance=np.inf
        timeout=False;interrupted=False;rotating=False;peak_approach=0.;phase=-1
        peak_tilt=0.;peak_reported_tilt=0.;peak_effort=np.zeros(12);support_min=3;rotation_ground_contact=False;max_penetration=0.
        phase_blocks={};saturation_samples=np.zeros(12,dtype=int)
        for n in range(65*520):
            command=request
            if interrupt and not interrupted and data.time-start>=3:interrupted=True
            if interrupt_phase is not None and phase==interrupt_phase:interrupted=True
            if interrupted:command=0
            axis=data.geom_xmat[wheel_geoms[leg]].reshape(3,3)[:,2]
            previous_bottom=data.geom_xpos[wheel_geoms[leg],2]-.048*np.sqrt(max(1-axis[2]**2,0))-.01675*abs(axis[2])
            o,g=tick(command);phase=int(o[12]);phase_times[phase]+=DT
            peak_tilt=max(peak_tilt,float(math.acos(np.clip(data.xmat[1].reshape(3,3)[2,2],-1,1))))
            peak_reported_tilt=max(peak_reported_tilt,float(math.acos(np.clip(-state()[1][2],-1,1))))
            peak_effort=np.maximum(peak_effort,np.abs(data.actuator_force))
            saturation_samples+=(np.abs(data.actuator_force)>=2.999)
            phase_blocks[str(int(o[15]))]=phase_blocks.get(str(int(o[15])),0)+1
            supported=set()
            for ci in range(data.ncon):
                contact=data.contact[ci];a,b=contact.geom1,contact.geom2
                if a==floor_geom or b==floor_geom:
                    other=b if a==floor_geom else a
                    if other in wheel_geoms:supported.add(wheel_geoms.index(other))
                elif contact.dist<0:max_penetration=max(max_penetration,float(-contact.dist))
            if phase==3 and o[15]==0:
                support_min=min(support_min,len(supported-{leg}));rotation_ground_contact |= leg in supported
            if phase==7:break
            axis=data.geom_xmat[wheel_geoms[leg]].reshape(3,3)[:,2]
            current_bottom=data.geom_xpos[wheel_geoms[leg],2]-.048*np.sqrt(max(1-axis[2]**2,0))-.01675*abs(axis[2])
            if phase in (4,5) and current_bottom<.008:
                peak_approach=max(peak_approach,(previous_bottom-current_bottom)/DT)
            min_gaps=np.minimum(min_gaps,o[17:20])
            if phase==3 and o[15]==0:
                rotating=True
                axes=data.geom_xmat[wheel_geoms[leg]].reshape(3,3)[:,2]
                bottom=data.geom_xpos[wheel_geoms[leg],2]-.048*np.sqrt(max(1-axes[2]**2,0))-.01675*abs(axes[2])
                actual_rotation_clearance=min(actual_rotation_clearance,bottom)
            if n%10==0:trace.append([float(data.time),request,*o.tolist()])
            timeout=timeout or o[16]>=0
            if phase==6 and (timeout or interrupted or int(o[14])&(1<<leg)):break
        completed=bool(int(o[14])&(1<<leg));ended=phase==6
        reports.append(dict(leg=model_config.LEGS[leg],completed=completed,returned_to_hold=ended,
            timed_out=bool(timeout),interrupted=interrupted,rotated=rotating,seconds=float(data.time-start),
            failed_alignment=bool(int(o[24])&(1<<leg)),
            peak_tilt_rad=peak_tilt,peak_reported_tilt_rad=peak_reported_tilt,peak_joint_effort_nm=peak_effort.tolist(),
            actuator_saturation_seconds=(saturation_samples*DT).tolist(),
            min_support_wheels_during_rotation=support_min,active_ground_contact_during_rotation=bool(rotation_ground_contact),
            max_non_ground_penetration_m=max_penetration,gate_mask_samples=phase_blocks,
            final_error_deg=math.degrees(math.atan2(math.sin(home[leg]+math.pi-data.qpos[7+WHEEL[leg]]),math.cos(home[leg]+math.pi-data.qpos[7+WHEEL[leg]]))),phase_seconds=phase_times.tolist(),
            min_wheel_gap=float(min_gaps[1]),min_body_gap=float(min_gaps[2]),
            peak_near_ground_descent_m_s=float(peak_approach),
            min_actual_rotation_clearance=None if not np.isfinite(actual_rotation_clearance) else actual_rotation_clearance))
        for _ in range(520):o,g=tick(0)
    if renderer:
        writer.close();renderer.close()
    final_errors=np.arctan2(np.sin(home+math.pi-data.qpos[7+WHEEL]),np.cos(home+math.pi-data.qpos[7+WHEEL]))
    final_mask=int(o[14]);failed_mask=int(o[24])
    controller.close()
    cancelled=interrupt or interrupt_phase is not None
    passed=all(x['returned_to_hold'] and not x['timed_out'] and not x['failed_alignment']
               and (x['completed'] or cancelled) and (cancelled or abs(x['final_error_deg'])<math.degrees(.035))
               and x['peak_near_ground_descent_m_s']<.030
               and x['min_wheel_gap']>0 and x['min_body_gap']>0 and x['peak_tilt_rad']<.12
               and not x['active_ground_contact_during_rotation'] and x['max_non_ground_penetration_m']<.001
               and (x['min_actual_rotation_clearance'] is None or x['min_actual_rotation_clearance']>.005) for x in reports)
    if not cancelled:passed=passed and final_mask==15 and failed_mask==0 and bool(np.all(np.abs(final_errors)<.035))
    return dict(passed=passed,simulation_only=True,optimizer_updates=0,seed=seed,synthetic_friction_nm=friction,
                heater_mass_scale=payload_scale,added_mass_per_wheel_kg=added_wheel_mass,
                added_mass_model='Sensitivity only: additional mass follows existing wheel assembly COM and inertia distribution; actual spacer mass is unspecified.',
                imu_bias_rad=list(tilt),interrupt=interrupt,interrupt_phase=interrupt_phase,legs=reports,
                sequence_final_errors_deg=np.degrees(final_errors).tolist(),completed_mask=final_mask,failed_mask=failed_mask),trace

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--video',action='store_true')
    p.add_argument('--friction',type=float,default=0.);p.add_argument('--imu-roll',type=float,default=0.)
    p.add_argument('--imu-pitch',type=float,default=0.);p.add_argument('--interrupt',action='store_true');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--payload-scale',type=float,default=1.);p.add_argument('--added-wheel-mass',type=float,default=0.)
    p.add_argument('--wandb',choices=['online','offline','disabled'],default='online',
                   help='Publish simulation audit/video; explicitly disable for local development checks')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.config.read_text());(args.output/'config.json').write_text(json.dumps(cfg,indent=2))
    report,trace=run(cfg,friction=args.friction,tilt=(args.imu_roll,args.imu_pitch),interrupt=args.interrupt,seed=args.seed,payload_scale=args.payload_scale,added_wheel_mass=args.added_wheel_mass,
                     video=str(args.output/'rollout.mp4') if args.video else None)
    # Save physics results before optional provenance/logging so failures are retained.
    (args.output/'audit.json').write_text(json.dumps(report,indent=2));np.savetxt(args.output/'trace.csv',trace,delimiter=',')
    from .provenance import provenance
    report.update(provenance());report['config_sha256']=hashlib.sha256(args.config.read_bytes()).hexdigest()
    (args.output/'audit.json').write_text(json.dumps(report,indent=2))
    if args.wandb!='disabled':
        from training.wandb_logging import ExperimentLogger
        logger=ExperimentLogger(args.output,dict(report,controller='deterministic-keyframes',checkpoint_step=None),mode=args.wandb)
        logger.run.summary.update({'keyframes/simulation_passed':report['passed'],'hardware_validated':False,
                                   'optimizer_updates':0,'checkpoint_applicable':False})
        if args.video:
            logger.video(args.output/'rollout.mp4',len(trace)*10,key='motion/keyframes',
                         caption=f'SIMULATION deterministic keyframes; seed {args.seed}; no neural checkpoint; passed={report["passed"]}')
        artifact=logger.sdk.Artifact('keyframe-alignment-'+logger.state['id'],type='motion-audit')
        for name in ('audit.json','config.json','trace.csv'):artifact.add_file(str(args.output/name))
        logger.run.log_artifact(artifact);logger.run.finish()
        report['wandb_mode']=args.wandb;report['wandb_run_url']=logger.state.get('url')
        (args.output/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':main()
