"""Replay a candidate through torque-limited PD and audit true physics.

The trajectory has no access to contact forces or ground-truth base position.
Those signals are used only in the audit. This is not a robot command program.
"""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import mujoco as mj
from .core import Robot,LEGS,PROX,HUB,HERE,smooth,duration,provenance,SPEED,versions
from .clearance import CADClearance

def trajectory(home,lift,leg,landing_delta=.15,direction=-1,*,pre_shift_pose=None,landing_pose=None,touchdown_pose=None):
    home=np.asarray(home,dtype=float);lift=np.asarray(lift,dtype=float)
    if home.shape!=(12,) or lift.shape!=(12,) or not np.isfinite([home,lift]).all():
        raise ValueError('Invalid home/lift pose')
    if direction not in [-1,1] or not np.allclose(lift[HUB],home[HUB],atol=1e-10,rtol=0):
        raise ValueError('Lift must preserve all commanded hub references')
    phases=[('initial_hold',home,2.)]
    shift_home=home
    if pre_shift_pose is not None:
        shift_home=np.asarray(pre_shift_pose,dtype=float)
        if shift_home.shape!=(12,) or not np.isfinite(shift_home).all() or not np.allclose(shift_home[HUB],home[HUB],atol=1e-10,rtol=0):
            raise ValueError('Body shift must preserve all commanded hub references')
        phases.extend([('body_shift',shift_home,4.),('body_shift_hold',shift_home,1.)])
    shift=lift.copy();shift[3*leg:3*leg+2]=shift_home[3*leg:3*leg+2]
    turned=lift.copy();turned[HUB[leg]]+=direction*np.pi
    landed=turned.copy();landed[3*leg+1]-=(1 if leg%2==0 else -1)*landing_delta
    if landing_pose is not None:
        landed=np.asarray(landing_pose,dtype=float)
        if landed.shape!=(12,) or not np.isfinite(landed).all() or not np.allclose(landed[HUB],turned[HUB],atol=1e-10,rtol=0):
            raise ValueError('Landing must preserve the turned hub references')
    phases.extend([('shift',shift,1.5),('lift',lift,2.),('clearance_hold',lift,1.),('rotate',turned,12.),('angle_hold',turned,1.),('land',landed,4. if landing_pose is not None else 3.),('planted_hold',landed,2.)])
    if touchdown_pose is not None:
        touchdown=np.asarray(touchdown_pose,dtype=float)
        if touchdown.shape!=(12,) or not np.isfinite(touchdown).all() or not np.allclose(touchdown[HUB],turned[HUB],atol=1e-10,rtol=0):
            raise ValueError('Touchdown must preserve the turned hub references')
        # Both segments are audited as landing, including near-contact descent.
        phases.insert(-2,('land',touchdown,4.))
        phases.insert(-2,('land',touchdown,1.))
    return phases

def run(candidate,output,video=False,friction=.8,landing_delta=.15,direction=-1,seed=0,cad=True,start_override=None,scenario=None):
    source=json.loads(Path(candidate).read_text());leg=LEGS.index(source['leg'])
    scenario=scenario or {}
    if set(scenario)-{'dynamics','initial_offset'}:raise ValueError('Unknown scenario field')
    r=Robot(friction=friction,dynamics=scenario.get('dynamics'));home=r.initial[7:].copy();lift=np.array(source['full_pose'])
    if source.get('model_sha256')!=r.manifest['model_sha256']:raise ValueError('Candidate model hash mismatch')
    if start_override is not None:home=r.restore(start_override)
    elif 'start_state' in source:home=r.restore(source['start_state'])
    if lift.shape!=(12,) or not np.all(np.isfinite(lift)) or direction not in [-1,1] or not np.isfinite(landing_delta):raise ValueError('Invalid trajectory')
    phases=trajectory(home,lift,leg,landing_delta,direction,pre_shift_pose=source.get('pre_shift_pose'),landing_pose=source.get('landing_pose'),touchdown_pose=source.get('touchdown_pose'))
    limits=r.m.jnt_range[1:][PROX]
    if any(np.any((target[PROX]<limits[:,0])|(target[PROX]>limits[:,1])) for _,target,_ in phases):raise ValueError('Proximal target exceeds software limits')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(seed)
    if seed:r.d.qpos[7:][PROX]+=rng.uniform(-.01,.01,8);mj.mj_forward(r.m,r.d)
    if start_override is None and 'start_state' not in source:
        offsets=scenario.get('initial_offset',{})
        if set(offsets)-{'height_m','roll_rad','pitch_rad'}:raise ValueError('Unknown initial offset')
        r.d.qpos[2]+=float(offsets.get('height_m',0.))
        roll=float(offsets.get('roll_rad',0.));pitch=float(offsets.get('pitch_rad',0.))
        if not np.isfinite([r.d.qpos[2],roll,pitch]).all():raise ValueError('Invalid initial offset')
        quat=np.array([np.cos(roll/2)*np.cos(pitch/2),np.sin(roll/2)*np.cos(pitch/2),np.cos(roll/2)*np.sin(pitch/2),-np.sin(roll/2)*np.sin(pitch/2)])
        combined=np.empty(4);mj.mju_mulQuat(combined,r.d.qpos[3:7].copy(),quat);r.d.qpos[3:7]=combined
        mj.mj_forward(r.m,r.d)
    (output/'start_state.json').write_text(json.dumps(r.snapshot(home),indent=2))
    cad_check=CADClearance(r) if cad else None
    trace=[];audit=dict(status='FAILED',leg=LEGS[leg],seed=seed,friction=friction,landing_delta=landing_delta,direction=direction,model_sha256=r.manifest['model_sha256'],candidate_sha256=hashlib.sha256(Path(candidate).read_bytes()).hexdigest(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parents[1]).decode().strip(),source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=HERE.parents[1]).strip()),source_hashes=provenance(),simulation_only=True,heating=False,minimum_rotation_floor_m=1.,minimum_cad_gap_m=1.,cad_intersections=[],max_tilt_deg=0.,peak_requested_torque_Nm=0.,minimum_support_force_N=1e9,maximum_rotation_contact_N=0.,peak_landing_descent_m_s=0.,maximum_unintended_floor_force_N=0.,maximum_command_speed_ratio=0.,early_termination=None)
    renderer=None;writer=None
    audit['scenario']=scenario;audit['effective_dynamics']=r.dynamics
    audit['environment']=versions();audit['continued_from_state']=start_override is not None or 'start_state' in source
    audit['cad_sample_period_s']=.1;audit['physics_sample_period_s']=.025
    audit['first_sampled_gate_violation']=None
    audit['minimum_cad_sample']=None
    audit['simulation_start_time_s']=float(r.d.time)
    audit['start_state_sha256']=hashlib.sha256((output/'start_state.json').read_bytes()).hexdigest()
    if video:
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw
        r.m.vis.global_.offwidth=960;r.m.vis.global_.offheight=640
        renderer=mj.Renderer(r.m,height=640,width=960);writer=imageio.get_writer(output/'rollout.mp4',fps=20,macro_block_size=16)
        camera=mj.MjvCamera();camera.lookat[:]=[0,0,.10];camera.distance=.65;camera.azimuth=140;camera.elevation=-18
    previous=home.copy();last_command=home.copy();tick=0;last_bottom=None;last_time=None
    for phase,target,seconds in phases:
        steps=int(np.ceil(duration(previous,target,seconds)/r.m.opt.timestep))
        for k in range(steps):
            command=previous+smooth((k+1)/steps)*(target-previous)
            audit['maximum_command_speed_ratio']=max(audit['maximum_command_speed_ratio'],float(np.max(np.abs(command-last_command)/(r.m.opt.timestep*SPEED))))
            last_command=command.copy()
            torque=r.tick(command);tick+=1
            audit['peak_requested_torque_Nm']=max(audit['peak_requested_torque_Nm'],float(np.abs(torque).max()))
            if tick%13:continue
            bottoms=r.bottoms();forces=r.contacts_precise();tilt=np.rad2deg(r.tilt())
            audit['maximum_unintended_floor_force_N']=max(audit['maximum_unintended_floor_force_N'],r.unintended_floor_force())
            audit['max_tilt_deg']=max(audit['max_tilt_deg'],float(tilt));audit['peak_requested_torque_Nm']=max(audit['peak_requested_torque_Nm'],float(np.abs(torque).max()))
            if phase in ['rotate','angle_hold']:
                audit['minimum_rotation_floor_m']=min(audit['minimum_rotation_floor_m'],float(bottoms[leg]))
                audit['minimum_support_force_N']=min(audit['minimum_support_force_N'],float(np.delete(forces,leg).min()))
                audit['maximum_rotation_contact_N']=max(audit['maximum_rotation_contact_N'],float(forces[leg]))
            descent=0.
            if phase=='land' and last_bottom is not None and bottoms[leg]<.008:
                descent=float((last_bottom-bottoms[leg])/(r.d.time-last_time))
                audit['peak_landing_descent_m_s']=max(audit['peak_landing_descent_m_s'],descent)
            last_bottom=float(bottoms[leg]);last_time=float(r.d.time)
            if cad_check and tick%52==0:
                c=cad_check.measure()
                if c['minimum_m']<audit['minimum_cad_gap_m']:
                    audit['minimum_cad_sample']=dict(time_s=float(r.d.time),phase=phase,nearest_pair=c['nearest_pair'])
                audit['minimum_cad_gap_m']=min(audit['minimum_cad_gap_m'],c['minimum_m'])
                for pair in c['intersections']:
                    if pair not in audit['cad_intersections']:audit['cad_intersections'].append(pair)
            violations=[]
            if descent>=.025:violations.append(dict(gate='gentle_landing',value=descent,threshold=.025))
            if phase in ['rotate','angle_hold']:
                if bottoms[leg]<.005:violations.append(dict(gate='rotation_floor',value=float(bottoms[leg]),threshold=.005))
                if np.delete(forces,leg).min()<1.:violations.append(dict(gate='three_supports',value=float(np.delete(forces,leg).min()),threshold=1.))
                if forces[leg]>=.2:violations.append(dict(gate='no_rotation_contact',value=float(forces[leg]),threshold=.2))
            if r.unintended_floor_force()>=.2:violations.append(dict(gate='no_other_floor_support',value=r.unintended_floor_force(),threshold=.2))
            if tilt>=8.:violations.append(dict(gate='tilt',value=float(tilt),threshold=8.))
            if cad_check and tick%52==0 and c['minimum_m']<.001:violations.append(dict(gate='collision_clear',value=c['minimum_m'],threshold=.001,pair=c['nearest_pair']))
            if violations and audit['first_sampled_gate_violation'] is None:
                audit['first_sampled_gate_violation']=dict(time_s=float(r.d.time),phase=phase,violations=violations)
            trace.append(dict(time=float(r.d.time),phase=phase,qpos=r.d.qpos.tolist(),qvel=r.d.qvel.tolist(),command=command.tolist(),force_N=forces.tolist(),bottom_m=bottoms.tolist(),tilt_deg=float(tilt),unintended_floor_force_N=r.unintended_floor_force(),landing_descent_m_s=descent,joint_tracking_error_rad=(command-r.d.qpos[7:]).tolist()))
            if writer and tick%26==0:
                renderer.update_scene(r.d,camera=camera)
                img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img);draw.rectangle((0,0,960,55),fill='black')
                draw.text((12,8),f'SIMULATION | rigid inverted triangles | {LEGS[leg]} | {phase} | candidate, not hardware validated',fill='white')
                draw.text((12,30),f't={r.d.time:.1f}s  active floor gap={bottoms[leg]*1000:.1f} mm  tilt={tilt:.1f} deg',fill='white');writer.append_data(np.asarray(img))
            if tilt>35 or r.d.qpos[2]<.045 or not np.all(np.isfinite(r.d.qpos)):
                audit['early_termination']=phase;break
        if audit['early_termination']:break
        previous=target.copy()
    if writer:writer.close();renderer.close()
    audit['final_normal_force_N']=r.contacts_precise().tolist();audit['final_hub_error_rad']=float(abs(r.d.qpos[7+HUB[leg]]-(home[HUB[leg]]+direction*np.pi)))
    audit['final_speed_rad_s']=float(np.abs(r.d.qvel[6:]).max())
    audit['final_tip_bottom_m']=r.tip_bottoms().tolist()
    audit['environment_steps']=tick
    gates=dict(completed=not bool(audit['early_termination']),rotation_floor=audit['minimum_rotation_floor_m']>=.005,three_supports=audit['minimum_support_force_N']>=1.,no_rotation_contact=audit['maximum_rotation_contact_N']<.2,no_other_floor_support=audit['maximum_unintended_floor_force_N']<.2,command_speed=audit['maximum_command_speed_ratio']<=1.0001,cad_checked=cad,collision_clear=bool(cad and not audit['cad_intersections'] and audit['minimum_cad_gap_m']>=.001),tilt=audit['max_tilt_deg']<8.,torque=audit['peak_requested_torque_Nm']<=3.,planted=audit['final_normal_force_N'][leg]>=2.,hub_angle=audit['final_hub_error_rad']<.035,settled=audit['final_speed_rad_s']<.1,gentle_landing=audit['peak_landing_descent_m_s']<.025)
    gates['final_four_supports']=min(audit['final_normal_force_N'])>=1.
    gates['planted_on_long_tip']=abs(audit['final_tip_bottom_m'][leg])<.003
    final_values=dict(planted=(audit['final_normal_force_N'][leg],2.),hub_angle=(audit['final_hub_error_rad'],.035),settled=(audit['final_speed_rad_s'],.1),final_four_supports=(min(audit['final_normal_force_N']),1.),planted_on_long_tip=(abs(audit['final_tip_bottom_m'][leg]),.003))
    audit['final_gate_violations']={key:dict(value=value,threshold=threshold) for key,(value,threshold) in final_values.items() if not gates[key]}
    if audit['first_sampled_gate_violation'] is None and audit['final_gate_violations']:
        audit['first_sampled_gate_violation']=dict(time_s=float(r.d.time),phase='planted_hold_end',violations=audit['final_gate_violations'])
    audit['gates']=gates;audit['status']='PASS_NOMINAL_SINGLE_FLIP' if all(gates.values()) else 'FAILED'
    if not cad:audit['minimum_cad_gap_m']=None
    (output/'end_state.json').write_text(json.dumps(r.snapshot(last_command),indent=2))
    audit['end_state_sha256']=hashlib.sha256((output/'end_state.json').read_bytes()).hexdigest()
    (output/'audit.json').write_text(json.dumps(audit,indent=2));(output/'trace.json').write_text(json.dumps(trace));(output/'candidate.json').write_text(json.dumps(source,indent=2))
    print(json.dumps(audit,indent=2),flush=True)
    return audit

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--video',action='store_true');p.add_argument('--friction',type=float,default=.8);p.add_argument('--landing-delta',type=float,default=.15);p.add_argument('--direction',type=int,choices=[-1,1],default=-1);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();run(a.candidate,a.output,a.video,a.friction,a.landing_delta,a.direction,a.seed)

if __name__=='__main__':main()
