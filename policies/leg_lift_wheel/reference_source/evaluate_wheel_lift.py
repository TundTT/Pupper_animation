"""Native MuJoCo audit of an exported policy with simulated button presses.

Uses actual hub position PD and measured geometric gates. A failed run is saved
as a failure; button timing never forces clearance or successful alignment.
"""
import argparse,json,math
from pathlib import Path
import mujoco
import numpy as np
from workspace import wheel_lift_config as c
from workspace.wheel_lift_sequence import LiftAlignSequence


def actor(net,obs):
    x=obs
    for layer in net['layers']:
        weight,bias=layer['weights']; x=x@np.asarray(weight)+np.asarray(bias)
        name=layer['activation']
        if name=='elu':x=np.where(x>0,x,np.expm1(np.minimum(x,0)))
        elif name=='tanh':x=np.tanh(x)
        elif name not in ('linear',''):raise ValueError('Unsupported activation '+name)
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');p.add_argument('--duration',type=float,default=180.);p.add_argument('--seed',type=int,default=0);p.add_argument('--sensor-noise',action='store_true');p.add_argument('--friction-scale',type=float,default=1.)
    a=p.parse_args();net=json.loads(a.policy.read_text())
    import hashlib
    if net.get('contract')!='leg-lift-wheel-position-v1' or net.get('model_sha256')!=hashlib.sha256(c.resolve_model_path().read_bytes()).hexdigest():
        raise ValueError('Expected a wheel-lift position export matching the pinned model')
    m=mujoco.MjModel.from_xml_path(str(c.resolve_model_path()));m.geom_friction[:,0]*=a.friction_scale;d=mujoco.MjData(m);rng=np.random.default_rng(a.seed)
    mujoco.mj_resetDataKeyframe(m,d,0);d.ctrl[:]=c.DEFAULT_POSE
    for _ in range(round(2/m.opt.timestep)):mujoco.mj_step(m,d)
    wheels=np.array([m.geom('leg_'+leg+'_3_wheel_collision').id for leg in c.FOOT_ROW_BY_LEG])
    body=m.body('base_link').id
    seq=LiftAlignSequence(np.zeros(4),d.qpos[7+c.HUBS]);seq.press()
    action=np.zeros(8);last_action=action.copy(); dwell=0.; events=[];reason='timeout';peak_tilt=0.;max_clearance=np.zeros(4)
    start_xy=d.qpos[:2].copy();peak_drift=0.;trace=[];writer=None;renderer=None
    if a.video:
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw
        renderer=mujoco.Renderer(m,height=480,width=640)
        writer=imageio.get_writer(str(a.out.with_suffix('.mp4')),fps=25,codec='libx264',quality=7)
    # Two seconds pose, then only advance when measured readiness qualifies.
    for tick in range(round(a.duration/m.opt.timestep)):
        dt=m.opt.timestep
        axis=d.geom_xmat[wheels].reshape(4,3,3)[:,2,2]
        bottoms=d.geom_xpos[wheels,2]-.048*np.sqrt(np.maximum(1-axis**2,0))-.01675*np.abs(axis)
        rotation=d.xmat[body].reshape(3,3);gravity=rotation.T@np.array([0.,0.,-1.])
        angular=d.sensor('body_gyro').data.copy();tilt=math.acos(float(np.clip(-gravity[2],-1,1)))
        peak_tilt=max(peak_tilt,tilt);drift=float(np.linalg.norm(d.qpos[:2]-start_xy));peak_drift=max(peak_drift,drift)
        if drift>.09:reason='body drift';break
        if tilt>.4 or d.qpos[2]<.08:reason='body instability';break
        stable=tilt<.12 and np.linalg.norm(angular)<.3
        supported=stable and np.ptp(bottoms)<.006 and np.max(np.abs(d.qvel[6:]))<.1
        k=seq.leg
        # Radius uncertainty bounds the selected bottom against the lowest
        # supporting upper bound, matching the flat-floor geometry assumption.
        lower=d.geom_xpos[wheels,2]-.0505*np.sqrt(np.maximum(1-axis**2,0))-.01675*np.abs(axis)
        upper=d.geom_xpos[wheels,2]-.0455*np.sqrt(np.maximum(1-axis**2,0))-.01675*np.abs(axis)
        clearance=lower[k]-np.min(upper[np.arange(4)!=k]);max_clearance[k]=max(max_clearance[k],clearance)
        centers=d.geom_xpos[wheels];delta=centers[:,None]-centers[None,:]
        wheel_gap=np.min(np.linalg.norm(delta+np.eye(4)[:,:,None],axis=-1))-2*np.hypot(.0505,.01675)
        # Retain all non-wheel contact geometry checks via native contacts.
        collision=False
        for contact in d.contact:
            if contact.dist<0 and m.geom_bodyid[contact.geom1]!=0 and m.geom_bodyid[contact.geom2]!=0:
                collision=True
        ready=stable and clearance>.01 and wheel_gap>.01 and not collision
        condition=supported if seq.stage in ('pose','lower') else ready if seq.stage=='lift' else seq.verified
        dwell=dwell+dt if condition else 0.
        if dwell>=2.:
            previous=seq.stage
            if seq.press(supported=supported,lift_ready=ready):
                events.append(dict(time=tick*dt,leg=k,previous=previous,stage=seq.stage));dwell=0.
        seq.tick(dt,d.qpos[7+c.HUBS],d.qvel[6+c.HUBS],lift_ready=ready)
        if seq.stage=='fault':reason='alignment fault';break
        if seq.stage=='done':reason='completed';break
        if tick % round(.02/dt)==0:
            obs=np.concatenate([angular,gravity,np.eye(5)[seq.command],(d.qpos[7:]-c.DEFAULT_POSE)[c.PROXIMAL],last_action])
            if a.sensor_noise:
                obs[:3]+=rng.normal(0,.01,3);obs[3:6]+=rng.normal(0,.002,3);obs[11:19]+=rng.normal(0,.005,8)
            action=actor(net,obs);last_action=action.copy()
        d.ctrl[:]=seq.positions(action)
        if tick % round(.02/dt)==0:
            trace.append(dict(time=tick*dt,stage=seq.stage,leg=k,command=seq.command,clearance_m=float(clearance),tilt_rad=tilt,drift_m=drift,hub_positions=d.qpos[7+c.HUBS].tolist(),hub_targets=seq.reference.tolist(),verified=bool(seq.verified),wheel_gap_m=float(wheel_gap),angular_speed=float(np.linalg.norm(angular)),collision_pairs=[(m.geom(con.geom1).name,m.geom(con.geom2).name,float(con.dist)) for con in d.contact if con.dist<0 and m.geom_bodyid[con.geom1]!=0 and m.geom_bodyid[con.geom2]!=0]))
        if writer and tick % round(.04/dt)==0:
            renderer.update_scene(d,camera='tracking_cam');im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,640,64),fill='black');draw.text((8,6),f'SIMULATION | {list(c.FOOT_ROW_BY_LEG)[k]} | {seq.stage} | t={tick*dt:.1f}s\nPosition control | wheel-blind neural policy\nClearance {clearance*1000:.1f} mm | tilt {math.degrees(tilt):.1f} deg',fill='white');writer.append_data(np.asarray(im))
        mujoco.mj_step(m,d)
    if writer:writer.close();renderer.close()
    a.out.with_suffix('.trace.json').write_text(json.dumps(trace)+'\n')
    result=dict(passed=reason=='completed',reason=reason,simulation=True,events=events,seed=a.seed,sensor_noise=a.sensor_noise,friction_scale=a.friction_scale,
        maximum_clearance_m=max_clearance.tolist(),peak_tilt_rad=peak_tilt,peak_drift_m=peak_drift,
        final_hub_positions=d.qpos[7+c.HUBS].tolist(),goal=seq.goal.tolist(),model_sha256=net['model_sha256'])
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
