"""Independent per-leg native-MuJoCo lift/hold/lower audit of a real export."""
import argparse,hashlib,json,math
from pathlib import Path
import mujoco
import numpy as np
from workspace import wheel_lift_config as c
from workspace.evaluate_wheel_lift import actor


def run(policy_path,out,video=False,seconds=12.):
    net=json.loads(Path(policy_path).read_text())
    if net.get('contract')!='leg-lift-wheel-position-v1':raise ValueError('Wrong policy contract')
    path=c.resolve_model_path()
    if net['model_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('Wrong robot model')
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    wheels=np.array([m.geom('leg_'+leg+'_3_wheel_collision').id for leg in c.FOOT_ROW_BY_LEG]);body=m.body('base_link').id
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    renderer=None;writer=None
    if video:
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw
        renderer=mujoco.Renderer(m,height=480,width=640)
        writer=imageio.get_writer(str(out/'lift_audit.mp4'),fps=25,codec='libx264',quality=7)
    results=[]
    try:
        for leg,command in enumerate([2,1,3,4]):
            mujoco.mj_resetDataKeyframe(m,d,0);d.ctrl[:]=c.DEFAULT_POSE
            for _ in range(round(2/m.opt.timestep)):mujoco.mj_step(m,d)
            qstart=d.qpos.copy();last=np.zeros(8);trace=[];failure=None
            for i in range(round((seconds+4)/m.opt.timestep)):
                t=i*m.opt.timestep;cmd=0 if t<2 or t>=seconds else command
                r=d.xmat[body].reshape(3,3);gravity=r.T@np.array([0.,0.,-1.]);angular=d.sensor('body_gyro').data.copy()
                if i%round(.02/m.opt.timestep)==0:
                    obs=np.r_[angular,gravity,np.eye(5)[cmd],(d.qpos[7:]-c.DEFAULT_POSE)[c.PROXIMAL],last]
                    last=actor(net,obs);d.ctrl[:]=c.DEFAULT_POSE;d.ctrl[c.PROXIMAL]+=last*c.ACTION_SCALE[c.PROXIMAL]
                    d.ctrl[c.PROXIMAL]=np.clip(d.ctrl[c.PROXIMAL],c.JOINT_LOWER_LIMITS[c.PROXIMAL],c.JOINT_UPPER_LIMITS[c.PROXIMAL]);d.ctrl[c.HUBS]=0
                    axis=d.geom_xmat[wheels].reshape(4,3,3)[:,2,2]
                    z=d.geom_xpos[wheels,2]-.048*np.sqrt(np.maximum(1-axis**2,0))-.01675*np.abs(axis)
                    tilt=math.acos(float(np.clip(-gravity[2],-1,1)));drift=np.linalg.norm(d.qpos[:2]-qstart[:2])
                    centers=d.geom_xpos[wheels];gap=float(np.min(np.linalg.norm(centers[:,None]-centers[None,:]+np.eye(4)[:,:,None],axis=-1))-2*np.hypot(.0505,.01675))
                    trace.append(dict(t=t,command=cmd,clearance=float(z[leg]),other_bottoms=z[np.arange(4)!=leg].tolist(),tilt=tilt,drift=float(drift),hub_speed=float(np.max(np.abs(d.qvel[6+c.HUBS]))),wheel_gap_m=gap))
                    if tilt>.4 or d.qpos[2]<.10 or drift>.09:failure='body instability';break
                if video and i%round(.04/m.opt.timestep)==0:
                    renderer.update_scene(d,camera='tracking_cam');im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                    draw.rectangle((0,0,640,50),fill='black');draw.text((8,8),f'SIMULATION | {list(c.FOOT_ROW_BY_LEG)[leg]} | t={t:.1f}s | lift-only policy\nwheel clearance {trace[-1]["clearance"]*1000:.1f} mm | tilt {math.degrees(trace[-1]["tilt"]):.1f} deg',fill='white')
                    writer.append_data(np.asarray(im))
                mujoco.mj_step(m,d)
            hold=[x for x in trace if 5<=x['t']<seconds]
            lower=[x for x in trace if x['t']>=seconds+2]
            result=dict(leg=list(c.FOOT_ROW_BY_LEG)[leg],failure=failure,seconds=trace[-1]['t'],
                hold_mean_clearance=float(np.mean([x['clearance'] for x in hold])) if hold else 0.,
                hold_min_clearance=min([x['clearance'] for x in hold],default=0.),
                hold_max_tilt=max([x['tilt'] for x in hold],default=10.),
                minimum_wheel_gap_m=min([x['wheel_gap_m'] for x in trace],default=-1.),
                lowered=bool(lower) and max(abs(x['clearance']) for x in lower)<.006,
                support_max_clearance=max([max(x['other_bottoms']) for x in hold],default=1.))
            result['passed']=failure is None and result['hold_min_clearance']>.0125 and result['hold_max_tilt']<.12 and result['support_max_clearance']<.006 and result['lowered'] and result['minimum_wheel_gap_m']>.01
            results.append(result);(out/(result['leg']+'.trace.json')).write_text(json.dumps(trace)+'\n')
    finally:
        if writer:writer.close()
        if renderer:renderer.close()
    result=dict(policy=str(policy_path),policy_sha256=hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),simulation=True,passed=all(x['passed'] for x in results),legs=results)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',required=True);p.add_argument('--out',required=True);p.add_argument('--video',action='store_true');p.add_argument('--seconds',type=float,default=12.);a=p.parse_args()
    run(a.policy,a.out,a.video,a.seconds)
