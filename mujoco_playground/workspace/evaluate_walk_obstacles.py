"""Native MuJoCo obstacle stress test for the exported walking interface.

No terrain observations are supplied. Compare policies using this same evaluator;
native stepping is an independent stress test, not bit-identical MJX evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import jax
import mujoco
import numpy as np
import mediapy
from brax.io import model
from brax.training.agents.ppo import networks
from brax.training.acme import running_statistics
from workspace.walk_config import get_config
from workspace.walk_env import PupperWalkEnv
from workspace import walk_geometry as g


def add_obstacles(env,height,offset=0.):
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'model.xml'
        g.save_effective_model(env.mj_model,path,env.model_path)
        spec=mujoco.MjSpec.from_file(str(path))
    centers=np.array([.32,.57,.82])+offset
    if height>0:
        for i,x in enumerate(centers):
            spec.worldbody.add_geom(name=f'obstacle_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x,0,height/2],size=[.015,.5,height/2],contype=1,conaffinity=0,
                friction=[2.,.005,.0001],solref=[.008,1.],solimp=[.99,.99,.001,.5,2.],
                rgba=[.85,.25,.12,1.])
    return spec.compile(),centers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params',required=True);p.add_argument('--output_dir',required=True)
    p.add_argument('--foot_model',choices=('rigid_flush','legacy_soft'),help='Explicit physics transfer test; default uses saved mode')
    p.add_argument('--heights',type=float,nargs='+',default=[0.,.005,.010])
    p.add_argument('--seeds',type=int,default=3);p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--speed',type=float,default=.2);p.add_argument('--video',action='store_true')
    args=p.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if min(args.heights)<0 or args.speed<=0 or args.seconds<=0 or args.seeds<1:
        raise ValueError('Use nonnegative heights and positive speed, seconds and seeds')
    meta=json.loads((Path(args.params).parent/'run.json').read_text())
    c=get_config();c.update(meta['config']);c.foot_model=args.foot_model or meta['config'].get('foot_model','legacy_soft')
    c.planned_swing_foot_weights=tuple(meta['config'].get('planned_swing_foot_weights',(1.,1.,1.,1.)))
    c.sensor_noise=0.;c.latency_probability=0.;c.reset_joint_noise=0.;c.push_probability=0.;c.floor_friction=2.
    env=PupperWalkEnv(c)
    if hashlib.sha256(Path(env.model_path).read_bytes()).hexdigest()!=meta['model_sha256']:
        raise ValueError('Checkpoint source model differs from the obstacle-test model')
    net=networks.make_ppo_networks(env.observation_size,12,preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(c.hidden_layer_sizes),activation=jax.nn.elu)
    policy=jax.jit(networks.make_inference_fn(net)(model.load_params(args.params),deterministic=True))
    reset=jax.jit(env.reset);cmd=np.array([args.speed,0.,0.]);rows=[]
    for height in args.heights:
        for seed in range(args.seeds):
            # Refresh the compiled source before mj_saveLastXML; obstacle models
            # from previous cases must not become the source of the next case.
            env=PupperWalkEnv(c)
            m,centers=add_obstacles(env,height,seed*.035)
            # Adding world geoms changes compiled geom IDs; resolve the feet
            # again instead of reusing IDs from the obstacle-free model.
            feet=np.array([m.geom(f'leg_{leg}_3_foot_collision').id for leg in g.LEGS])
            obstacle_ids={m.geom(f'obstacle_{i}').id for i in range(3)} if height else set()
            d=mujoco.MjData(m);state=reset(jax.random.PRNGKey(seed));d.qpos[:]=np.asarray(state.pipeline_state.qpos)
            mujoco.mj_forward(m,d)
            history=np.asarray(state.obs).reshape(c.observation_history,36).copy();history[:,6:9]=cmd
            side_hits=np.zeros(4,dtype=int);top_hits=np.zeros(4,dtype=int);ring_overlap=np.zeros(4,dtype=int)
            bad_hits=0;fallen=False;velocity_errors=[];frames=[];samples=0
            render=args.video and height==max(args.heights) and seed==0
            renderer=mujoco.Renderer(m,480,640) if render else None
            camera=mujoco.MjvCamera();camera.distance=.65;camera.azimuth=90;camera.elevation=-12
            for step in range(round(args.seconds/c.ctrl_dt)):
                action=np.asarray(policy(history.ravel(),jax.random.PRNGKey(0))[0]);action=np.clip(action,-1,1)
                d.ctrl[:]=np.clip(np.asarray(env.home)+action*np.asarray(env.action_scale),m.jnt_range[1:,0],m.jnt_range[1:,1])-np.asarray(env.home)
                for _ in range(env.frames):
                    mujoco.mj_step(m,d);samples+=1
                    for contact in d.contact:
                        if contact.dist>0 or not obstacle_ids.intersection(contact.geom):continue
                        foot=np.flatnonzero(np.isin(feet,contact.geom))
                        if len(foot):
                            (top_hits if abs(contact.frame[2])>.7 else side_hits)[foot[0]]+=1
                        else:bad_hits+=1
                    if height:
                        points=g.world_points(d.xpos[env.body_ids],d.xmat[env.body_ids],np.asarray(env.ring_local))[:,~np.asarray(env.bottom_mask)]
                        inside=(np.abs(points[:,:,None,0]-centers)<.015)&(np.abs(points[:,:,None,1])<.5)&(points[:,:,None,2]<height)&(points[:,:,None,2]>0.)
                        ring_overlap+=inside.any(axis=(1,2))
                mujoco.mj_forward(m,d)
                rot=d.xmat[env.torso].reshape(3,3)
                frame=np.r_[rot.T@d.cvel[env.torso,:3],rot.T@np.array([0.,0.,-1.]),cmd,[0.,0.,1.],d.qpos[7:]-np.asarray(env.home),action]
                history=np.roll(history,1,axis=0);history[0]=frame
                torso_v=g.point_velocities(d.xpos[env.torso][None,None,:],d.subtree_com[env.torso][None,:],d.cvel[env.torso][None,:])[0,0]
                if step>=50:velocity_errors.append(np.linalg.norm((rot.T@torso_v)[:2]-cmd[:2]))
                if renderer:
                    camera.lookat[:]=d.xpos[env.torso];renderer.update_scene(d,camera=camera);frames.append(renderer.render().copy())
                fallen=bool(np.arccos(np.clip(rot[2,2],-1,1))>c.terminal_tilt or d.xpos[env.torso,2]<c.terminal_height or not np.isfinite(d.qpos).all())
                if fallen:break
            if renderer:
                renderer.close();mediapy.write_video(out/f'obstacles_{height*1000:g}mm.mp4',frames,fps=50)
            row=dict(height_mm=height*1000,seed=seed,fallen=fallen,finished=bool(not fallen and d.qpos[0]>centers[-1]+.15),
                torso_x_m=float(d.qpos[0]),velocity_error=float(np.mean(velocity_errors)),foot_side_contact_samples=side_hits.tolist(),
                foot_top_contact_samples=top_hits.tolist(),ring_side_obstacle_overlap_fraction=(ring_overlap/samples).tolist(),body_obstacle_contact_samples=bad_hits)
            rows.append(row);print(json.dumps(row),flush=True)
            (out/'report.json').write_text(json.dumps(dict(protocol='native MuJoCo, 250 Hz, blind policy, friction 2; three 30mm-wide transverse bars',
                params=str(Path(args.params).resolve()),params_sha256=hashlib.sha256(Path(args.params).read_bytes()).hexdigest(),
                evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                foot_model=c.foot_model,speed=args.speed,seconds=args.seconds,rows=rows),indent=2))


if __name__=='__main__':main()
