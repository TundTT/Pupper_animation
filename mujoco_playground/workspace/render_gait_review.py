"""Render the expanded, push-free 50 fps gait review for a saved checkpoint."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import jax
import mediapy
from brax.io import model
from brax.training.agents.ppo import networks
from brax.training.acme import running_statistics
from workspace.walk_config import get_config
from workspace.walk_env import PupperWalkEnv
from workspace.train_walk import render_showcase

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params',required=True);p.add_argument('--video',required=True)
    p.add_argument('--seed',type=int,default=2)
    p.add_argument('--floor_friction',type=float,help='Override floor friction for a diagnostic video')
    p.add_argument('--foot_model',choices=('rigid_flush','legacy_soft'),help='Explicit physics transfer test; default uses saved model')
    p.add_argument('--command',type=float,nargs=3,metavar=('VX','VY','YAW'),help='Render a single fixed command')
    p.add_argument('--seconds',type=float,default=8.,help='Duration with --command')
    args=p.parse_args();meta=json.loads((Path(args.params).parent/'run.json').read_text())
    c=get_config();c.update(meta.get('eval_config',meta['config']))
    c.foot_model=args.foot_model or meta['config'].get('foot_model','legacy_soft')
    c.planned_swing_foot_weights=tuple(meta['config'].get('planned_swing_foot_weights',(1.,1.,1.,1.)))
    if args.floor_friction is not None:c.floor_friction=args.floor_friction
    c.sensor_noise=0.;c.latency_probability=0.;c.push_probability=0.;c.reset_joint_noise=0.
    env=PupperWalkEnv(c)
    if hashlib.sha256(Path(env.model_path).read_bytes()).hexdigest()!=meta['model_sha256']:raise ValueError('Model differs from training')
    net=networks.make_ppo_networks(env.observation_size,12,preprocess_observations_fn=running_statistics.normalize,policy_hidden_layer_sizes=tuple(c.hidden_layer_sizes),activation=jax.nn.elu)
    policy=networks.make_inference_fn(net)(model.load_params(args.params),deterministic=True)
    kwargs={} if args.command is None else dict(commands=[args.command],steps_per_command=round(args.seconds/c.ctrl_dt))
    if args.seconds<=0.:raise ValueError('seconds must be positive')
    frames,fps=render_showcase(env,policy,jax.random.PRNGKey(args.seed),c.observation_history,**kwargs)
    video=Path(args.video);video.parent.mkdir(parents=True,exist_ok=True)
    mediapy.write_video(str(video),frames,fps=fps)
    print(json.dumps(dict(video=str(video),frames=len(frames),fps=fps,floor_friction=float(env.mj_model.geom_friction[env.floor,0]),command=args.command)),flush=True)

if __name__=='__main__':main()
