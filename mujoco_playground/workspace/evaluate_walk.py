"""Evaluate a saved walking policy and optionally render the actual contact proxies."""
import argparse
from pathlib import Path
import hashlib
import json
import jax
from jax import numpy as jp
import numpy as np
import mujoco
from brax.io import model
from brax.training.agents.ppo import networks
from brax.training.acme import running_statistics
from workspace.walk_config import get_config
from workspace.walk_env import PupperWalkEnv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params',required=True)
    p.add_argument('--model_path')
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--command',type=float,nargs=3,default=[.2,0.,0.])
    p.add_argument('--video',help='Optional MP4 path (requires mediapy/ffmpeg)')
    args=p.parse_args()
    metadata=json.loads((Path(args.params).parent/'run.json').read_text())
    c=get_config();c.update(metadata['config']);c.sensor_noise=0.;c.latency_probability=0.;c.reset_joint_noise=0.
    c.command_hold_steps=args.steps+1
    env=PupperWalkEnv(c,args.model_path)
    if args.steps <= 0:raise ValueError('--steps must be positive')
    if hashlib.sha256(Path(__file__).with_name('ring_outline.json').read_bytes()).hexdigest()!=metadata['ring_sha256']:
        raise ValueError('Ring outline differs from training.')
    if hashlib.sha256(Path(env.model_path).read_bytes()).hexdigest()!=metadata['model_sha256']:
        raise ValueError('Model differs from training; use the matching model and mesh assets.')
    net=networks.make_ppo_networks(env.observation_size,12,preprocess_observations_fn=running_statistics.normalize,policy_hidden_layer_sizes=tuple(c.hidden_layer_sizes),activation=jax.nn.elu)
    policy=jax.jit(networks.make_inference_fn(net)(model.load_params(args.params),deterministic=True))
    state=jax.jit(env.reset)(jax.random.PRNGKey(1))
    command=jp.array(args.command)
    state.info['command']=command;state.info['initial_command']=command
    # All four history frames must describe the requested evaluation command.
    obs=state.obs.reshape(c.observation_history,36).at[:,6:9].set(command)
    state=state.replace(obs=obs.reshape(-1))
    step=jax.jit(env.step); frames=[]; records=[]; key=jax.random.PRNGKey(2)
    renderer=mujoco.Renderer(env.mj_model,height=480,width=640) if args.video else None
    data=mujoco.MjData(env.mj_model)
    try:
        for i in range(args.steps):
            key,pk=jax.random.split(key); action,_=policy(state.obs,pk)
            state=step(state,action)
            records.append({k:float(v) for k,v in state.metrics.items()})
            if renderer is not None:
                data.qpos[:]=np.asarray(state.pipeline_state.qpos);mujoco.mj_forward(env.mj_model,data)
                renderer.update_scene(data,camera='tracking_cam');frames.append(renderer.render())
            if float(state.done):break
    finally:
        if renderer is not None:renderer.close()
    report={'steps':len(records),'fell':bool(float(state.done)), 'means':{k:float(np.mean([r[k] for r in records])) for k in records[0]}}
    print(json.dumps(report,indent=2))
    if args.video:
        import mediapy
        mediapy.write_video(args.video,frames,fps=round(1/env.dt))

if __name__=='__main__':main()
