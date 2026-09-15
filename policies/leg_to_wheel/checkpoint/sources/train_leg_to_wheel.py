"""PPO leg-to-wheel entry point. Use the venv Python directly, not uv run."""
import argparse
from datetime import datetime
from pathlib import Path
import functools
import hashlib
import json
import platform
import shutil
import importlib.metadata
import jax
from jax import numpy as jp
import mediapy as media
import mujoco
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks, train as ppo
from workspace.leg_to_wheel_env import PupperLegToWheelEnv
from workspace.leg_to_wheel_config import get_config
from workspace.leg_to_wheel_randomize import domain_randomize

def render_showcase(env, inference_fn, rng, observation_history, steps_per_command=150, render_every=2):
    """Fixed FL → FR → BR → BL cycle with stand between every lifted hold."""
    from workspace.evaluate_leg_to_wheel import rollout
    result, frames = rollout(env, inference_fn, int(rng[1]),
                             [(0, 2.)]+[(cmd, steps_per_command*env.dt) for leg in (1,2,3,4) for cmd in (leg,0)],
                             render=True, render_every=render_every)
    return frames, round(1/env.dt/render_every)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_timesteps',type=int)
    parser.add_argument('--num_envs',type=int)
    parser.add_argument('--learning_rate',type=float)
    parser.add_argument('--entropy_cost',type=float)
    parser.add_argument('--max_devices_per_host',type=int,default=1,help='Default to one GPU; multiple require working GPU collectives')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--model_path')
    parser.add_argument('--output_dir',default=str(Path(__file__).with_name('output')))
    parser.add_argument('--smoke',action='store_true',help='Tiny CPU-compatible PPO integration check; not a trained policy')
    parser.add_argument('--init_params',help='Warm-start weights, not optimizer-state resume')
    parser.add_argument('--use_wandb',action='store_true')
    parser.add_argument('--wandb_project',default='pupper-leg-to-wheel')
    parser.add_argument('--wandb_entity',default='QuadMorph')
    parser.add_argument('--no_eval_videos',action='store_true',help='skip rendering rollout videos to W&B')
    parser.add_argument('--no_randomization',action='store_true')
    args=parser.parse_args()
    c=get_config(); c.ppo.seed=args.seed
    if args.smoke:
        c.ppo.update(num_timesteps=64,num_envs=2,num_evals=2,num_eval_envs=2,
                     unroll_length=4,batch_size=2,num_minibatches=1,num_updates_per_batch=1)
        c.episode_length=16; c.hidden_layer_sizes=(32,32)
        c.sensor_noise=0.; c.latency_probability=0.
        c.command_hold_seconds=(.04,.12)
    if args.num_timesteps is not None:c.ppo.num_timesteps=args.num_timesteps
    if args.num_envs is not None:c.ppo.num_envs=args.num_envs
    if args.learning_rate is not None:
        if args.learning_rate<=0:raise ValueError('--learning_rate must be positive')
        c.ppo.learning_rate=args.learning_rate
    if args.entropy_cost is not None:
        if args.entropy_cost<0:raise ValueError('--entropy_cost must be nonnegative')
        c.ppo.entropy_cost=args.entropy_cost
    if args.smoke:jax.config.update('jax_platforms','cpu')
    devices=jax.devices()
    print('JAX devices:',devices,flush=True)
    if not args.smoke and not any(d.platform=='gpu' for d in devices):
        raise RuntimeError('Full training requires JAX CUDA on the remote Linux host; use --smoke for a CPU check.')
    if args.max_devices_per_host<1:raise ValueError('--max_devices_per_host must be positive')
    count=min(len(devices),args.max_devices_per_host)
    if c.ppo.num_envs % count or c.ppo.batch_size % count:
        raise ValueError('num_envs and batch_size must be divisible by visible device count')
    if (c.ppo.batch_size*c.ppo.num_minibatches) % c.ppo.num_envs:
        raise ValueError('batch_size * num_minibatches must be divisible by num_envs')
    out=Path(args.output_dir).resolve()/f'leg_to_wheel_{datetime.now():%Y-%m-%d_%H-%M-%S}'
    out.mkdir(parents=True)
    env=PupperLegToWheelEnv(c,args.model_path)
    eval_c=get_config(); eval_c.update(c.to_dict()); eval_c.sensor_noise=0.; eval_c.latency_probability=0.; eval_c.reset_joint_noise=0.
    eval_env=PupperLegToWheelEnv(eval_c,args.model_path)
    metadata={'config':c.to_dict(),'model_path':env.model_path,
              'init_params':str(Path(args.init_params).resolve()) if args.init_params else None,
              'randomization_enabled':not (args.no_randomization or args.smoke),'smoke_test':args.smoke,
              'model_sha256':hashlib.sha256(Path(env.model_path).read_bytes()).hexdigest(),
              'ring_sha256':hashlib.sha256(Path(__file__).with_name('ring_outline.json').read_bytes()).hexdigest(),
              'home_joint_pos':env.mj_model.qpos0[7:].tolist(),
              'joint_lower_limits':env.mj_model.jnt_range[1:,0].tolist(),
              'joint_upper_limits':env.mj_model.jnt_range[1:,1].tolist(),
              'kp':env.mj_model.actuator_gainprm[:,0].tolist(),
              'kd':(-env.mj_model.actuator_biasprm[:,2]).tolist(),'platform':platform.platform(),
              'devices':[str(d) for d in devices[:count]],'max_devices_per_host':args.max_devices_per_host,
              'versions':{p:importlib.metadata.version(p) for p in ('jax','jaxlib','mujoco','mujoco-mjx','brax','flax','orbax-checkpoint')}}
    source_dir=out/'sources'; source_dir.mkdir()
    for name in ('leg_to_wheel_config.py','leg_to_wheel_env.py','leg_to_wheel_randomize.py',
                 'train_leg_to_wheel.py','walk_geometry.py','walk_randomize.py','ring_outline.json'):
        shutil.copy2(Path(__file__).with_name(name),source_dir/name)
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    Path(out/'model.xml').write_bytes(Path(env.model_path).read_bytes())
    run=None
    if args.use_wandb:
        import wandb
        run=wandb.init(project=args.wandb_project,entity=args.wandb_entity,name=out.name,config=metadata)
    # PPO has no built-in checkpoint selection -- it just runs to num_timesteps and
    # stops, even mid-collapse. policy_params_fn (save) sees params for a step but
    # not that step's eval reward; progress_fn (called right after, same step) sees
    # the reward but not params. `pending` bridges the two so the best-eval params
    # can be kept regardless of where training happens to end.
    pending={'step':None,'params':None}
    best={'step':None,'reward':-float('inf'),'kl':float('inf'),'params':None}
    def progress(step,metrics):
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,**{k:float(v) for k,v in metrics.items()}})+'\n')
        n=max(float(metrics.get('eval/avg_episode_length',1)),1)
        mean=lambda k:float(metrics.get('eval/episode_'+k,float('nan')))/n
        reward=float(metrics.get('eval/episode_reward',float('nan')))
        kl=float(metrics.get('training/kl_mean',0.))
        print(f"{step:,} steps | reward {reward:.3f} | length {n:.0f} | tilt {mean('tilt_deg'):.2f} deg | feet {mean('foot_contacts'):.2f} | ring side {mean('ring_side_fraction'):.3f} | clearance {mean('lifted_clearance_m')*1000:.1f} mm | kl {kl:.3f}",flush=True)
        if kl>0.15:print(f'WARNING: elevated KL ({kl:.3f}) at step {step:,} -- possible PPO collapse in progress',flush=True)
        if pending['step']==step and np.isfinite(reward) and reward>best['reward']:
            best.update(step=step,reward=reward,kl=kl,params=pending['params'])
            model.save_params(str(out/'best_params'),best['params'])
        if run is not None:run.log(metrics,step=step)
    def save(step,make_policy,params):
        # Independent parameter snapshots survive interrupted SSH sessions/runs.
        model.save_params(str(out/f'params_{step:012d}'),params)
        pending.update(step=step,params=params)
        if args.smoke or args.no_eval_videos:return
        try:
            inference_fn=make_policy(params,deterministic=True)
            frames,fps=render_showcase(eval_env,inference_fn,jax.random.PRNGKey(0),c.observation_history)
            path=out/f'rollout_step_{step:012d}.mp4'
            media.write_video(str(path),frames,fps=fps)
            if run is not None:run.log({'eval/video':wandb.Video(str(path),fps=fps,format='mp4')},step=step)
        except Exception as e:  # noqa: BLE001 -- a video hiccup must not kill a multi-hour run
            print(f'WARNING: eval video render failed at step {step}: {e!r}',flush=True)
    factory=functools.partial(networks.make_ppo_networks,policy_hidden_layer_sizes=tuple(c.hidden_layer_sizes),activation=jax.nn.elu)
    kwargs=c.ppo.to_dict()
    kwargs['max_devices_per_host']=args.max_devices_per_host
    make,params,_=ppo.train(environment=env,eval_env=eval_env,episode_length=c.episode_length,
                           network_factory=factory,progress_fn=progress,policy_params_fn=save,
                           randomization_fn=None if args.no_randomization or args.smoke else functools.partial(domain_randomize,shortening_range=tuple(c.leg_shortening_range),converted_probability=c.converted_probability),
                           restore_params=model.load_params(args.init_params) if args.init_params else None,
                           **kwargs)
    model.save_params(str(out/'mjx_params'),params)
    print(f'Saved {out}/mjx_params',flush=True)
    if not args.smoke and not args.no_eval_videos:
        try:
            inference_fn=make(params,deterministic=True)
            frames,fps=render_showcase(eval_env,inference_fn,jax.random.PRNGKey(1),c.observation_history)
            path=out/'rollout_final.mp4'
            media.write_video(str(path),frames,fps=fps)
            print(f'Final video -> {path}',flush=True)
            if run is not None:run.log({'eval/video_final':wandb.Video(str(path),fps=fps,format='mp4')})
        except Exception as e:  # noqa: BLE001
            print(f'WARNING: final video render failed: {e!r}',flush=True)
    if not args.smoke and best['params'] is not None:
        model.save_params(str(out/'best_params'),best['params'])
        print(f"Best eval checkpoint: step {best['step']:,} (reward {best['reward']:.3f}, kl {best['kl']:.3f}) -> {out}/best_params",flush=True)
        print('Use best_params for evaluation/export unless the final step matched it.',flush=True)
        if not args.no_eval_videos:
            try:
                inference_fn=make(best['params'],deterministic=True)
                frames,fps=render_showcase(eval_env,inference_fn,jax.random.PRNGKey(2),c.observation_history)
                path=out/'rollout_best.mp4'
                media.write_video(str(path),frames,fps=fps)
                print(f'Best video -> {path}',flush=True)
                if run is not None:run.log({'eval/video_best':wandb.Video(str(path),fps=fps,format='mp4')})
            except Exception as e:  # noqa: BLE001
                print(f'WARNING: best-checkpoint video render failed: {e!r}',flush=True)
    if run is not None:run.finish()

if __name__=='__main__':main()
