"""PPO walking entry point. Use the venv Python directly, not uv run."""
import argparse
from datetime import datetime
from pathlib import Path
import functools
import hashlib
import json
import platform
import importlib.metadata
import jax
from brax.io import model
from brax.training.agents.ppo import networks, train as ppo
from workspace.walk_env import PupperWalkEnv
from workspace.walk_config import get_config
from workspace.walk_randomize import domain_randomize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_timesteps',type=int)
    parser.add_argument('--num_envs',type=int)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--model_path')
    parser.add_argument('--output_dir',default=str(Path(__file__).with_name('output')))
    parser.add_argument('--smoke',action='store_true',help='Tiny CPU-compatible PPO integration check; not a trained policy')
    parser.add_argument('--init_params',help='Warm-start weights, not optimizer-state resume')
    parser.add_argument('--use_wandb',action='store_true')
    parser.add_argument('--wandb_project',default='pupper-leg-walk')
    parser.add_argument('--no_randomization',action='store_true')
    args=parser.parse_args()
    c=get_config(); c.ppo.seed=args.seed
    if args.smoke:
        c.ppo.update(num_timesteps=64,num_envs=2,num_evals=2,num_eval_envs=2,
                     unroll_length=4,batch_size=2,num_minibatches=1,num_updates_per_batch=1)
        c.episode_length=16; c.hidden_layer_sizes=(32,32)
        c.sensor_noise=0.; c.latency_probability=0.
    if args.num_timesteps is not None:c.ppo.num_timesteps=args.num_timesteps
    if args.num_envs is not None:c.ppo.num_envs=args.num_envs
    devices=jax.devices()
    print('JAX devices:',devices,flush=True)
    if not args.smoke and not any(d.platform=='gpu' for d in devices):
        raise RuntimeError('Full training requires JAX CUDA on the remote Linux host; use --smoke for a CPU check.')
    count=len(devices)
    if c.ppo.num_envs % count or c.ppo.batch_size % count:
        raise ValueError('num_envs and batch_size must be divisible by visible device count')
    if (c.ppo.batch_size*c.ppo.num_minibatches) % c.ppo.num_envs:
        raise ValueError('batch_size * num_minibatches must be divisible by num_envs')
    out=Path(args.output_dir).resolve()/f'walk_{datetime.now():%Y-%m-%d_%H-%M-%S}'
    out.mkdir(parents=True)
    env=PupperWalkEnv(c,args.model_path)
    eval_c=get_config(); eval_c.update(c.to_dict()); eval_c.sensor_noise=0.; eval_c.latency_probability=0.; eval_c.reset_joint_noise=0.
    eval_env=PupperWalkEnv(eval_c,args.model_path)
    metadata={'config':c.to_dict(),'model_path':env.model_path,
              'randomization_enabled':not (args.no_randomization or args.smoke),'smoke_test':args.smoke,
              'model_sha256':hashlib.sha256(Path(env.model_path).read_bytes()).hexdigest(),
              'ring_sha256':hashlib.sha256(Path(__file__).with_name('ring_outline.json').read_bytes()).hexdigest(),
              'home_joint_pos':env.mj_model.qpos0[7:].tolist(),
              'joint_lower_limits':env.mj_model.jnt_range[1:,0].tolist(),
              'joint_upper_limits':env.mj_model.jnt_range[1:,1].tolist(),
              'kp':env.mj_model.actuator_gainprm[:,0].tolist(),
              'kd':(-env.mj_model.actuator_biasprm[:,2]).tolist(),'platform':platform.platform(),
              'devices':[str(d) for d in devices],
              'versions':{p:importlib.metadata.version(p) for p in ('jax','jaxlib','mujoco','mujoco-mjx','brax','flax','orbax-checkpoint')}}
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    Path(out/'model.xml').write_bytes(Path(env.model_path).read_bytes())
    run=None
    if args.use_wandb:
        import wandb
        run=wandb.init(project=args.wandb_project,name=out.name,config=metadata)
    def progress(step,metrics):
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,**{k:float(v) for k,v in metrics.items()}})+'\n')
        n=max(float(metrics.get('eval/avg_episode_length',1)),1)
        mean=lambda k:float(metrics.get('eval/episode_'+k,float('nan')))/n
        print(f"{step:,} steps | reward {metrics.get('eval/episode_reward',float('nan')):.3f} | length {n:.0f} | tilt {mean('tilt_deg'):.2f} deg | feet {mean('foot_contacts'):.2f} | ring side {mean('ring_side_fraction'):.3f} | velocity error {mean('velocity_error'):.3f}",flush=True)
        if run is not None:run.log(metrics,step=step)
    def save(step,make_policy,params):
        del make_policy
        # Independent parameter snapshots survive interrupted SSH sessions/runs.
        model.save_params(str(out/f'params_{step:012d}'),params)
    factory=functools.partial(networks.make_ppo_networks,policy_hidden_layer_sizes=tuple(c.hidden_layer_sizes),activation=jax.nn.elu)
    kwargs=c.ppo.to_dict()
    make,params,_=ppo.train(environment=env,eval_env=eval_env,episode_length=c.episode_length,
                           network_factory=factory,progress_fn=progress,policy_params_fn=save,
                           randomization_fn=None if args.no_randomization or args.smoke else domain_randomize,
                           restore_params=model.load_params(args.init_params) if args.init_params else None,
                           **kwargs)
    model.save_params(str(out/'mjx_params'),params)
    print(f'Saved {out}/mjx_params',flush=True)
    if run is not None:run.finish()

if __name__=='__main__':main()
