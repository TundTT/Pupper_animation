"""Train lift-only on the wheeled robot; alignment never enters training inputs."""
import argparse
import functools
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import jax
import imageio.v2 as imageio
from brax.io import model
from brax.training.agents.ppo import networks, train
from workspace import wheel_lift_config as c
from workspace.wheel_lift_env import PupperWheelLiftEnv


_ROLLOUT_CACHE = {}

def rollout(env, policy, seed=0):
    """Actual deterministic checkpoint rollout, including stand between legs."""
    import numpy as np
    import jax.numpy as jp
    rng=jax.random.PRNGKey(seed)
    if id(env) not in _ROLLOUT_CACHE:
        _ROLLOUT_CACHE[id(env)] = (jax.jit(env.reset),jax.jit(env.step))
    reset,step=_ROLLOUT_CACHE[id(env)]
    state=reset(rng); infer=jax.jit(policy)
    trajectory=[]; terminated=False
    for command in [0,2,0,1,0,3,0,4,0]:
        for _ in range(200):
            state.info['command']=jp.int32(command)
            state.info['command_switch_step']=jp.int32(2_000_000_000)
            # Update the current frame command immediately, preserving history.
            state=state.replace(obs=state.obs.at[6:11].set(jax.nn.one_hot(command,5)))
            rng,key=jax.random.split(rng); action,_=infer(state.obs,key)
            state=step(state,action); trajectory.append(state.pipeline_state)
            if bool(state.done):terminated=True;break
        if terminated:break
    frames=np.asarray(env.render(trajectory[::2],camera='tracking_cam'))
    return frames,dict(seed=seed,early_termination=terminated,steps=len(trajectory),simulation=True)


def training_wrapper(env, episode_length, action_repeat, randomization_fn=None):
    from brax.envs.wrappers import training
    import jax.numpy as jp
    if randomization_fn is not None:
        raise ValueError('Physics randomization requires a separately validated wrapper')
    class ResetTask(training.AutoResetWrapper):
        def reset(self, rng):
            state = super().reset(rng)
            keys = ('step','command','command_switch_step','last_act','last_vel','action_buffer','init_xy','init_forward')
            state.info['task_initial'] = {k: state.info[k] for k in keys}
            return state
        def step(self, state, action):
            state = super().step(state, action)
            for key, initial in state.info['task_initial'].items():
                current = state.info[key]
                mask = state.done.reshape(state.done.shape + (1,)*(current.ndim-state.done.ndim))
                state.info[key] = jp.where(mask, initial, current)
            return state
    return ResetTask(training.EpisodeWrapper(training.VmapWrapper(env),episode_length,action_repeat))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-timesteps',type=int,default=20_000_000)
    p.add_argument('--num-envs',type=int,default=1024)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--offline',action='store_true')
    p.add_argument('--restore',type=Path)
    p.add_argument('--num-evals',type=int,default=15)
    p.add_argument('--config-overrides',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    config=c.get_config(); config.ppo.num_timesteps=a.num_timesteps; config.ppo.num_envs=a.num_envs
    config.ppo.seed=a.seed; config.ppo.episode_length=config.episode_length;config.ppo.num_evals=a.num_evals
    if a.config_overrides:
        overrides=json.loads(a.config_overrides.read_text())
        for dotted,value in overrides.items():
            target=config
            fields=dotted.split('.')
            for field in fields[:-1]:target=target[field]
            if fields[-1] not in target:raise ValueError('Unknown config override: '+dotted)
            target[fields[-1]]=value
    record=dict(restore_sha256=hashlib.sha256(a.restore.read_bytes()).hexdigest() if a.restore else None,config=config.to_dict(),model=json.loads((c.DEFAULT_MODEL_PATH.parent/'provenance.json').read_text()),
                contract='leg-lift-wheel-position-v1',actions=8,frame_size=27,training_wheel_rotation=False,
                physics_randomization=False,created=datetime.now(timezone.utc).isoformat(),
                source_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*wheel_lift*.py')})
    (a.output/'config.json').write_text(json.dumps(record,indent=2)+'\n')
    import wandb
    run=wandb.init(entity='QuadMorph',project='wheel-leg lift and align triangle base',
                   name=a.output.name,config=record,mode='offline' if a.offline else 'online',dir=str(a.output))
    env=PupperWheelLiftEnv(config); evaluation=PupperWheelLiftEnv(config)
    factory=functools.partial(networks.make_ppo_networks,policy_hidden_layer_sizes=config.policy.hidden_layer_sizes,
                               value_hidden_layer_sizes=config.policy.hidden_layer_sizes,activation=jax.nn.elu)
    def progress(step,metrics):
        print(step,metrics,flush=True);run.log(metrics,step=int(step))
    def checkpoint(step,make_policy,params):
        path=a.output/f'params_{int(step)}';model.save_params(str(path),params)
        # Failure stays visible and the checkpoint is already saved.
        frames,audit=rollout(evaluation,make_policy(params,deterministic=True),a.seed)
        video=a.output/f'rollout_{int(step)}.mp4';imageio.mimwrite(str(video),frames,fps=25,codec='libx264',quality=7)
        audit['checkpoint_step']=int(step);audit['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        (a.output/f'rollout_{int(step)}.json').write_text(json.dumps(audit,indent=2)+'\n')
        run.log({'rollout/video':wandb.Video(str(video),fps=25,format='mp4'),**{'rollout/'+k:v for k,v in audit.items()}},step=int(step))
    try:
        _,params,_=train.train(environment=env,eval_env=evaluation,network_factory=factory,
            progress_fn=progress,policy_params_fn=checkpoint,wrap_env_fn=training_wrapper,
            restore_params=model.load_params(str(a.restore)) if a.restore else None,**dict(config.ppo))
        model.save_params(str(a.output/'mjx_params'),params)
    finally:run.finish()

if __name__=='__main__':main()
