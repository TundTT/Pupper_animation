"""Brax autoreset with task-state parity, including commands and wheel snapshots."""
import jax
from jax import numpy as jp
from brax.envs import training


class TaskAutoResetWrapper(training.AutoResetWrapper):
    def reset(self,rng):
        state=self.env.reset(rng)
        # Wrapper termination/episode statistics describe the transition that
        # just ended; PPO must receive them intact (especially truncation).
        wrapper_keys={'steps','truncation','episode_done','episode_metrics'}
        state.info['initial_task_info']={k:v for k,v in state.info.items() if k not in wrapper_keys}
        state.info['first_pipeline_state']=state.pipeline_state
        state.info['first_obs']=state.obs
        return state

    def step(self,state,action):
        state=super().step(state,action)
        def reset_value(initial,current):
            done=state.done.reshape(state.done.shape+(1,)*(current.ndim-state.done.ndim))
            return jp.where(done,initial,current)
        for key,initial in state.info['initial_task_info'].items():
            # Keep the noise/latency random stream moving across episodes.
            if key!='rng':
                state.info[key]=jax.tree.map(reset_value,initial,state.info[key])
        return state


def wrap_for_training(env,episode_length=1200,action_repeat=1,randomization_fn=None):
    env=training.VmapWrapper(env) if randomization_fn is None else training.DomainRandomizationVmapWrapper(env,randomization_fn)
    return TaskAutoResetWrapper(training.EpisodeWrapper(env,episode_length,action_repeat))
