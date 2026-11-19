"""Episode reset for a task whose persistent state must reset WITH physics."""

import dataclasses
import jax
from brax.envs.base import Wrapper
from brax.envs.wrappers import training
from brax.training.agents.ppo import networks as ppo_networks
from flax.core import unfreeze
from jax import numpy as jp


def make_alignment_networks(observation_size, action_size, initial_std=0.2, **kwargs):
    """Ordinary Brax tanh PPO, initialized at the tested zero-correction controller.

    Preserve random hidden features; zero only the final mean head and initialize
    its scale head to a small constant. Avoid discarding a working baseline via
    random actuator corrections before the first PPO update.
    """
    net = ppo_networks.make_ppo_networks(observation_size, action_size, **kwargs)
    init = net.policy_network.init
    output_name = f"hidden_{len(kwargs['policy_hidden_layer_sizes'])}"

    def supported_init(key):
        params = unfreeze(init(key))
        output = params["params"][output_name]
        assert output["kernel"].shape[-1] == 2 * action_size
        output["kernel"] = jp.zeros_like(output["kernel"])
        # Brax NormalTanhDistribution uses softplus(scale) + 0.001.
        output["bias"] = (
            jp.zeros_like(output["bias"])
            .at[action_size:]
            .set(jp.log(jp.expm1(initial_std - 0.001)))
        )
        return params

    return dataclasses.replace(
        net, policy_network=dataclasses.replace(net.policy_network, init=supported_init)
    )


class AlignAutoResetWrapper(Wrapper):
    """Reset all task state on done, retaining terminal metrics for PPO.

    Wrap outside the DR vmap: Brax's DR wrapper deliberately unwraps its inner
    env. Fresh reset candidates use that same randomized model; only finished
    rows are replaced. The reset branch runs only when at least one row is done.
    """

    def step(self, state, action):
        state.info["steps"] = jp.where(state.done, 0.0, state.info["steps"])
        state = state.replace(done=jp.zeros_like(state.done))
        stepped = self.env.step(state, action)

        def reset_finished(s):
            fresh = self.env.reset(s.info["rng"])

            def choose(new, old):
                mask = s.done.reshape(s.done.shape + (1,) * (new.ndim - s.done.ndim))
                return jp.where(mask, new, old)

            info = jax.tree.map(choose, fresh.info, s.info)
            # Brax consumes the just-finished episode's statistics and timeout
            # status alongside reward/done; everything task-specific is fresh.
            for key in ("steps", "truncation", "episode_done", "episode_metrics"):
                info[key] = s.info[key]
            return s.replace(
                pipeline_state=jax.tree.map(
                    choose, fresh.pipeline_state, s.pipeline_state
                ),
                obs=jax.tree.map(choose, fresh.obs, s.obs),
                info=info,
            )

        return jax.lax.cond(jp.any(stepped.done), reset_finished, lambda s: s, stepped)


def wrap_for_alignment(env, episode_length, action_repeat, randomization_fn=None):
    if randomization_fn is None:
        env = training.VmapWrapper(env)
    else:
        env = training.DomainRandomizationVmapWrapper(env, randomization_fn)
    env = training.EpisodeWrapper(env, episode_length, action_repeat)
    return AlignAutoResetWrapper(env)
