"""Training-only task-state critic; deployed actor remains sensor-only and unchanged."""

import dataclasses

import jax
from jax import numpy as jp
from brax.training import networks
from brax.training.acme import running_statistics
from workspace.wheel_align_direct_env import PupperWheelAlignDirectEnv


class PupperWheelAlignCriticEnv(PupperWheelAlignDirectEnv):
    # Critic context is a separate training-only entry, never an actor feature.
    critic_observation_size = 51

    @property
    def observation_size(self):
        return {"sensors": self._obs_history*self._single_obs_dim,
                "critic": self.critic_observation_size}

    def _with_critic(self, state):
        ps, info = state.pipeline_state, state.info
        torso = ps.x.pos[self._torso_idx - 1]
        context = jp.concatenate(
            (
                self._wheel_clearance(ps) / 0.02,
                self._wheel_loads(ps) / 10.0,
                jax.nn.one_hot(info["phase"], 5).reshape(-1),
                info["rotation_verified"].astype(jp.float32),
                info["completed"].astype(jp.float32),
                info["initial_completed"].astype(jp.float32),
                (torso[:2] - info["init_xy"]) / 0.05,
                ps.qd[:3],
                torso[2:3] / 0.2,
                info["phase_age"].astype(jp.float32) / 100.0,
                jp.atleast_1d(
                    info["step"].astype(jp.float32) / self._config.episode_length
                ),
            )
        )
        if context.shape != (self.critic_observation_size,):
            raise ValueError("Unexpected critic context shape")
        return state.replace(obs={"sensors": state.obs, "critic": context})

    def reset(self, rng):
        return self._with_critic(super().reset(rng))

    def step(self, state, action):
        state = state.replace(obs=state.obs["sensors"])
        return self._with_critic(super().step(state, action))


def with_task_critic(base, critic_params, sensor_size, context_size, hidden_layers):
    """Reuse actor exactly; zero-expand the existing critic's first layer."""
    policy_apply = base.policy_network.apply

    def actor_apply(normalizer, params, observation):
        sensor = (
            observation["sensors"] if isinstance(observation, dict) else observation
        )
        return policy_apply(normalizer, params, sensor)

    value = networks.make_value_network(
        sensor_size + context_size,
        hidden_layer_sizes=hidden_layers,
        activation=jax.nn.elu,
    )

    def value_apply(normalizer, params, observation):
        features = jp.concatenate(
            (
                running_statistics.normalize(observation["sensors"], normalizer),
                observation["critic"],
            ),
            axis=-1,
        )
        return value.apply(None, params, features)

    critic_params = jax.tree.map(lambda a: a, critic_params)
    first = critic_params["params"]["hidden_0"]["kernel"]
    if first.shape[0] == sensor_size:
        critic_params["params"]["hidden_0"]["kernel"] = jp.pad(
            first, ((0, context_size), (0, 0))
        )
    elif first.shape[0] != sensor_size + context_size:
        raise ValueError(
            "Critic checkpoint input shape does not match sensor/context contract"
        )
    result = dataclasses.replace(
        base,
        policy_network=dataclasses.replace(base.policy_network, apply=actor_apply),
        value_network=dataclasses.replace(value, apply=value_apply),
    )
    return result, critic_params
