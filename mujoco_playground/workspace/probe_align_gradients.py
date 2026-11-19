"""Measure PPO/action-supervision gradient competition without updating weights."""

import argparse
import functools
import json
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jp
from brax.io import model
from brax.training import acting
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import losses, networks
from ml_collections import ConfigDict
from workspace.supervised_align_ppo import (
    phase_balanced_sequences,
    feedback_action_loss,
)
from workspace.wheel_align_direct_env import PupperWheelAlignDirectEnv
from workspace.wheel_align_env import domain_randomize_wheel_align
from workspace.wheel_align_training import wrap_for_alignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = ConfigDict(json.loads((args.params.parent / "config.json").read_text()))
    env = PupperWheelAlignDirectEnv(cfg)
    layers = tuple(cfg.policy.hidden_layer_sizes)
    net = networks.make_ppo_networks(
        env.observation_size,
        12,
        policy_hidden_layer_sizes=layers,
        value_hidden_layer_sizes=layers,
        activation=jax.nn.elu,
        preprocess_observations_fn=rs.normalize,
    )
    norm, actor, critic = model.load_params(args.params)
    params = losses.PPONetworkParams(policy=actor, value=critic)
    policy = networks.make_inference_fn(net)((norm, actor, critic))
    wrapped = wrap_for_alignment(
        env,
        cfg.episode_length,
        1,
        functools.partial(
            domain_randomize_wheel_align,
            rng=jax.random.split(jax.random.PRNGKey(2601), 64),
        ),
    )
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(2602), 64))
    bx, by, memory = phase_balanced_sequences(
        cfg.direct.supervision_path,
        env._history_taps,
        cfg.episode_length,
        length=64,
        per_phase=32,
    )
    ids = np.random.default_rng(2603).choice(len(bx), 16, replace=False)
    bx, by, memory = bx[ids], by[ids], memory[ids]
    scale = jp.array(cfg.action_scale) / jp.tile(jp.array([0.01, 0.015, 0.05]), 4)

    @jax.jit
    def rollout(state, key):
        return acting.generate_unroll(
            wrapped, state, policy, key, 200, extra_fields=("truncation",)
        )

    @jax.jit
    def gradients(data):
        def ppo(p):
            return losses.compute_ppo_loss(
                p,
                norm,
                data,
                jax.random.PRNGKey(2604),
                net,
                entropy_cost=0.001,
                discounting=0.995,
                clipping_epsilon=0.2,
            )[0]

        def supervised(p):
            return feedback_action_loss(
                net.policy_network,
                norm,
                p.policy,
                bx,
                by,
                memory,
                env._history_taps,
                scale,
            )

        ppo_loss, pg = jax.value_and_grad(ppo)(params)
        bc_loss, bg = jax.value_and_grad(supervised)(params)
        pg, bg = jax.tree.leaves(pg.policy), jax.tree.leaves(bg.policy)
        pnorm = jp.sqrt(sum(jp.sum(g * g) for g in pg))
        bnorm = jp.sqrt(sum(jp.sum(g * g) for g in bg))
        cosine = sum(jp.sum(p * b) for p, b in zip(pg, bg)) / jp.maximum(
            pnorm * bnorm, 1e-12
        )
        return dict(
            ppo_loss=ppo_loss,
            supervision_loss=bc_loss,
            ppo_actor_gradient_norm=pnorm,
            supervision_actor_gradient_norm=bnorm,
            gradient_cosine=cosine,
            weighted_supervision_ratio=5 * bnorm / jp.maximum(pnorm, 1e-12),
        )

    reports = []
    for chunk in range(5):
        state, data = rollout(state, jax.random.PRNGKey(2700 + chunk))
        # Match the actual 32-environment, 20-tick PPO minibatch. Advancing
        # 200 ticks between probes samples different portions of the sequence.
        data = jax.tree.map(lambda a: jp.swapaxes(a, 0, 1)[:32, -20:], data)
        result = {k: float(v) for k, v in gradients(data).items()}
        result["control_ticks"] = (chunk + 1) * 200
        reports.append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
