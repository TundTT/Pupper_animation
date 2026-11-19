"""Sensor-only actor rollout with privileged, evaluation-only stage diagnostics."""

import argparse
import functools
import json
from pathlib import Path

import jax
import numpy as np
from brax.envs.wrappers.training import DomainRandomizationVmapWrapper
from brax.io import model
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import networks
from jax import numpy as jp
from ml_collections import ConfigDict
from workspace import configs
from workspace.wheel_align_direct_env import PupperWheelAlignDirectEnv
from workspace.wheel_align_env import domain_randomize_wheel_align


def main():
    p = argparse.ArgumentParser()
    p.add_argument("params", type=Path)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--seed", type=int, default=901)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = ConfigDict(json.loads((args.params.parent / "config.json").read_text()))
    cfg.align.initial_completed_max = 0
    base_env = PupperWheelAlignDirectEnv
    if cfg.direct.get('encoder_hold', False):
        from workspace.wheel_align_refine_env import PupperWheelAlignRefineEnv
        base_env = PupperWheelAlignRefineEnv
        cfg.direct.reference_reset_probability = 0.0
        if cfg.direct.get('encoder_gate', False):
            from workspace.wheel_align_gate_env import PupperWheelAlignGateEnv
            base_env = PupperWheelAlignGateEnv

    class Probe(base_env):
        def reset(self, key):
            state = super().reset(key)
            state.info["probe"] = jp.zeros((4, 4))
            return state

        def step(self, state, action):
            state = super().step(state, action)
            state.info["probe"] = jp.stack(
                (
                    self._wheel_clearance(state.pipeline_state),
                    self._wheel_loads(state.pipeline_state),
                    self._errors(state.pipeline_state, state.info),
                    state.pipeline_state.qd[self._wheel_dof],
                ),
                axis=-1,
            )
            return state

    env = Probe(cfg)
    layers = tuple(cfg.policy.hidden_layer_sizes)
    net = networks.make_ppo_networks(
        env.observation_size,
        12,
        policy_hidden_layer_sizes=layers,
        value_hidden_layer_sizes=layers,
        activation=jax.nn.elu,
        preprocess_observations_fn=rs.normalize,
    )
    policy = networks.make_inference_fn(net)(
        model.load_params(args.params), deterministic=True
    )
    wrapped = DomainRandomizationVmapWrapper(
        env,
        functools.partial(
            domain_randomize_wheel_align,
            rng=jax.random.split(jax.random.PRNGKey(args.seed), args.count),
        ),
    )

    @jax.jit
    def rollout():
        state = wrapped.reset(
            jax.random.split(jax.random.PRNGKey(args.seed + 1), args.count)
        )

        def advance(carry, _):
            state, alive, key = carry
            key, action_key = jax.random.split(key)
            action, _ = policy(state.obs, action_key)
            nxt = wrapped.step(state, action)
            sample = (
                state.obs,
                state.info["command"],
                state.info["phase"],
                nxt.info["probe"],
                action,
                nxt.metrics["body_drift_dist"],
                alive,
                nxt.info["completed"],
            )
            return (nxt, alive & ~nxt.done.astype(bool), key), sample

        return jax.lax.scan(
            advance,
            (state, jp.ones(args.count, dtype=bool), jax.random.PRNGKey(args.seed + 2)),
            None,
            length=cfg.episode_length,
        )[1]

    arrays = jax.device_get(rollout())
    keys = ("obs", "command", "phase", "probe", "action", "drift", "alive", "completed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **dict(zip(keys, arrays)))
    obs, command, phase, probe, action, drift, alive, completed = arrays
    result = {
        "params": str(args.params),
        "completed_per_leg": completed[-1].sum(axis=0).tolist(),
    }
    for leg, row in configs.FOOT_ROW_BY_LEG.items():
        selected = (command == configs.COMMAND_STATES.index(leg)) & alive
        stages = {}
        for ph in (1, 2, 3, 4):
            mask = selected & (phase[:, :, row] == ph)
            data = probe[:, :, row][mask]
            if not len(data):
                continue
            stages[str(ph)] = dict(
                samples=len(data),
                clearance_mm=np.percentile(data[:, 0] * 1000, [5, 50, 95]).tolist(),
                load_mean=float(data[:, 1].mean()),
                angle_p50=float(np.median(np.abs(data[:, 2]))),
                speed_p50=float(np.median(np.abs(data[:, 3]))),
                unloaded=float(((data[:, 0] >= 0.001) & (data[:, 1] < 0.5)).mean()),
                angle_pass=float((np.abs(data[:, 2]) < 0.1).mean()),
                speed_pass=float((np.abs(data[:, 3]) < 0.25).mean()),
                joint_command_p50=np.median(
                    action[mask][:, row * 3 : row * 3 + 3]
                    * np.array(cfg.action_scale)[row * 3 : row * 3 + 3],
                    axis=0,
                ).tolist(),
            )
        result[leg] = stages
        failures = []
        for episode in range(args.count):
            ticks = np.flatnonzero(selected[:, episode])
            if not len(ticks) or completed[-1, episode, row]:
                continue
            phases = phase[ticks, episode, row]
            last = probe[ticks[-1], episode, row]
            failures.append(
                dict(
                    episode=episode,
                    reached_rotate=bool(np.any(phases == 2)),
                    lost_unloading=int(np.sum((phases[:-1] == 2) & (phases[1:] == 1))),
                    final_phase=int(phases[-1]),
                    final_angle_error=float(abs(last[2])),
                    final_speed=float(abs(last[3])),
                    final_clearance=float(last[0]),
                )
            )
        result[leg]["failures"] = failures
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
