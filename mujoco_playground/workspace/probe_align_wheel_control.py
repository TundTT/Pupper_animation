"""Paired wheel-only interventions; oracle mode is diagnostic, never deployable."""

import argparse
import json
from pathlib import Path

import jax
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks
from jax import numpy as jp
from ml_collections import ConfigDict

from workspace.wheel_align_direct_env import PupperWheelAlignDirectEnv
from workspace.wheel_align_env import ROTATE, domain_randomize_wheel_align, wrap
from workspace.wheel_align_evaluate import make_evaluator


class WheelProbeEnv(PupperWheelAlignDirectEnv):
    def _initialize_direct_info(self, ps, info):
        super()._initialize_direct_info(ps, info)
        info["probe_wheel_reference"] = ps.q[self._wheel_qpos]
        info["probe_wheel_held"] = jp.zeros(4, dtype=bool)
        info["probe_settle_count"] = jp.zeros(4, dtype=jp.int32)

    def _advance_physics(self, state, action):
        mode = self._config.direct.wheel_probe_mode
        if mode == "baseline":
            return super()._advance_physics(state, action)
        info, ps = state.info, state.pipeline_state
        current = ps.q[self._wheel_qpos]
        error = wrap(info["wheel_zero"] - current)
        speed = ps.qd[self._wheel_dof]
        selected = (jp.arange(4) == self._command_leg[info["command"]]) & (
            info["command"] != 0
        )
        close = (jp.abs(error) < 0.08) & (jp.abs(speed) < 0.25) & selected
        info["probe_settle_count"] = jp.where(close, info["probe_settle_count"] + 1, 0)
        info["probe_wheel_held"] |= info["probe_settle_count"] >= 5
        held = info["probe_wheel_held"]
        if mode == "snap":
            target = jp.where(
                held,
                current + error,
                self.position_targets(info, action)[self._wheel_rows],
            )
        else:
            if mode == "oracle":
                release = (
                    (info["phase"] == ROTATE)
                    & (self._wheel_clearance(ps) >= 0.001)
                    & (self._wheel_loads(ps) < 0.5)
                )
            elif mode == "hip":
                hip = (
                    ps.q[self._hip_qpos] - self._default_pose[self._hip_rows]
                ) * self._hip_sign
                release = (
                    selected
                    & (hip > self._config.direct.wheel_probe_threshold)
                    & (info["command_age"] * self.dt >= 0.4)
                )
            else:
                raise ValueError(mode)
            reference = info["probe_wheel_reference"]
            reference = jp.where(selected & ~release & ~held, current, reference)
            delta = jp.clip(
                wrap(info["wheel_zero"] - reference), -3.0 * self.dt, 3.0 * self.dt
            )
            reference += jp.where(release | held, delta, 0.0)
            info["probe_wheel_reference"] = reference
            target = current + wrap(reference - current)
        action = action.at[self._wheel_rows].set(
            (target - info["wheel_zero"]) / self._action_scale[self._wheel_rows]
        )
        return super()._advance_physics(state, action)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("params", type=Path)
    p.add_argument("--modes", nargs="+", default=["baseline", "snap", "oracle", "hip"])
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument('--position-noise', type=float,
                   help='diagnostic sensor-noise intervention; physical DR unchanged')
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=901)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cfg = ConfigDict(json.loads((a.params.parent / "config.json").read_text()))
    if a.position_noise is not None:
        cfg.dr.motor_angle_noise = a.position_noise
    params = model.load_params(a.params)
    net = networks.make_ppo_networks(
        int(cfg.direct.observation_frame_size) * cfg.observation_history,
        12,
        policy_hidden_layer_sizes=tuple(cfg.policy.hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.policy.hidden_layer_sizes),
        activation=jax.nn.elu,
        preprocess_observations_fn=running_statistics.normalize,
    )
    report = {
        "params": str(a.params),
        "seed": a.seed,
        "count": a.count,
        "hip_threshold": a.threshold,
        "results": {},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in a.modes:
        cfg.direct.wheel_probe_mode = mode
        cfg.direct.wheel_probe_threshold = a.threshold
        evaluate = make_evaluator(
            cfg,
            networks.make_inference_fn(net),
            domain_randomize_wheel_align,
            count=a.count,
            seed=a.seed,
            env_type=WheelProbeEnv,
        )
        result = evaluate(params)
        report["results"][mode] = result
        a.output.write_text(json.dumps(report, indent=2))
        print(mode, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
