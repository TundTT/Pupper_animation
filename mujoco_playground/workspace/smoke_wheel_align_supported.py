"""Closed-loop full-sequence smoke of the simplified supported-lift controller."""

import json
from pathlib import Path
import jax
import numpy as np
from jax import numpy as jp
from workspace import configs
from workspace.wheel_align_env import PupperWheelAlignEnv


def main():
    cfg = configs.get_wheel_align_config()
    cfg.align.initial_completed_max = 0
    cfg.dr.kick_probability = 0.0
    cfg.align.wheel_disturbance_probability = 0.0
    env = PupperWheelAlignEnv(cfg)

    @jax.jit
    def rollout():
        s = env.reset(jax.random.PRNGKey(2))

        def advance(s, _):
            s = env.step(s, jp.zeros(3))
            return s, (
                s.info["command"],
                s.info["phase"],
                env._wheel_clearance(s.pipeline_state),
                env._wheel_loads(s.pipeline_state),
                s.info["lift_reference"],
                s.metrics["body_drift_dist"],
                s.done,
                env._errors(s.pipeline_state, s.info),
                s.pipeline_state.qd[env._wheel_dof],
            )

        return jax.lax.scan(advance, s, None, length=cfg.episode_length)

    state, trace = jax.device_get(rollout())
    cmd, phase, z, load, lift, drift, done, error, speed = trace
    np.savez_compressed(
        "workspace/output/align_diagnosis/supported_trace.npz",
        command=cmd,
        phase=phase,
        clearance=z,
        load=load,
        lift_reference=lift,
        drift=drift,
        done=done,
        error=error,
        speed=speed,
    )
    for leg, row in configs.FOOT_ROW_BY_LEG.items():
        selected = cmd == configs.COMMAND_STATES.index(leg)
        print(
            leg,
            "phases",
            np.unique(phase[selected, row], return_counts=True),
            "maxclear",
            z[selected, row].max(),
            "maxref",
            lift[selected, row].max(),
            "minerr",
            np.abs(error[selected, row]).min(),
            flush=True,
        )
    print(
        "completed",
        state.info["completed"],
        "drift",
        np.percentile(drift, [50, 95, 100]),
        "falls",
        done.sum(),
        flush=True,
    )
    assert state.obs.shape == (376,) and env.action_size == 3
    assert all(
        np.isfinite(x).all()
        for x in jax.tree.leaves((state.obs, state.reward, state.metrics))
    )
    assert not np.any(done), "nominal supported sequence fell"
    assert np.all(state.info["completed"]), (
        "nominal sequence did not complete all four legs"
    )
    assert np.max(np.abs(error[-100:])) < 0.1
    print(
        "PASS: zero-residual nominal full four-leg lift/rotate/lower/hold sequence",
        flush=True,
    )


if __name__ == "__main__":
    main()
