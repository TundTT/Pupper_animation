"""GPU evaluator/render smoke, separate from the CPU physics regression suite."""

import jax
import mediapy as media
import numpy as np
from jax import numpy as jp

from workspace import configs
from workspace.wheel_align_env import PupperWheelAlignEnv, domain_randomize_wheel_align
from workspace.wheel_align_evaluate import make_evaluator


def main():
    cfg = configs.get_wheel_align_config()
    cfg.episode_length = 8

    def zero_policy(params, deterministic):
        del params, deterministic
        return lambda obs, rng: (jp.zeros(obs.shape[:-1] + (3,)), {})

    evaluate = make_evaluator(cfg, zero_policy, domain_randomize_wheel_align, count=2)
    metrics = evaluate(None)
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["audit/mean_completed_legs"] == 0
    assert metrics["audit/all_four_success_fraction"] == 0
    assert metrics["audit/held_samples"] == 0
    print(
        f"PASS: full-sequence evaluator, no partial-completion/empty-hold success: {metrics}",
        flush=True,
    )
    env = PupperWheelAlignEnv(cfg)
    state = jax.jit(env.reset)(jax.random.PRNGKey(44))
    trajectory = [state.pipeline_state]
    step = jax.jit(env.step)
    for _ in range(5):
        state = step(state, jp.zeros(3))
        assert np.isfinite(state.reward) and state.obs.shape == (376,)
        trajectory.append(state.pipeline_state)
    media.write_video(
        "smoke_wheel_align_phase.mp4", np.array(env.render(trajectory)), fps=50
    )
    print(f"PASS: GPU rollout + EGL video on {jax.devices()}", flush=True)


if __name__ == "__main__":
    main()
