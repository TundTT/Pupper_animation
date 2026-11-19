"""Fixed full-sequence showcase through the same rendering/W&B path as driving."""

import jax
import numpy as np
from jax import numpy as jp


def render(eval_env, inference_fn, rng, render_every=3):
    # The video env starts with no completed legs and a fixed seed, then follows
    # the external clockwise button sequence. No state resets between legs.
    reset = jax.jit(eval_env.reset)
    step = jax.jit(eval_env.step)
    select = jax.jit(eval_env.set_command)
    inference = jax.jit(inference_fn)
    state = reset(rng)
    trajectory = [state.pipeline_state]
    drift, scrub, hold_error = [], [], []
    for command, duration in [
        (0, 50),
        (1, 275),
        (0, 50),
        (2, 275),
        (0, 50),
        (3, 275),
        (0, 50),
        (4, 275),
        (0, 300),
    ]:
        state = select(state, jp.int32(command))
        for i in range(duration):
            rng, key = jax.random.split(rng)
            action, _ = inference(state.obs, key)
            state = step(state, action)
            if i % render_every == 0:
                trajectory.append(state.pipeline_state)
                drift.append(float(state.metrics["body_drift_dist"]))
                scrub.append(float(state.metrics["active_scrub_speed"]))
                hold_error.append(float(state.metrics["hold_angle_error"]))
            if bool(state.done):
                print(
                    "  Showcase terminated early (fall/nonfinite state); video shows the failure.",
                    flush=True,
                )
                frames = eval_env.render(trajectory)
                return np.array(frames), round(1.0 / eval_env.dt / render_every)
    print(
        f"  Showcase: completed={int(jp.sum(state.info['completed']))}/4 "
        f"drift_p95={np.percentile(drift, 95):.4f}m drift_max={max(drift):.4f}m "
        f"hold_error_p95={np.percentile(hold_error, 95):.3f}rad "
        f"active_scrub_mean={np.mean(scrub):.3f}rad/s",
        flush=True,
    )
    return np.array(eval_env.render(trajectory)), round(
        1.0 / eval_env.dt / render_every
    )
