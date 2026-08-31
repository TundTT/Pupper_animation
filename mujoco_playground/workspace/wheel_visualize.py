"""Render a wheeled-locomotion rollout to video.

Rolls the policy out through a fixed sequence of velocity commands (stop, drive
forward, turn each way, reverse) so every eval video shows the same manoeuvres
and runs are comparable frame-for-frame. Mirrors `visualize.py`'s structure --
collect brax `pipeline_state`s, render with the env's `tracking_cam` -- but the
command here is a continuous (vx, vy, yaw) vector rather than the leg-lift
command index, so this needs its own schedule.
"""

from typing import Callable, List, Tuple

import jax
import numpy as np
from jax import numpy as jp

# (vx m/s, vy m/s, yaw rad/s, label). Kept inside
# configs.lin_vel_x_range / ang_vel_yaw_range so the showcase only ever asks for
# things the policy was actually trained on.
_SHOWCASE: List[Tuple[float, float, float, str]] = [
    (0.0, 0.0, 0.0, "stop"),
    (0.5, 0.0, 0.0, "forward"),
    (0.8, 0.0, 0.0, "forward fast"),
    (0.4, 0.0, 1.0, "arc left"),
    (0.4, 0.0, -1.0, "arc right"),
    (0.0, 0.0, 1.5, "spin in place"),
    (-0.4, 0.0, 0.0, "reverse"),
    (0.0, 0.0, 0.0, "stop"),
]


def showcase_schedule(steps_per_command: int) -> List[jax.Array]:
    """A per-step list of (vx, vy, yaw) command vectors walking the showcase."""
    schedule: List[jax.Array] = []
    for vx, vy, wz, _ in _SHOWCASE:
        schedule.extend([jp.array([vx, vy, wz])] * steps_per_command)
    return schedule


def rollout(eval_env, inference_fn: Callable, rng: jax.Array, schedule: List[jax.Array]) -> List:
    """Roll the policy out following `schedule`, returning the pipeline-state trajectory."""
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference = jax.jit(inference_fn)

    state = jit_reset(rng)
    traj = [state.pipeline_state]
    for cmd in schedule:
        # Force the command and disable the env's internal resampling so the
        # rollout follows the showcase exactly.
        state.info["command"] = cmd
        state.info["command_switch_step"] = jp.int32(2_000_000_000)
        rng, act_rng = jax.random.split(rng)
        action, _ = jit_inference(state.obs, act_rng)
        state = jit_step(state, action)
        traj.append(state.pipeline_state)
    return traj


def render(
    eval_env,
    inference_fn: Callable,
    rng: jax.Array,
    steps_per_command: int = 100,
    render_every: int = 2,
) -> Tuple[np.ndarray, int]:
    """Roll out and render the showcase. Returns (frames, fps)."""
    schedule = showcase_schedule(steps_per_command)
    traj = rollout(eval_env, inference_fn, rng, schedule)
    frames = eval_env.render(traj[::render_every], camera="tracking_cam")
    fps = max(int(1.0 / eval_env.dt / render_every), 1)
    return np.array(frames), fps
