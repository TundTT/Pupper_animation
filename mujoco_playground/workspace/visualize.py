"""Render a leg-lift rollout to video.

Rolls the policy out while stepping the command through the on-robot O-button
sequence (stand -> front_l -> front_r -> back_r -> back_l -> stand), so the video
shows each leg being raised, held, and lowered in turn. Mirrors pupperv3-mjx's
`utils.visualize_policy`: collect brax `pipeline_state`s and render with the env's
`tracking_cam`. The result is what you watch to judge whether the policy looks sane.
"""

from typing import Callable, List, Tuple

import jax
import numpy as np
from jax import numpy as jp

from workspace import configs

# Order the O button cycles through (indices into configs.COMMAND_STATES).
_SHOWCASE_ORDER = [0, 1, 2, 3, 4, 0]  # stand, FL, FR, BR, BL, stand


def showcase_schedule(steps_per_command: int) -> List[int]:
    """A per-step list of command indices walking the O-button sequence."""
    schedule: List[int] = []
    for cmd in _SHOWCASE_ORDER:
        schedule.extend([cmd] * steps_per_command)
    return schedule


def rollout(eval_env, inference_fn: Callable, rng: jax.Array, schedule: List[int]) -> List:
    """Roll the policy out following `schedule`, returning the pipeline-state trajectory."""
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference = jax.jit(inference_fn)

    state = jit_reset(rng)
    traj = [state.pipeline_state]
    for cmd in schedule:
        # Force the command and disable the env's internal random switching so the
        # rollout follows the showcase sequence exactly (mimics button presses).
        state.info["command"] = jp.int32(cmd)
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
