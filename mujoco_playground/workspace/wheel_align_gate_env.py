"""Eight learned joint targets plus four learned wheel-enable outputs.

Only the externally selected wheel can advance. Its reference uses encoders,
the calibrated target and the network's enable output. No contact estimator.
"""

import jax
from jax import numpy as jp

from workspace.wheel_align_env import wrap
from workspace.wheel_align_refine_env import PupperWheelAlignRefineEnv


class PupperWheelAlignGateEnv(PupperWheelAlignRefineEnv):
    def __init__(self, config, model_path=None):
        super().__init__(config, model_path)
        self._capture_goal = config.direct.get("capture_finished_reference", False)

    def _initialize_direct_info(self, ps, info):
        super()._initialize_direct_info(ps, info)
        info["issued_action"] = info["last_policy_action"]
        info["last_policy_action"] = (
            info["last_policy_action"].at[self._wheel_rows].set(-0.8)
        )
        info["spin_reference"] = ps.q[self._wheel_qpos]
        info["enable_ticks"] = jp.zeros(4, dtype=jp.int32)
        # Partial training resets represent goals commanded earlier in this
        # sequence. Normal mode entry has no such prior command history.
        info["goal_commanded"] = info["tracked"] & self._capture_goal

    def _get_obs(self, ps, info, history, rng):
        initializing = "wheel_zero" not in info
        if initializing:
            self._initialize_direct_info(ps, info)
        raw = info["last_policy_action"]
        # Keep the teacher-compatible previous physical wheel-command features.
        # The eight ordinary joints retain their previous raw network outputs.
        info["last_policy_action"] = raw.at[self._wheel_rows].set(
            info["issued_action"][self._wheel_rows]
        )
        obs = self.sensor_observation(ps, info, history, rng, initializing)
        info["last_policy_action"] = raw
        return obs

    def wheel_hold_action(self, ps, info, action):
        super().wheel_hold_action(ps, info, action)  # encoder-only settled latch
        selected = (jp.arange(4) == self._command_leg[info["command"]]) & (
            info["command"] != 0
        )
        enable = selected & (action[self._wheel_rows] > 0.0)
        info["enable_ticks"] = jp.where(
            enable, jp.minimum(info["enable_ticks"] + 1, 3), 0
        )
        turning = info["enable_ticks"] >= 3
        current = ps.q[self._wheel_qpos]
        held = info["encoder_held"] | info["goal_commanded"]
        reference = jp.where(
            selected & ~turning & ~held, current, info["spin_reference"]
        )
        delta = jp.clip(
            wrap(info["wheel_zero"] - reference), -3.0 * self.dt, 3.0 * self.dt
        )
        reference += jp.where(turning, delta, 0.0)
        info["goal_commanded"] |= (
            self._capture_goal
            & turning
            & (jp.abs(wrap(info["wheel_zero"] - reference)) < 1e-4)
        )
        held |= info["goal_commanded"]
        reference = jp.where(held, info["wheel_zero"], reference)
        info["spin_reference"] = reference
        target = current + wrap(reference - current)
        issued = action.at[self._wheel_rows].set(
            (target - info["wheel_zero"]) / self._action_scale[self._wheel_rows]
        )
        info["issued_action"] = issued
        return issued

    def reset(self, key):
        state = super().reset(key)
        if self._reset_bank is not None:
            reference = state.info["reset_horizon"] < self._config.episode_length
            physical = state.info["last_policy_action"]
            state.info["issued_action"] = jp.where(
                reference, physical, state.info["issued_action"]
            )
            state.info["spin_reference"] = state.pipeline_state.q[self._wheel_qpos]
            state.info["goal_commanded"] = jp.where(
                reference,
                state.info["initial_completed"] & self._capture_goal,
                state.info["goal_commanded"],
            )
            state.info["last_policy_action"] = physical.at[self._wheel_rows].set(-0.8)
        return state
