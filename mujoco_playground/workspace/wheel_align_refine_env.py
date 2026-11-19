"""Direct learned joint control with an encoder-only settled-wheel latch.

Optional demonstration resets are training-only. Neither the latch nor the
actor reads simulation phase, contact force, clearance, or body displacement.
"""

from pathlib import Path

import jax
import numpy as np
from jax import numpy as jp

from workspace.wheel_align_direct_env import PupperWheelAlignDirectEnv
from workspace.wheel_align_env import wrap


class PupperWheelAlignRefineEnv(PupperWheelAlignDirectEnv):
    def __init__(self, config, model_path=None):
        super().__init__(config, model_path)
        self._reset_probability = float(config.direct.reference_reset_probability)
        if not 0 <= self._reset_probability <= 1:
            raise ValueError("Reference reset probability must be in [0,1]")
        self._reset_bank = None
        if self._reset_probability:
            with np.load(Path(config.direct.reference_reset_path)) as data:
                self._reset_bank = {k: jp.array(data[k]) for k in data.files}
            if self._reset_bank["sensor_history"].shape[1:] != (
                51,
                self._single_obs_dim,
            ):
                raise ValueError("Reference history does not match the sensor contract")

    def _initialize_direct_info(self, ps, info):
        super()._initialize_direct_info(ps, info)
        info["encoder_held"] = jp.zeros(4, dtype=bool)
        info["encoder_settle_ticks"] = jp.zeros(4, dtype=jp.int32)
        info["reset_horizon"] = jp.int32(self._config.episode_length)

    def wheel_hold_action(self, ps, info, action):
        current = ps.q[self._wheel_qpos]
        delta = wrap(info["wheel_zero"] - current)
        selected = (jp.arange(4) == self._command_leg[info["command"]]) & (
            info["command"] != 0
        )
        settled = (
            selected & (jp.abs(delta) < 0.08) & (jp.abs(ps.qd[self._wheel_dof]) < 0.25)
        )
        info["encoder_settle_ticks"] = jp.where(
            settled, info["encoder_settle_ticks"] + 1, 0
        )
        info["encoder_held"] |= info["encoder_settle_ticks"] >= 5
        # Native motor position PD provides the restoring torque. Nearest angular
        # representative remains valid after arbitrary whole encoder revolutions.
        normalized = (current + delta - info["wheel_zero"]) / self._action_scale[
            self._wheel_rows
        ]
        return action.at[self._wheel_rows].set(
            jp.where(info["encoder_held"], normalized, action[self._wheel_rows])
        )

    def _advance_physics(self, state, action):
        action = self.wheel_hold_action(state.pipeline_state, state.info, action)
        return super()._advance_physics(state, action)

    def reset(self, key):
        state = super().reset(key)
        if self._reset_bank is None:
            return state
        choose, sample = jax.random.split(jax.random.fold_in(key, 313))
        idx = jax.random.randint(sample, (), 0, len(self._reset_bank["q"]))

        def reference_reset(state):
            row = {k: v[idx] for k, v in self._reset_bank.items()}
            info = dict(state.info)
            for name in row:
                if name not in ("q", "qd", "sensor_history"):
                    info[name] = row[name]
            info["reset_horizon"] = jp.int32(self._config.episode_length) - row["step"]
            info["command_switch_step"] -= row["step"]
            info["step"] = jp.int32(0)
            info["dwell"] = jp.zeros(4, dtype=jp.int32)
            info["initial_completed"] = info["completed"]
            info["settle_paid"] = info["rotation_verified"]
            info["last_act"] = info["last_policy_action"]
            info["last_applied_action"] = info["last_policy_action"]
            info["last_vel"] = row["qd"][self._joint_dof]
            info["action_buffer"] = jp.broadcast_to(
                info["last_policy_action"][:, None], (12, self._lat_buf_len)
            )
            if self._sparse_history:
                info["sensor_history"] = row["sensor_history"][
                    : max(self._history_taps) + 1
                ]
            ps = self.pipeline_init(
                row["q"],
                row["qd"],
                ctrl=self.position_targets(info, info["last_policy_action"]),
            )
            # Replay only encoder/IMU history actually observed in the teacher.
            # No phase or contact feature is introduced into the actor.
            obs = row["sensor_history"][jp.array(self._history_taps)].reshape(-1)
            return state.replace(pipeline_state=ps, info=info, obs=obs)

        return jax.lax.cond(
            jax.random.uniform(choose) < self._reset_probability,
            reference_reset,
            lambda x: x,
            state,
        )

    def step(self, state, action):
        state = super().step(state, action)
        if self._reset_bank is None:
            return state
        return state.replace(
            done=jp.maximum(
                state.done,
                (state.info["step"] >= state.info["reset_horizon"]).astype(jp.float32),
            )
        )


def refinement_config(config):
    """Explicit public defaults for evaluation and encoder-hold-only runs."""
    config.direct.encoder_hold = True
    config.direct.control_convention = "default_plus_scaled_position_with_encoder_hold"
    if "reference_reset_probability" not in config.direct:
        config.direct.reference_reset_probability = 0.0
        config.direct.reference_reset_path = ""
    return config
