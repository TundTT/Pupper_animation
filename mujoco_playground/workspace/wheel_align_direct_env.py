"""Direct 12-position-output alignment; no runtime phase/clearance controller.

The parent's phase machine is only a reward/evaluation label. It never gates an
action. Actuation is exactly default + scale * action, held for one control step
by the ordinary position actuators. Wheel defaults are calibrated-target angular
representatives latched once at mode entry, not recomputed by a runtime servo.
"""

from pathlib import Path

import jax
import numpy as np
from brax import math
from brax.envs.base import PipelineEnv
from jax import numpy as jp

from workspace import configs
from workspace.wheel_align_env import (
    HOLD,
    IDLE,
    LIFT,
    LOWER,
    ROTATE,
    PupperWheelAlignEnv,
    wrap,
)
from workspace.wheel_env import _sample_lagged

DIRECT_FRAME_SIZE = 56


class PupperWheelAlignDirectEnv(PupperWheelAlignEnv):
    def __init__(self, config, model_path=None):
        super().__init__(config, model_path)
        self._single_obs_dim = int(config.direct.observation_frame_size)
        if self._single_obs_dim not in (55, 56, 64):
            raise ValueError("Expected 55, 56, or 64 sensor features")
        # Legacy checkpoints used four adjacent frames. Sparse taps change only
        # which real sensor frames are retained, never the action mapping.
        self._history_taps = (
            tuple(config.direct.history_taps)
            if "history_taps" in config.direct
            else tuple(range(self._obs_history))
        )
        if (
            len(self._history_taps) != self._obs_history
            or self._history_taps[0] != 0
            or tuple(sorted(set(self._history_taps))) != self._history_taps
        ):
            raise ValueError(
                "History taps must be increasing, start at zero, and match observation_history"
            )
        self._sparse_history = self._history_taps != tuple(range(self._obs_history))
        self._lift_clearance_upper = (
            float(config.direct.lift_clearance_upper)
            if "lift_clearance_upper" in config.direct
            else None
        )
        if (
            self._lift_clearance_upper is not None
            and self._lift_clearance_upper <= config.align.clearance_min
        ):
            raise ValueError(
                "Lift reward band must extend above the unloading threshold"
            )
        self._ctrl_sign = jp.ones(12)  # all outputs use native position coordinates
        path = config.direct.reference_path
        if path:
            with np.load(Path(path)) as reference:
                self._reference_poses = jp.array(reference["poses"])
                self._reference_valid = jp.array(reference["valid"])
        else:
            if config.reward_config.scales.imitation != 0:
                raise ValueError(
                    "Nonzero imitation reward requires a recorded reference bank"
                )
            self._reference_poses = jp.zeros((5, 5, 32, 8))
            self._reference_valid = jp.zeros((5, 5, 32), dtype=bool)

    @property
    def action_size(self):
        return 12

    def pipeline_step(self, pipeline_state, targets):
        # Bypass the teacher's per-physics-step wrapped wheel servo.
        return PipelineEnv.pipeline_step(self, pipeline_state, targets)

    def _leg_action(self, state, action):
        del state
        return action  # no selected-hip clamp, reference slew, or phase gating

    def position_targets(self, info, action):
        default = self._default_pose.at[self._wheel_rows].set(info["wheel_zero"])
        return jp.clip(
            default + self._action_scale * action, self._lowers, self._uppers
        )

    def _advance_physics(self, state, action):
        info = state.info
        info["command_age"] += 1
        info["rng"], kick_rng, lag_rng, obs_rng = jax.random.split(info["rng"], 4)
        lagged, info["action_buffer"] = _sample_lagged(
            lag_rng, info["action_buffer"], action, self._lat_dist
        )
        key1, key2 = jax.random.split(kick_rng)
        kick = (
            jax.random.uniform(key1, (2,), minval=-1.0, maxval=1.0)
            * self._config.dr.kick_vel
        )
        kick *= jax.random.bernoulli(key2, self._config.dr.kick_probability)
        ps = state.pipeline_state.replace(qd=state.pipeline_state.qd.at[:2].add(kick))
        info["last_applied_action"] = lagged
        return self.pipeline_step(ps, self.position_targets(info, lagged)), obs_rng

    def _select(self, info, command):
        phase = info["phase"]
        info["command_age"] = jp.where(
            command != info["command"], 0, info["command_age"]
        )
        super()._select(info, command)
        info["phase_age"] = jp.where(info["phase"] != phase, 0, info["phase_age"])

    def set_command(self, state, command):
        # Refresh the current command frame WITHOUT adding a phantom time sample.
        # This matches the robot: one history shift per inference tick.
        info = dict(state.info)
        self._select(info, jp.asarray(command, dtype=jp.int32))
        info["rng"], key = jax.random.split(info["rng"])
        if self._sparse_history:
            info["sensor_history"] = jp.roll(info["sensor_history"], -1, axis=0)
        obs = self._get_obs(
            state.pipeline_state, info, jp.roll(state.obs, -self._single_obs_dim), key
        )
        return state.replace(info=info, obs=obs)

    def reset(self, rng):
        # Parent supplies full sequences, wheel-radius-aware spawn and calibration.
        # _get_obs initializes the direct-action frame before the first observation.
        state = super().reset(rng)
        state.metrics["new_settles"] = jp.float32(0.0)
        return state

    def _initialize_direct_info(self, ps, info):
        info["command_age"] = jp.int32(0)
        if self._sparse_history:
            info["sensor_history"] = jp.zeros(
                (max(self._history_taps) + 1, self._single_obs_dim)
            )
        current = ps.q[self._wheel_qpos]
        info["wheel_zero"] = current + wrap(info["target"] - current)
        info["initial_completed"] = info["completed"]
        info["settle_paid"] = info["completed"]
        initial = (
            jp.zeros(12)
            .at[self._wheel_rows]
            .set((current - info["wheel_zero"]) / self._action_scale[self._wheel_rows])
        )
        info["last_act"] = initial
        info["last_policy_action"] = initial
        info["last_applied_action"] = initial
        info["action_buffer"] = jp.broadcast_to(
            initial[:, None], (12, self._lat_buf_len)
        )

    def sensor_observation(self, ps, info, history, rng, reset_history=False):
        key1, key2 = jax.random.split(rng)
        ang, gravity, mixed, _ = self._proprioception(ps, info, key1)
        local_wheel = ps.q[self._wheel_qpos] - info["wheel_zero"]
        local_wheel += (
            jax.random.uniform(key2, (4,), minval=-1.0, maxval=1.0)
            * self._config.align.encoder_angle_noise
        )
        joints = mixed.at[self._wheel_rows].set(local_wheel)
        error = wrap(-local_wheel)
        # 35 legacy leg-lift entries, then encoder-derived wheel speed/error and
        # a four-bit "ever selected" latch obtainable solely from button commands.
        frame = jp.concatenate(
            (
                ang,
                gravity,
                jax.nn.one_hot(info["command"], configs.NUM_COMMANDS),
                joints,
                info["last_policy_action"],
                mixed[self._wheel_rows],
                jp.sin(error),
                jp.cos(error),
                info["tracked"].astype(jp.float32),
                # A fixed encoder anchor prevents an unselected wheel from learning
                # an accumulating "copy current/last output" approximation.
                (info["wheel_idle_target"] - info["wheel_zero"])
                / self._action_scale[self._wheel_rows],
            )
        )
        if self._single_obs_dim >= 56:
            # Observable button-event age, capped after one second. This cannot
            # time-index a whole rotation; the angle-error feedback is retained.
            frame = jp.concatenate(
                (frame, jp.atleast_1d(jp.minimum(info["command_age"] * self.dt, 1.0)))
            )
        if self._single_obs_dim == 64:
            velocity = ps.qd[self._joint_dof[self._pos_rows]]
            velocity += (
                jax.random.uniform(
                    jax.random.fold_in(key2, 1), (8,), minval=-1.0, maxval=1.0
                )
                * self._config.direct.position_velocity_noise
            )
            frame = jp.concatenate(
                (frame, velocity / self._config.direct.position_velocity_scale)
            )
        frame = jp.clip(frame, -100.0, 100.0)
        if self._sparse_history:
            info["sensor_history"] = (
                jp.tile(frame, (max(self._history_taps) + 1, 1))
                if reset_history
                else jp.roll(info["sensor_history"], 1, axis=0).at[0].set(frame)
            )
            return info["sensor_history"][jp.array(self._history_taps)].reshape(-1)
        if reset_history:
            return jp.tile(frame, self._obs_history)
        return (
            jp.roll(history, self._single_obs_dim).at[: self._single_obs_dim].set(frame)
        )

    def _get_obs(self, ps, info, history, rng):
        # Explicit adaptation of the inherited reset schema, once per episode.
        initializing = "wheel_zero" not in info
        if initializing:
            self._initialize_direct_info(ps, info)
        return self.sensor_observation(ps, info, history, rng, initializing)

    def _lift_cost(self, clearance):
        cfg = self._config.align
        if self._lift_clearance_upper is None:
            # Preserve the exact reward of existing checkpoint configurations.
            return jp.minimum(
                jp.square((clearance - cfg.clearance_target) / cfg.clearance_target),
                4.0,
            )
        deficit = jp.maximum(cfg.clearance_min - clearance, 0.0) / cfg.clearance_min
        excess = (
            jp.maximum(clearance - self._lift_clearance_upper, 0.0)
            / cfg.clearance_target
        )
        return jp.minimum(jp.square(deficit) + jp.square(excess), 4.0)

    def _alignment_reward(self, ps, info, action, clearance, load, error, speed, phase):
        del action
        cfg = self._config.align
        selected = (jp.arange(4) == self._command_leg[info["command"]]) & (
            info["command"] != 0
        )
        airborne = (phase == LIFT) | (phase == ROTATE)
        contact = (clearance < cfg.contact_tolerance) | (load >= cfg.unloaded_force)
        reweight = jp.clip(load / cfg.reweight_force, 0.0, 1.0)
        torso = self._torso_idx - 1
        drift = jp.linalg.norm(ps.x.pos[torso, :2] - info["init_xy"])
        returning = (phase == LOWER) & (
            info["phase_age"] < self._config.direct.return_steps
        )
        allowance = jp.where(
            jp.any(airborne | returning), cfg.support_body_drift, cfg.allowed_body_drift
        )
        tilt = math.rotate(jp.array([0.0, 0.0, 1.0]), ps.x.rot[torso])[2]
        held = info["rotation_verified"]
        fresh_settle = held & ~info["settle_paid"] & ~info["initial_completed"]
        info["settle_paid"] |= held
        owner = jp.maximum(self._command_leg[info["command"]], 0)
        reference_phase = jp.where(info["command"] != 0, phase[owner], IDLE)
        # The LIFT bank includes preparation at home. Nearest-pose matching to
        # that bank rewards stalling. Guide lifting toward the demonstrated
        # unloaded support poses, without a time-indexed joint trajectory.
        reference_phase = jp.where(reference_phase == LIFT, ROTATE, reference_phase)
        poses = self._reference_poses[info["command"], reference_phase]
        valid = self._reference_valid[info["command"], reference_phase]
        position = (
            ps.q[self._joint_qpos[self._pos_rows]] - self._default_pose[self._pos_rows]
        )
        distances = jp.mean(jp.square((poses - position) / 0.15), axis=-1)
        imitation = jp.where(
            jp.any(valid), jp.min(jp.where(valid, distances, 1e6)), 0.0
        )
        rewards = dict(
            alive=jp.float32(1.0),
            lift=jp.sum(airborne * self._lift_cost(clearance)),
            unload=jp.sum((phase == LIFT) * load) / jp.maximum(jp.sum(load) / 4.0, 1.0),
            rotate=jp.sum((phase == ROTATE) * (1.0 - jp.cos(error))),
            lower=jp.sum(
                (phase == LOWER)
                * (
                    jp.minimum(
                        jp.square(jp.maximum(clearance, 0.0) / cfg.clearance_target),
                        4.0,
                    )
                    + 1.0
                    - reweight
                )
            ),
            hold=jp.sum(
                held
                * (
                    jp.minimum(jp.square(error / 0.1), 4.0)
                    + jp.minimum(jp.square(speed / 0.4), 4.0)
                )
            ),
            idle_wheels=jp.sum(
                ~info["tracked"]
                * jp.minimum(
                    jp.square(
                        wrap(info["wheel_idle_target"] - ps.q[self._wheel_qpos]) / 0.1
                    ),
                    4.0,
                )
            ),
            active_scrub=jp.sum(selected * contact * jp.square(speed)),
            inactive_motion=jp.sum(~selected * jp.square(speed)),
            support=jp.sum(~airborne * (1.0 - reweight)),
            imitation=jp.minimum(imitation, 10.0),
            orientation=(1.0 - tilt) / 0.05,
            body_drift=jp.square(jp.maximum(drift - allowance, 0.0) / cfg.drift_scale),
            action_rate=jp.float32(0.0),
            ground_contact=jp.sum(
                jp.clip(
                    (
                        0.015
                        - (ps.x.pos[self._wheel_body_id - 1, 2] - configs.KNEE_RADIUS)
                    )
                    / 0.015,
                    0.0,
                    1.0,
                )
            ),
            # Parent integrates running rewards by dt; these are discrete events.
            settle=jp.sum(fresh_settle.astype(jp.float32)) / self.dt,
        )
        diagnostics = dict(
            body_drift_dist=drift,
            drift_outside_tolerance=jp.float32(drift > cfg.allowed_body_drift),
            active_clearance=jp.sum(selected * clearance),
            active_angle_error=jp.sum(selected * jp.abs(error)),
            active_scrub_speed=jp.sum(selected * contact * jp.abs(speed)),
            hold_angle_error=jp.sum(held * jp.abs(error)) / jp.maximum(jp.sum(held), 1),
            held_angle_error=jp.sum((phase == HOLD) * jp.abs(error))
            / jp.maximum(jp.sum(phase == HOLD), 1),
            lower_angle_error=jp.sum((phase == LOWER) * jp.abs(error))
            / jp.maximum(jp.sum(phase == LOWER), 1),
            held_error_sum=jp.sum((phase == HOLD) * jp.abs(error)),
            held_samples=jp.sum(phase == HOLD).astype(jp.float32),
            completed_legs=jp.sum(info["completed"]).astype(jp.float32),
            tracked_legs=jp.sum(info["tracked"]).astype(jp.float32),
            wheel_contacts=jp.sum(contact).astype(jp.float32),
            active_phase=jp.sum(selected * info["phase"]).astype(jp.float32),
            new_settles=jp.sum(fresh_settle).astype(jp.float32),
        )
        return rewards, diagnostics
