"""QuadMorph wheel-to-leg Phase 1 (still entirely in wheel form).

The button command selects a leg; it never depends on the policy's success.
Only the measured intra-leg phase advances automatically. References are one-time
calibrations, independent of encoder phase after prior driving, NOT current+pi
on every step/button. RL adjusts one small clearance target and two coordinated
stance corrections. A clearance servo lifts only the selected hip; measured
phases rotate its hub, lower it, and hold the persistent wheel goal. All stance
joints stay near the drive home pose through one coupled support adjustment.
This is feedback control with learned corrections, not a timed joint animation.

Training's clock-driven button emulator is disabled by automatic_commands=False.
Call set_command(state, index) before inference for a real external button. A
completed/reselected leg keeps its original target, and stand lowers any active
leg while preserving its target. A failed/interrupted leg is never marked done.
"""

import jax
import numpy as np
import mujoco
from brax import math
from brax.envs.base import State
from jax import numpy as jp
from mujoco.mjx._src import support

from workspace import configs
from workspace.randomize import domain_randomize_wheeled
from workspace.wheel_env import PupperWheelEnv

IDLE, LIFT, ROTATE, LOWER, HOLD = range(5)


def wrap(angle):
    return jp.arctan2(jp.sin(angle), jp.cos(angle))


def domain_randomize_wheel_align(sys, rng, **kwargs):
    """Reuse wheel physics DR, preserving alignment's wheel position springs.

    The drive randomizer intentionally zeroes wheel bias[1]. Repair that ONLY
    here, after randomization, using the randomized kp (never the nominal kp).
    Its independent kp/kd draws on the existing position rows give wheel PD the
    same gain uncertainty without coupling stiffness and damping as velocity kv
    does. Friction, geometry, mass, CoM, inertia and their draws are unchanged.
    """
    randomized, in_axes = domain_randomize_wheeled(sys, rng, **kwargs)
    wheel = np.array(configs.WHEEL_ACTUATOR_ROWS)
    position = configs.POSITION_ACTUATOR_ROWS[0]
    kp_multiplier = (
        randomized.actuator_gainprm[:, position, 0] / sys.actuator_gainprm[position, 0]
    )
    kd_multiplier = (
        randomized.actuator_biasprm[:, position, 2] / sys.actuator_biasprm[position, 2]
    )
    kp = sys.actuator_gainprm[wheel, 0] * kp_multiplier[:, None]
    kd = -sys.actuator_biasprm[wheel, 2] * kd_multiplier[:, None]
    return randomized.replace(
        actuator_gainprm=randomized.actuator_gainprm.at[:, wheel, 0].set(kp),
        actuator_biasprm=randomized.actuator_biasprm.at[:, wheel, 1]
        .set(-kp)
        .at[:, wheel, 2]
        .set(-kd),
    ), in_axes


class PupperWheelAlignEnv(PupperWheelEnv):
    def __init__(self, config, model_path=None):
        super().__init__(config, model_path)
        self._wheel_qpos = self._joint_qpos[self._wheel_rows]
        self._wheel_dof = self._joint_dof[self._wheel_rows]
        legs = list(configs.FOOT_ROW_BY_LEG)
        self._hip_rows = np.array(
            [configs.JOINT_NAMES.index(f"leg_{leg}_2") for leg in legs]
        )
        self._hip_qpos = self._joint_qpos[self._hip_rows]
        self._hip_sign = jp.array([configs.HIP_LIFT_SIGN[leg] for leg in legs])
        # These were velocity bounds on wheel-drive actions. An equivalent
        # position target may exceed them after any number of revolutions.
        self._lowers = self._lowers.at[self._wheel_rows].set(-jp.inf)
        self._uppers = self._uppers.at[self._wheel_rows].set(jp.inf)
        self._command_leg = jp.array(
            [-1] + [configs.FOOT_ROW_BY_LEG[n] for n in configs.COMMAND_STATES[1:]]
        )
        self._leg_command = jp.array(
            [configs.COMMAND_STATES.index(n) for n in configs.FOOT_ROW_BY_LEG]
        )
        self._position_leg = jp.array(
            [
                configs.FOOT_ROW_BY_LEG[configs.JOINT_NAMES[r][4:-2]]
                for r in self._pos_rows
            ]
        )
        # A shared body-shift command adjusts the home stance coherently. The
        # map is measured from named wheel-geom Jacobians at the settled pose;
        # it never gives RL eight independent joint offsets.
        data = mujoco.MjData(self.sys.mj_model)
        data.qpos[:] = np.asarray(self._init_q)
        mujoco.mj_forward(self.sys.mj_model, data)
        body_axes = data.xmat[self._torso_idx].reshape(3, 3)[:, :2]
        shift_map = []
        for leg in legs:
            rows = [configs.JOINT_NAMES.index(f"leg_{leg}_{j}") for j in (1, 2)]
            jac = np.zeros((3, self._nv))
            mujoco.mj_jacGeom(
                self.sys.mj_model,
                data,
                jac,
                None,
                int(self._wheel_geom_id[configs.FOOT_ROW_BY_LEG[leg]]),
            )
            shift_map.append(-np.linalg.pinv(jac[:, self._joint_dof[rows]]) @ body_axes)
        self._shift_map = jp.asarray(np.array(shift_map).reshape(8, 2))
        self._support_shifts = jp.array(
            [config.align.support_shifts[leg] for leg in legs]
        )
        self._single_obs_dim = 94

    @property
    def action_size(self):
        return 3

    def _leg_action(self, state, action):
        """Clearance trim + two small balance corrections around a home stance.

        Only the selected hip receives a lift reference. All other leg changes
        are one coordinated, bounded body shift. Lowering returns the reference
        gradually to home; a previous leg returns before another can rise.
        """
        info, cfg, ps = state.info, self._config.align, state.pipeline_state
        airborne = (info["phase"] == LIFT) | (info["phase"] == ROTATE)
        returning = ~airborne & (info["lift_reference"] > 1e-4)
        selected = self._command_leg[info["command"]]
        selected = jp.where(
            (selected >= 0)
            & (
                airborne[jp.maximum(selected, 0)]
                | (info["phase"][jp.maximum(selected, 0)] == LOWER)
            ),
            selected,
            -1,
        )
        owner = jp.where(jp.any(returning), jp.argmax(returning), selected)
        working = (jp.arange(4) == owner) & airborne
        has_owner = owner >= 0
        shift_goal = jp.where(
            has_owner,
            jp.where(
                airborne[jp.maximum(owner, 0)]
                | (info["lift_reference"][jp.maximum(owner, 0)] > 1e-4),
                self._support_shifts[jp.maximum(owner, 0)],
                jp.zeros(2),
            )
            + action[1:] * cfg.balance_adjust,
            jp.zeros(2),
        )
        info["stance_shift"] += jp.clip(
            shift_goal - info["stance_shift"], -cfg.shift_step, cfg.shift_step
        )
        desired_clearance = cfg.clearance_target + action[0] * cfg.clearance_adjust
        lift_delta = jp.clip(
            cfg.clearance_gain * (desired_clearance - self._wheel_clearance(ps)),
            -cfg.lift_step,
            cfg.lift_step,
        )
        prepared = info["phase_age"] >= cfg.prepare_steps
        raise_or_hold = working & prepared & ~jp.any(returning)
        refs = jp.where(
            raise_or_hold,
            info["lift_reference"] + lift_delta,
            jp.where(
                airborne,
                info["lift_reference"],
                jp.maximum(info["lift_reference"] - cfg.lower_step, 0.0),
            ),
        )
        info["lift_reference"] = jp.clip(refs, 0.0, cfg.max_lift_angle)
        offset = (
            jp.zeros(12).at[self._pos_rows].set(self._shift_map @ info["stance_shift"])
        )
        offset = offset.at[self._hip_rows].add(self._hip_sign * info["lift_reference"])
        # Shared physics expects normalized per-row offsets, not policy outputs.
        return offset / self._action_scale

    @staticmethod
    def _override_actuator_gains(sys, config):
        """Convert this env's private model to wheel position PD; retain torque caps."""
        sys = PupperWheelEnv._override_actuator_gains(sys, config)
        wheel = np.array(configs.WHEEL_ACTUATOR_ROWS)
        kp, kd = config.align.wheel_position_kp, config.align.wheel_position_kd
        limited = sys.actuator_ctrllimited.copy()
        limited[wheel] = False
        sys = sys.replace(
            actuator_gainprm=sys.actuator_gainprm.at[wheel, 0].set(kp),
            actuator_biasprm=sys.actuator_biasprm.at[wheel, 1]
            .set(-kp)
            .at[wheel, 2]
            .set(-kd),
            actuator_ctrllimited=limited,
            actuator_ctrlrange=sys.actuator_ctrlrange.at[wheel].set(
                jp.array([-jp.inf, jp.inf])
            ),
        )
        # Keep Brax's legacy actuator metadata consistent too, although MJX uses
        # the arrays above. Only this freshly loaded MjModel is changed.
        sys = sys.replace(
            actuator=sys.actuator.replace(
                gain=sys.actuator.gain.at[wheel].set(kp),
                bias_q=sys.actuator.bias_q.at[wheel].set(-kp),
                bias_qd=sys.actuator.bias_qd.at[wheel].set(-kd),
                ctrl_range=sys.actuator.ctrl_range.at[wheel].set(
                    jp.array([-jp.inf, jp.inf])
                ),
            )
        )
        sys.mj_model.actuator_gainprm[wheel, 0] = kp
        sys.mj_model.actuator_biasprm[wheel, 1] = -kp
        sys.mj_model.actuator_biasprm[wheel, 2] = -kd
        sys.mj_model.actuator_ctrllimited[wheel] = False
        sys.mj_model.actuator_ctrlrange[wheel] = [-np.inf, np.inf]
        return sys

    def _position_ctrl(self, ps, goals):
        """Nearest equivalent unwrapped ctrl, evaluated AFTER command latency."""
        current = ps.q[self._wheel_qpos]
        return goals.at[self._wheel_rows].set(
            current + wrap(goals[self._wheel_rows] - current)
        )

    def pipeline_step(self, pipeline_state, goals):
        # MuJoCo subtracts RAW actuator length from ctrl. Recenter the angular
        # representative every physics step, keeping actual position PD active
        # between policy updates and through arbitrary whole revolutions.
        def advance(ps, _):
            ctrl = self._position_ctrl(ps, goals)
            return self._pipeline.step(self.sys, ps, ctrl, self._debug), None

        return jax.lax.scan(advance, pipeline_state, (), self._n_frames)[0]

    def _advance_physics(self, state, action):
        # The leg-lift policy's scoped hip clamp: raising AND lowering get a
        # gradual reference while stance hips keep full balance authority.
        info, cfg = state.info, self._config.align
        phase = info["phase"]
        limited = (phase == LIFT) | (phase == ROTATE) | (phase == LOWER)
        previous = info["last_applied_action"][self._hip_rows]
        hip = jp.where(
            limited,
            jp.clip(
                action[self._hip_rows],
                previous - cfg.hip_max_action_delta,
                previous + cfg.hip_max_action_delta,
            ),
            action[self._hip_rows],
        )
        applied = action.at[self._hip_rows].set(hip)
        info["last_applied_action"] = applied
        self._advance_wheel_reference(state)
        signed_scale = (
            self._action_scale[self._wheel_rows] * self._ctrl_sign[self._wheel_rows]
        )
        goal_action = applied.at[self._wheel_rows].set(
            info["wheel_reference"] / signed_scale
        )
        return super()._advance_physics(state, goal_action)

    def _advance_wheel_reference(self, state):
        """Advance angular goals only after measured unloading; always restore holds."""
        info, ps, cfg = state.info, state.pipeline_state, self._config.align
        phase = info["phase"]
        threshold = jp.where(
            phase == ROTATE, cfg.rotate_clearance_min, cfg.clearance_min
        )
        unloaded = (self._wheel_clearance(ps) >= threshold) & (
            self._wheel_loads(ps) < cfg.unloaded_force
        )
        rotating = (phase == ROTATE) & unloaded
        holding = (phase == HOLD) | ((phase == LOWER) & info["rotation_verified"])
        # Stop an interrupted/unloaded-failure rotation without driving toward
        # an unfinished target against the ground, including queued old goals.
        blocked = (((phase == LIFT) | (phase == ROTATE)) & ~unloaded) | (
            (phase == LOWER) & ~info["rotation_verified"]
        )
        reference = jp.where(blocked, ps.q[self._wheel_qpos], info["wheel_reference"])
        delta = jp.clip(
            wrap(info["target"] - reference),
            -cfg.wheel_reference_speed * self.dt,
            cfg.wheel_reference_speed * self.dt,
        )
        reference = wrap(reference + jp.where(rotating | holding, delta, 0.0))
        info["wheel_reference"] = reference
        signed_scale = (
            self._action_scale[self._wheel_rows] * self._ctrl_sign[self._wheel_rows]
        )
        buffer = info["action_buffer"][self._wheel_rows]
        info["action_buffer"] = (
            info["action_buffer"]
            .at[self._wheel_rows]
            .set(
                jp.where(blocked[:, None], (reference / signed_scale)[:, None], buffer)
            )
        )

    def _errors(self, ps, info):
        return wrap(info["target"] - ps.q[self._wheel_qpos])

    def _wheel_loads(self, ps):
        # Contact forces distinguish touching from bearing weight. MJX supplies
        # zero for inactive contacts; match wheel geoms, not contact-array order.
        forces, ids = support.contact_force_dim(self.sys, ps, 3)
        normal = jp.maximum(forces[:, 0], 0.0)
        matches = jp.any(
            ps.contact.geom[ids, :, None] == self._wheel_geom_id[None, None, :], axis=1
        )
        return jp.sum(normal[:, None] * matches, axis=0)

    def _forward_xy(self, ps):
        forward = math.rotate(jp.array([1.0, 0.0, 0.0]), ps.x.rot[self._torso_idx - 1])[
            :2
        ]
        return forward / jp.maximum(jp.linalg.norm(forward), 1e-6)

    def reset(self, rng):
        rng, phase_rng, home_rng, count_rng, order_rng, pose_rng, obs_rng, hold_rng = (
            jax.random.split(rng, 8)
        )
        cfg = self._config.align
        order = jax.random.permutation(order_rng, 4)
        count_rng, full_rng = jax.random.split(count_rng)
        count = jax.random.randint(count_rng, (), 0, cfg.initial_completed_max + 1)
        count = jp.where(
            jax.random.uniform(full_rng) < cfg.full_sequence_probability, 0, count
        )
        completed = jp.zeros(4, dtype=bool).at[order].set(jp.arange(4) < count)
        # Independent startup reference and arbitrary accumulated driving phase.
        # home is latched from calibration_home once on each leg's first button.
        calibration = jax.random.uniform(home_rng, (4,), minval=-jp.pi, maxval=jp.pi)
        phase = jax.random.uniform(phase_rng, (4,), minval=-jp.pi, maxval=jp.pi)
        current = calibration + phase
        current = jp.where(
            completed,
            calibration + jp.pi + phase * cfg.hold_start_error / jp.pi,
            current,
        )
        q = self._init_q.at[self._joint_qpos].add(
            jax.random.uniform(pose_rng, (12,), minval=-1.0, maxval=1.0)
            * self._config.dr.start_joint_jitter
            * self._pos_mask
        )
        q = q.at[self._wheel_qpos].set(current)
        # Initial forward dynamics must not accidentally command raw zero on
        # continuous position joints before the first sampled action arrives.
        initial_goal = jp.where(completed, calibration + jp.pi, current)
        ctrl = self._default_pose.at[self._wheel_rows].set(
            current + wrap(initial_goal - current)
        )
        ps = self.pipeline_init(q, jp.zeros(self._nv), ctrl=ctrl)
        # Avoid a nominal-radius spawn penetrating a larger randomized cylinder.
        q = q.at[2].add(jp.maximum(-jp.min(self._wheel_clearance(ps)), 0.0))
        ps = self.pipeline_init(q, jp.zeros(self._nv), ctrl=ctrl)
        signed_scale = (
            self._action_scale[self._wheel_rows] * self._ctrl_sign[self._wheel_rows]
        )
        action_buffer = (
            jp.zeros((12, self._lat_buf_len))
            .at[self._wheel_rows]
            .set(
                jp.broadcast_to(
                    (initial_goal / signed_scale)[:, None], (4, self._lat_buf_len)
                )
            )
        )
        info = dict(
            rng=rng,
            step=jp.int32(0),
            command=jp.int32(0),
            last_act=jp.zeros(12),
            last_applied_action=jp.zeros(12),
            last_policy_action=jp.zeros(3),
            lift_reference=jp.zeros(4),
            stance_shift=jp.zeros(2),
            phase_age=jp.zeros(4, dtype=jp.int32),
            last_vel=jp.zeros(12),
            action_buffer=action_buffer,
            imu_buffer=jp.zeros((6, self._imu_lat_buf_len)),
            calibration_home=calibration,
            home=calibration,
            target=calibration + jp.pi,
            wheel_idle_target=current,
            wheel_reference=initial_goal,
            rotation_verified=completed,
            tracked=completed,
            completed=completed,
            phase=jp.where(completed, HOLD, IDLE),
            dwell=jp.zeros(4, dtype=jp.int32),
            sequence_order=order,
            sequence_cursor=count,
            command_switch_step=jax.random.randint(
                hold_rng, (), cfg.idle_steps_min, cfg.idle_steps_max + 1
            ),
            init_xy=ps.x.pos[self._torso_idx - 1, :2],
            init_forward=self._forward_xy(ps),
        )
        obs = self._get_obs(
            ps, info, jp.zeros(self._single_obs_dim * self._obs_history), obs_rng
        )
        names = list(self._config.reward_config.scales) + [
            "body_drift_dist",
            "drift_outside_tolerance",
            "active_clearance",
            "active_angle_error",
            "active_scrub_speed",
            "hold_angle_error",
            "held_angle_error",
            "lower_angle_error",
            "new_completions",
            "held_error_sum",
            "held_samples",
            "completed_legs",
            "tracked_legs",
            "torso_z",
            "tilt_deg",
            "wheel_contacts",
            "termination_penalty",
            "active_phase",
        ]
        metrics = {k: jp.float32(0.0) for k in names}
        return State(ps, obs, jp.float32(0.0), jp.float32(0.0), metrics, info)

    def _select(self, info, command):
        """Latch once; never redefine the home of a previously started leg."""
        leg = self._command_leg[command]
        selected = (jp.arange(4) == leg) & (command != 0)
        first = selected & ~info["tracked"]
        interrupted = (info["phase"] == LIFT) | (info["phase"] == ROTATE)
        info["phase"] = jp.where(interrupted & ~selected, LOWER, info["phase"])
        info["home"] = jp.where(first, info["calibration_home"], info["home"])
        info["target"] = jp.where(first, info["home"] + jp.pi, info["target"])
        info["tracked"] = info["tracked"] | first
        # Retrying an unfinished leg uses the SAME target. Completed legs hold.
        info["phase"] = jp.where(
            selected & ~info["completed"] & (command != info["command"]),
            LIFT,
            info["phase"],
        )
        info["dwell"] = jp.where(
            selected & (command != info["command"]), 0, info["dwell"]
        )
        info["phase_age"] = jp.where(
            selected & (command != info["command"]), 0, info["phase_age"]
        )
        info["rotation_verified"] = jp.where(
            selected & ~info["completed"] & (command != info["command"]),
            False,
            info["rotation_verified"],
        )
        info["command"] = command

    def set_command(self, state, command):
        """External command index: stand/front_l/front_r/back_r/back_l (0..4)."""
        info = dict(state.info)
        self._select(info, jp.asarray(command, dtype=jp.int32))
        info["rng"], key = jax.random.split(info["rng"])
        obs = self._get_obs(state.pipeline_state, info, state.obs, key)
        return state.replace(info=info, obs=obs)

    def _schedule(self, info):
        cfg = self._config.align
        info["rng"], idle_rng, active_rng = jax.random.split(info["rng"], 3)
        switch = (info["step"] >= info["command_switch_step"]) & (
            info["sequence_cursor"] < 5
        )
        start = (info["command"] == 0) & (info["sequence_cursor"] < 4)
        leg = info["sequence_order"][jp.minimum(info["sequence_cursor"], 3)]
        command = jp.where(start, self._leg_command[leg], 0)
        self._select(info, jp.where(switch, command, info["command"]))
        info["sequence_cursor"] += (switch & start).astype(jp.int32)
        # Cursor 4 still needs the final switch to stand; 5 means indefinite hold.
        info["sequence_cursor"] = jp.where(
            switch & ~start & (info["sequence_cursor"] == 4), 5, info["sequence_cursor"]
        )
        duration = jp.where(
            start,
            jax.random.randint(
                active_rng, (), cfg.active_steps_min, cfg.active_steps_max + 1
            ),
            jax.random.randint(
                idle_rng, (), cfg.idle_steps_min, cfg.idle_steps_max + 1
            ),
        )
        info["command_switch_step"] = jp.where(
            switch, info["step"] + duration, info["command_switch_step"]
        )

    def _advance_phase(self, info, clearance, load, error, speed):
        cfg = self._config.align
        phase = info["phase"]
        # Hysteresis: enter at 3 mm; remain in rotation down to 1 mm only if
        # still contact-free. The old single threshold repeatedly reset rotation.
        min_clearance = jp.where(
            phase == ROTATE, cfg.rotate_clearance_min, cfg.clearance_min
        )
        unloaded = (clearance >= min_clearance) & (load < cfg.unloaded_force)
        settled = (jp.abs(error) < cfg.angle_tolerance) & (
            jp.abs(speed) < cfg.speed_tolerance
        )
        reweighted = (clearance < cfg.contact_tolerance) & (load >= cfg.reweight_force)
        ready = (
            ((phase == LIFT) & unloaded)
            | ((phase == ROTATE) & unloaded & settled)
            | ((phase == LOWER) & info["rotation_verified"] & reweighted & settled)
        )
        dwell = jp.where(ready, info["dwell"] + 1, 0)
        required = jp.where(
            phase == LIFT,
            cfg.unload_dwell_steps,
            jp.where(
                phase == ROTATE, cfg.settle_dwell_steps, cfg.touchdown_dwell_steps
            ),
        )
        advance = ready & (dwell >= required)
        info["rotation_verified"] |= (phase == ROTATE) & advance
        # A touchdown during rotation returns to lift; spinning on ground cannot
        # earn rotation/settling credit. HOLD never forgets its angle feedback.
        previous_phase = phase
        phase = jp.where((phase == ROTATE) & ~unloaded, LIFT, phase)
        info["phase"] = jp.where(advance, phase + 1, phase)
        info["phase_age"] = jp.where(
            info["phase"] == previous_phase, info["phase_age"] + 1, 0
        )
        info["dwell"] = jp.where(advance, 0, dwell)
        newly_complete = (info["phase"] == HOLD) & ~info["completed"]
        info["completed"] |= newly_complete
        return jp.sum(newly_complete.astype(jp.float32))

    def step(self, state, action):
        if action.shape != (self.action_size,):
            raise ValueError(
                f"Expected {self.action_size} clearance/balance actions, got {action.shape}"
            )
        policy_action = jp.clip(action, -1.0, 1.0)
        # Explicit wheel-speed disturbances exercise restoring feedback even in
        # idle and partial-completion resets. Keep inherited torso pushes intact.
        info = dict(state.info)
        state = state.replace(info=info)
        action = self._leg_action(state, policy_action)
        info["rng"], kick_rng, apply_rng = jax.random.split(info["rng"], 3)
        cfg = self._config.align
        kick = jax.random.uniform(
            kick_rng,
            (4,),
            minval=-cfg.wheel_disturbance_speed,
            maxval=cfg.wheel_disturbance_speed,
        )
        kick *= (
            jax.random.bernoulli(apply_rng, cfg.wheel_disturbance_probability, (4,))
            * info["tracked"]
        )
        ps = state.pipeline_state.replace(
            qd=state.pipeline_state.qd.at[self._wheel_dof].add(kick)
        )
        state = state.replace(info=info, pipeline_state=ps)
        ps, obs_rng = self._advance_physics(state, action)
        clearance, load = self._wheel_clearance(ps), self._wheel_loads(ps)
        error, speed = self._errors(ps, info), ps.qd[self._wheel_dof]
        old_phase = info["phase"]
        completion = self._advance_phase(info, clearance, load, error, speed)
        rewards, diagnostics = self._alignment_reward(
            ps, info, action, clearance, load, error, speed, old_phase
        )
        rewards["action_rate"] = jp.mean(
            jp.square(policy_action - info["last_policy_action"])
        )
        rewards["completion"] = completion
        rewards = {
            k: rewards[k] * weight
            for k, weight in self._config.reward_config.scales.items()
        }
        torso_z = ps.x.pos[self._torso_idx - 1, 2]
        cos_tilt = math.rotate(
            jp.array([0.0, 0.0, 1.0]), ps.x.rot[self._torso_idx - 1]
        )[2]
        done = (cos_tilt < jp.cos(self._config.terminal_body_angle)) | (
            torso_z < self._config.terminal_body_z
        )
        done |= ~jp.all(jp.isfinite(ps.q)) | ~jp.all(jp.isfinite(ps.qd))
        termination = self._config.reward_config.termination_penalty * self.dt * done
        # A completion is an event, not a per-second reward. Do not multiply its
        # bonus by dt as the pilot did (8 became only 0.16 per actual completion).
        reward = (
            (sum(rewards.values()) - rewards["completion"]) * self.dt
            + rewards["completion"]
            + termination
        )
        info["last_act"], info["last_vel"] = action, ps.qd[self._joint_dof]
        info["last_policy_action"] = policy_action
        info["step"] += 1
        if self._config.automatic_commands:
            self._schedule(info)
        obs = self._get_obs(ps, info, state.obs, obs_rng)
        metrics = dict(state.metrics)
        metrics.update(rewards)
        metrics.update(diagnostics)
        metrics.update(
            torso_z=torso_z,
            tilt_deg=jp.rad2deg(jp.arccos(jp.clip(cos_tilt, -1.0, 1.0))),
            termination_penalty=termination,
            new_completions=completion,
        )
        return state.replace(
            pipeline_state=ps,
            obs=obs,
            info=info,
            metrics=metrics,
            reward=reward,
            done=jp.float32(done),
        )

    def _alignment_reward(self, ps, info, action, clearance, load, error, speed, phase):
        cfg = self._config.align
        selected = (jp.arange(4) == self._command_leg[info["command"]]) & (
            info["command"] != 0
        )
        airborne = (phase == LIFT) | (phase == ROTATE)
        contact = (clearance < cfg.contact_tolerance) | (load >= cfg.unloaded_force)
        reweight = jp.clip(load / cfg.reweight_force, 0.0, 1.0)
        holding = info["tracked"] & ((phase == HOLD) | (phase == LOWER))
        nhold = jp.maximum(jp.sum(holding), 1)
        pos = ps.q[self._joint_qpos[self._pos_rows]]
        pos_default = self._default_pose[self._pos_rows]
        stance_mask = ~airborne[self._position_leg]
        torso = self._torso_idx - 1
        drift = jp.linalg.norm(ps.x.pos[torso, :2] - info["init_xy"])
        cos_tilt = math.rotate(jp.array([0.0, 0.0, 1.0]), ps.x.rot[torso])[2]
        # Penalize knee sphere proximity so lifting cannot be supported by a knee.
        knee_clearance = ps.x.pos[self._wheel_body_id - 1, 2] - configs.KNEE_RADIUS
        # Small set of directly interpretable objectives. Holding initialized
        # targets earns no bonus: it only incurs an error cost when disturbed.
        desired_stance = pos_default + (self._shift_map @ info["stance_shift"])
        supporting_lift = jp.any(airborne) | jp.any(info["lift_reference"] > 1e-4)
        allowed_drift = jp.where(
            supporting_lift, cfg.support_body_drift, cfg.allowed_body_drift
        )
        rewards = dict(
            alive=jp.float32(1.0),
            lift=jp.sum(
                airborne
                * jp.minimum(
                    jp.square(
                        (clearance - cfg.clearance_target) / cfg.clearance_target
                    ),
                    4.0,
                )
            ),
            # Grounded clearance barely changes while weight transfers. Give
            # balance corrections a dense signal before the wheel leaves the
            # floor, normalized by supported weight across randomized masses.
            # Once airborne this costs zero; no extra reward for a higher lift.
            unload=jp.sum((phase == LIFT) * load)
            / jp.maximum(jp.sum(load) / 4.0, 1.0),
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
                info["rotation_verified"]
                * (
                    jp.minimum(jp.square(error / cfg.angle_tolerance), 4.0)
                    + jp.minimum(jp.square(speed / 0.4), 4.0)
                )
            ),
            active_scrub=jp.sum(selected * contact * jp.square(speed)),
            support=jp.sum(~airborne * (1.0 - reweight)),
            stance_pose=jp.mean(stance_mask * jp.square((pos - desired_stance) / 0.15)),
            orientation=(1.0 - cos_tilt) / 0.05,
            body_drift=jp.square(
                jp.maximum(drift - allowed_drift, 0.0) / cfg.drift_scale
            ),
            action_rate=jp.float32(
                0.0
            ),  # actual three policy outputs are scored in step
            ground_contact=jp.sum(jp.clip((0.015 - knee_clearance) / 0.015, 0.0, 1.0)),
        )
        diagnostics = dict(
            body_drift_dist=drift,
            drift_outside_tolerance=(drift > cfg.allowed_body_drift).astype(jp.float32),
            active_clearance=jp.sum(selected * clearance),
            active_angle_error=jp.sum(selected * jp.abs(error)),
            active_scrub_speed=jp.sum(selected * contact * jp.abs(speed)),
            hold_angle_error=jp.sum(holding * jp.abs(error)) / nhold,
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
        )
        return rewards, diagnostics

    def _get_obs(self, ps, info, history, rng):
        rng, encoder_rng = jax.random.split(rng)
        ang, gravity, joints, _ = self._proprioception(ps, info, rng)
        error = (
            self._errors(ps, info)
            + jax.random.uniform(encoder_rng, (4,), minval=-1.0, maxval=1.0)
            * self._config.align.encoder_angle_noise
        )
        # Targets of unstarted legs are hidden until their calibration is latched.
        sin_error = jp.sin(error) * info["tracked"]
        cos_error = jp.cos(error) * info["tracked"]
        forward, initial = self._forward_xy(ps), info["init_forward"]
        heading = jp.array(
            [
                initial[0] * forward[1] - initial[1] * forward[0],
                jp.dot(initial, forward),
            ]
        )
        obs = jp.concatenate(
            [
                ang,
                gravity,
                joints,
                info["last_policy_action"],
                jax.nn.one_hot(info["command"], configs.NUM_COMMANDS),
                sin_error,
                cos_error,
                jax.nn.one_hot(info["phase"], 5).ravel(),
                info["tracked"].astype(jp.float32),
                info["completed"].astype(jp.float32),
                self._wheel_clearance(ps) / 0.01,
                self._wheel_loads(ps) / 10.0,
                (ps.x.pos[self._torso_idx - 1, :2] - info["init_xy"]) / 0.05,
                self._body_frame_vel(ps)[:2],
                heading,
                jp.sin(info["wheel_reference"] - ps.q[self._wheel_qpos]),
                jp.cos(info["wheel_reference"] - ps.q[self._wheel_qpos]),
                info["lift_reference"],
                info["stance_shift"] / 0.02,
                jp.minimum(info["phase_age"] / self._config.align.prepare_steps, 1.0),
            ]
        )
        return (
            jp.roll(history, obs.size).at[: obs.size].set(jp.clip(obs, -100.0, 100.0))
        )
