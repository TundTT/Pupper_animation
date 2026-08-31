"""Pupper V3 wheeled-locomotion environment (MJX / brax).

Trains ONE policy to drive the robot at a commanded body velocity
(vx, vy, yaw rate) on four wheels, holding the torso upright and level and the
legs in the splayed `configs.DEFAULT_POSE` stance.

Structurally this follows the providers' `pupperv3_mjx.environment.PupperV3Env`
(action = offset from a default ctrl vector; gravity/angular-velocity
proprioception; a command resampled during the episode; the same sensor-noise /
kick / action-latency domain randomization) rather than this repo's
`leg_lift_env.py`, whose reward and command space describe a completely
different task. `leg_lift_env.py` is left untouched as a reference.

Mixed actuation -- the one structural difference from both references
---------------------------------------------------------------------
This robot's 12 joints are NOT uniformly actuated. Per leg:

  _1 (abduction), _2 (hip)  -> POSITION control, ctrl is an angle   (rad)
  _3 (wheel)                -> VELOCITY control, ctrl is a speed    (rad/s)

Both reference envs do

    sys.replace(actuator_gainprm=...at[:, 0].set(kp),
                actuator_biasprm=...at[:, 1].set(-kp).at[:, 2].set(-kd))

i.e. they stamp ONE position-PD gain set over EVERY actuator row. Doing that
here would quietly overwrite the wheels' velocity gains with position gains and
produce a model that still loads and still trains, just against wrong physics.
`_override_actuator_gains` below writes the two groups separately, indexed by
`configs.POSITION_ACTUATOR_ROWS` / `configs.WHEEL_ACTUATOR_ROWS`.

The same split runs through the rest of the file:
  * `DEFAULT_POSE` and `ACTION_SCALE` are mixed-unit vectors (rad on the
    position rows, rad/s on the wheel rows), so the shared
    `default + action * scale` ctrl formula stays valid for both groups.
  * The joint OBSERVATION reports angle for the position rows but SPEED for the
    wheel rows. A free-spinning wheel's angle grows without bound and wraps, so
    feeding it to the policy would inject a meaningless unbounded input; its
    speed is both bounded and the thing the policy actually needs to know.
  * `dof_pos_limits` and `stance_pose` are computed over the position rows only
    -- a continuously-spinning joint has neither a limit to respect nor a home
    angle to hold.

Status: no wheeled policy has been trained yet. Reward weights and command
ranges in `configs.get_wheel_config()` are a first pass, not tuned values.
"""

from typing import Any, Dict, List, Optional, Sequence

import jax
import mujoco
import numpy as np
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jp
from ml_collections import config_dict

from workspace import configs


def _buf_push_front(buf: jax.Array, new_val: jax.Array) -> jax.Array:
    """Roll buffer forward and place new_val at index 0 (newest slot)."""
    return jp.roll(buf, shift=1, axis=1).at[:, 0].set(new_val)


def _sample_lagged(rng: jax.Array, buf: jax.Array, new_val: jax.Array, dist: jax.Array):
    """Push new_val and sample one time-slot from `buf` according to `dist`.

    buf shape: (action_dim, buf_len); dist shape: (buf_len,); newest first.
    """
    buf = _buf_push_front(buf, new_val)
    return jax.random.choice(rng, buf, axis=1, p=dist), buf


class PupperWheelEnv(PipelineEnv):
    """Drive the wheeled Pupper at a commanded body velocity."""

    def __init__(self, config: config_dict.ConfigDict, model_path: Optional[str] = None):
        self._config = config
        path = str(configs.resolve_model_path(model_path))

        sys = mjcf.load(path)
        self._dt = config.ctrl_dt
        sys = sys.tree_replace({"opt.timestep": config.sim_dt})

        self._assert_actuator_layout(sys.mj_model)
        sys = self._override_actuator_gains(sys, config)

        n_frames = int(round(self._dt / sys.opt.timestep))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self._default_pose = jp.array(configs.DEFAULT_POSE)
        self._lowers = jp.array(configs.JOINT_LOWER_LIMITS)
        self._uppers = jp.array(configs.JOINT_UPPER_LIMITS)
        self._action_scale = jp.array(config.action_scale)
        assert self._action_scale.shape == (12,), "action_scale must be per-joint (12,)"

        # Row masks for the two actuation groups (see module docstring).
        self._pos_rows = np.array(configs.POSITION_ACTUATOR_ROWS)
        self._wheel_rows = np.array(configs.WHEEL_ACTUATOR_ROWS)
        pos_mask = np.zeros(12)
        pos_mask[self._pos_rows] = 1.0
        self._pos_mask = jp.array(pos_mask)
        wheel_mask = np.zeros(12)
        wheel_mask[self._wheel_rows] = 1.0
        self._wheel_mask = jp.array(wheel_mask)

        # Per-row sign that makes "+1 = drives the robot forward" true on every
        # wheel despite the left/right legs' mirrored frames (see
        # configs.WHEEL_FORWARD_SIGN). 1.0 on the position rows leaves them alone.
        ctrl_sign = np.ones(12)
        ctrl_sign[self._wheel_rows] = configs.WHEEL_FORWARD_SIGN
        self._ctrl_sign = jp.array(ctrl_sign)

        # Soft joint limits, position rows only. The wheel rows of
        # JOINT_*_LIMITS are velocity ctrl bounds, not angle bounds, so folding
        # them into a joint-angle penalty would be meaningless -- the mask below
        # zeroes them out of `dof_pos_limits`.
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        f = config.soft_joint_pos_limit_factor
        self._soft_lowers = c - 0.5 * r * f
        self._soft_uppers = c + 0.5 * r * f

        # Spawn already settled on the wheels rather than dropping from the
        # keyframe's z=0.28, which would bounce and waste the start of every
        # episode. Same approach as leg_lift_env, re-measured for this model.
        self._init_q = jp.array(self._settled_qpos(sys.mj_model))
        self._nv = sys.nv

        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, configs.TORSO_NAME
        )
        assert self._torso_idx != -1, "torso body not found"
        wheel_ids = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, b)
            for b in configs.WHEEL_BODY_NAMES
        ]
        assert -1 not in wheel_ids, "a wheel body was not found"
        self._wheel_body_id = np.array(wheel_ids)
        self._wheel_radius = configs.WHEEL_RADIUS
        self._wheel_center_local = jp.array([0.0, 0.0, configs.WHEEL_CENTER_LOCAL_Z])
        self._stand_height = configs.STAND_TORSO_HEIGHT

        # [ang_vel(3), gravity(3), command(3), joint_pos_or_vel(12), last_act(12)]
        self._single_obs_dim = 3 + 3 + 3 + 12 + 12
        self._obs_history = config.observation_history

        lat_dist = list(config.dr.latency_distribution)
        self._lat_dist = jp.array(lat_dist)
        self._lat_buf_len = len(lat_dist)

    # ------------------------------------------------------- model / actuators
    @staticmethod
    def _assert_actuator_layout(mj_model) -> None:
        """Fail loudly if the model is not the 12-joint mixed-actuation wheeled one.

        Everything in this env indexes joints and actuators by the same row
        number (configs.JOINT_NAMES order), and splits those rows into position
        vs. wheel groups by hard-coded index. If the model ever changes shape or
        ordering, that assumption breaks SILENTLY -- gains land on the wrong
        actuator, the obs mixes rad with rad/s -- so it is checked here instead.
        """
        if mj_model.nu != 12:
            raise ValueError(f"expected 12 actuators on the wheeled model, found {mj_model.nu}")
        for row, name in enumerate(configs.JOINT_NAMES):
            act_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, row)
            if act_name != name:
                raise ValueError(
                    f"actuator row {row} is '{act_name}', expected '{name}'. "
                    "configs.JOINT_NAMES must match the model's actuator order."
                )
            joint_id = mj_model.actuator_trnid[row, 0]
            joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, joint_id)
            if joint_name != name:
                raise ValueError(
                    f"actuator '{act_name}' drives joint '{joint_name}', expected '{name}'."
                )
        for row in configs.WHEEL_ACTUATOR_ROWS:
            joint_id = mj_model.actuator_trnid[row, 0]
            if mj_model.jnt_limited[joint_id]:
                raise ValueError(
                    f"wheel joint '{configs.JOINT_NAMES[row]}' is limited; the wheel joints "
                    'must be continuous (limited="false") on this model.'
                )
        # The MJCF's "home" keyframe must agree with DEFAULT_POSE on the position
        # rows -- the keyframe is what reset() settles from, DEFAULT_POSE is what
        # actions are offsets from, and if they drift apart the policy starts every
        # episode being commanded away from where it was spawned.
        key_qpos = mj_model.keyframe("home").qpos[7:]
        pos_rows = configs.POSITION_ACTUATOR_ROWS
        if not np.allclose(key_qpos[pos_rows], configs.DEFAULT_POSE[pos_rows], atol=1e-6):
            raise ValueError(
                "the model's 'home' keyframe disagrees with configs.DEFAULT_POSE on the "
                f"position joints: keyframe={key_qpos[pos_rows]}, "
                f"DEFAULT_POSE={configs.DEFAULT_POSE[pos_rows]}"
            )

    @staticmethod
    def _override_actuator_gains(sys, config):
        """Write kp/kd onto the position rows and kv onto the wheel rows.

        Deliberately NOT the single blanket `.at[:, 0].set(kp)` both reference
        envs use -- see the module docstring. For a MuJoCo position actuator the
        affine bias is (0, -kp, -kd); for a velocity actuator it is (0, 0, -kv).
        """
        kp = config.position_control_kp
        kd = config.dof_damping
        kv = config.wheel_kv
        pos = np.array(configs.POSITION_ACTUATOR_ROWS)
        wheel = np.array(configs.WHEEL_ACTUATOR_ROWS)

        sys = sys.replace(
            actuator_gainprm=sys.actuator_gainprm.at[pos, 0].set(kp).at[wheel, 0].set(kv),
            actuator_biasprm=(
                sys.actuator_biasprm
                .at[pos, 1].set(-kp).at[pos, 2].set(-kd)
                .at[wheel, 1].set(0.0).at[wheel, 2].set(-kv)
            ),
        )
        # Mirror onto the underlying MjModel so the plain-MuJoCo settling pass
        # below sees the same actuator model training will use.
        sys.mj_model.actuator_gainprm[pos, 0] = kp
        sys.mj_model.actuator_biasprm[pos, 1] = -kp
        sys.mj_model.actuator_biasprm[pos, 2] = -kd
        sys.mj_model.actuator_gainprm[wheel, 0] = kv
        sys.mj_model.actuator_biasprm[wheel, 1] = 0.0
        sys.mj_model.actuator_biasprm[wheel, 2] = -kv
        # NOTE: unlike leg_lift_env, the "home" keyframe is deliberately NOT
        # rewritten from DEFAULT_POSE here -- the MJCF's keyframe already IS the
        # splayed wheeled pose, and its wheel entries are joint ANGLES (0) while
        # DEFAULT_POSE's are wheel SPEEDS, so copying one onto the other would be
        # a unit error. configs.DEFAULT_POSE is asserted to agree with the
        # keyframe's position rows at construction instead.
        return sys

    @staticmethod
    def _settled_qpos(mj_model, settle_seconds: float = 3.0) -> np.ndarray:
        """Simulate the default pose to rest and return the resulting qpos."""
        data = mujoco.MjData(mj_model)
        mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        data.ctrl[:] = configs.DEFAULT_POSE
        for _ in range(int(settle_seconds / mj_model.opt.timestep)):
            mujoco.mj_step(mj_model, data)
        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError("settling the wheeled pose diverged; check the model/gains")
        qpos = data.qpos.copy()
        qpos[:2] = 0.0  # re-center horizontally; only height/attitude matter
        return qpos

    # ------------------------------------------------------------------ helpers
    def _wheel_center_z(self, pipeline_state: base.State) -> jax.Array:
        """World z of each wheel's centre (4,).

        The `_3` body origin is the motor output, not the wheel centre, so the
        local offset is rotated into world coordinates rather than just read off
        the body position.
        """
        rows = self._wheel_body_id - 1  # x.pos excludes the world body
        offsets = jax.vmap(math.rotate, in_axes=(None, 0))(
            self._wheel_center_local, pipeline_state.x.rot[rows]
        )
        return pipeline_state.x.pos[rows][:, 2] + offsets[:, 2]

    def _body_frame_vel(self, pipeline_state: base.State) -> jax.Array:
        """Torso linear velocity expressed in the torso frame (3,)."""
        inv_rot = math.quat_inv(pipeline_state.x.rot[self._torso_idx - 1])
        return math.rotate(pipeline_state.xd.vel[self._torso_idx - 1], inv_rot)

    def _body_frame_angvel(self, pipeline_state: base.State) -> jax.Array:
        inv_rot = math.quat_inv(pipeline_state.x.rot[self._torso_idx - 1])
        return math.rotate(pipeline_state.xd.ang[self._torso_idx - 1], inv_rot)

    # ------------------------------------------------------------------ command
    def _sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample (vx, vy, yaw_rate), sometimes exactly zero."""
        rng_x, rng_y, rng_w, rng_zero = jax.random.split(rng, 4)
        cfg = self._config
        vx = jax.random.uniform(rng_x, (), minval=cfg.lin_vel_x_range[0], maxval=cfg.lin_vel_x_range[1])
        vy = jax.random.uniform(rng_y, (), minval=cfg.lin_vel_y_range[0], maxval=cfg.lin_vel_y_range[1])
        wz = jax.random.uniform(rng_w, (), minval=cfg.ang_vel_yaw_range[0], maxval=cfg.ang_vel_yaw_range[1])
        cmd = jp.array([vx, vy, wz])
        is_zero = jax.random.uniform(rng_zero, ()) < cfg.zero_command_prob
        return jp.where(is_zero, jp.zeros(3), cmd)

    def _sample_hold(self, rng: jax.Array) -> jax.Array:
        return jax.random.randint(
            rng, (), self._config.command_resample_steps_min,
            self._config.command_resample_steps_max + 1,
        )

    # ------------------------------------------------------------------ reset
    def reset(self, rng: jax.Array) -> State:
        rng, cmd_rng, hold_rng, obs_rng = jax.random.split(rng, 4)
        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        info = {
            "rng": rng,
            "step": 0,
            "command": self._sample_command(cmd_rng),
            "command_switch_step": self._sample_hold(hold_rng),
            "last_act": jp.zeros(12),
            "last_vel": jp.zeros(12),
            "action_buffer": jp.zeros((12, self._lat_buf_len)),
        }
        obs_history = jp.zeros(self._obs_history * self._single_obs_dim)
        obs = self._get_obs(pipeline_state, info, obs_history, obs_rng)
        metrics: Dict[str, Any] = {k: 0.0 for k in self._config.reward_config.scales.keys()}
        metrics["lin_vel_error"] = 0.0
        metrics["ang_vel_error"] = 0.0
        metrics["torso_z"] = 0.0
        metrics["tilt_deg"] = 0.0
        metrics["wheel_contacts"] = 0.0
        metrics["termination_penalty"] = 0.0
        return State(pipeline_state, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # ------------------------------------------------------------------- step
    def step(self, state: State, action: jax.Array) -> State:
        state.info["rng"], kick_rng, lat_rng, obs_rng = jax.random.split(state.info["rng"], 4)

        # Action latency: sample which recent action actually reaches the motors.
        lagged_action, action_buffer = _sample_lagged(
            lat_rng, state.info["action_buffer"], action, self._lat_dist
        )
        state.info["action_buffer"] = action_buffer

        # Random horizontal kick to the torso (low probability).
        kick_rng1, kick_rng2 = jax.random.split(kick_rng)
        kick_vec = jax.random.uniform(kick_rng1, (2,), minval=-1.0, maxval=1.0) * self._config.dr.kick_vel
        kick_applied = jax.random.bernoulli(kick_rng2, self._config.dr.kick_probability)
        kicked_qd = state.pipeline_state.qd.at[:2].set(
            state.pipeline_state.qd[:2] + kick_vec * kick_applied
        )

        # Mixed-unit ctrl: rad on the position rows, rad/s on the wheel rows.
        # The clip bounds carry the same mixed units (see configs). `_ctrl_sign`
        # flips the wheel rows onto the model's mirrored left/right spin axes so
        # a positive action means "forward" on all four wheels.
        motor_targets = jp.clip(
            self._default_pose + lagged_action * self._action_scale * self._ctrl_sign,
            self._lowers,
            self._uppers,
        )
        pipeline_state = self.pipeline_step(state.pipeline_state.replace(qd=kicked_qd), motor_targets)

        obs = self._get_obs(pipeline_state, state.info, state.obs, obs_rng)

        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]
        x = pipeline_state.x

        command = state.info["command"]
        body_vel = self._body_frame_vel(pipeline_state)
        body_angvel = self._body_frame_angvel(pipeline_state)
        lin_vel_error = jp.sum(jp.square(command[:2] - body_vel[:2]))
        ang_vel_error = jp.square(command[2] - body_angvel[2])

        wheel_clearance = self._wheel_center_z(pipeline_state) - self._wheel_radius
        wheel_contact = wheel_clearance < 5e-3

        up = jp.array([0.0, 0.0, 1.0])
        cos_tilt = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up)
        torso_z = x.pos[self._torso_idx - 1, 2]

        done = cos_tilt < jp.cos(self._config.terminal_body_angle)
        done |= torso_z < self._config.terminal_body_z

        rewards = self._get_reward(
            command, joint_angles, joint_vel, pipeline_state, body_vel, body_angvel,
            lin_vel_error, ang_vel_error, wheel_contact, cos_tilt, torso_z, action, state.info,
        )
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        # The shaped terms are clipped to >=0 (inherited from the leg-lift env), so
        # the fall penalty has to be added AFTER the clip -- inside it, a negative
        # term would simply be erased and the penalty would silently do nothing.
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        termination_penalty = self._config.reward_config.termination_penalty * self.dt * done
        reward = reward + termination_penalty

        # Resample the velocity command when the hold window elapses.
        state.info["rng"], cmd_rng, hold_rng = jax.random.split(state.info["rng"], 3)
        switch = state.info["step"] >= state.info["command_switch_step"]
        state.info["command"] = jp.where(switch, self._sample_command(cmd_rng), command)
        state.info["command_switch_step"] = jp.where(
            switch, state.info["step"] + self._sample_hold(hold_rng),
            state.info["command_switch_step"],
        )

        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["step"] = state.info["step"] + 1

        state.metrics.update(rewards)
        state.metrics["lin_vel_error"] = lin_vel_error
        state.metrics["ang_vel_error"] = ang_vel_error
        state.metrics["torso_z"] = torso_z
        state.metrics["tilt_deg"] = jp.rad2deg(jp.arccos(jp.clip(cos_tilt, -1.0, 1.0)))
        state.metrics["wheel_contacts"] = jp.sum(wheel_contact.astype(jp.float32))
        state.metrics["termination_penalty"] = termination_penalty

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=jp.float32(done))

    # ----------------------------------------------------------------- reward
    def _get_reward(
        self, command, joint_angles, joint_vel, pipeline_state, body_vel, body_angvel,
        lin_vel_error, ang_vel_error, wheel_contact, cos_tilt, torso_z, action, info,
    ) -> Dict[str, jax.Array]:
        """Reward: track the commanded velocity while holding the wheeled stance."""
        cfg = self._config.reward_config

        # -- the task ----------------------------------------------------------
        tracking_lin_vel = jp.exp(-lin_vel_error / cfg.tracking_lin_vel_sigma)
        tracking_ang_vel = jp.exp(-ang_vel_error / cfg.tracking_ang_vel_sigma)

        # -- hold the wheeled stance -------------------------------------------
        orientation = jp.exp(-(1.0 - jp.clip(cos_tilt, -1.0, 1.0)) / cfg.orientation_sigma)
        torso_height = jp.exp(-jp.square(torso_z - self._stand_height) / cfg.torso_height_sigma)

        # Position joints only: a wheel has no home angle to hold.
        stance_err = jp.sum(
            self._pos_mask * jp.square(joint_angles - self._default_pose)
        )
        stance_pose = jp.exp(-stance_err / cfg.stance_pose_sigma)

        wheels_on_ground = jp.mean(wheel_contact.astype(jp.float32))

        # Zero command => actually come to rest. Only paid out when the command
        # really is (near) zero, so it never fights the tracking terms.
        cmd_speed = jp.linalg.norm(command)
        is_stand = (cmd_speed < self._config.stand_still_threshold).astype(jp.float32)
        body_speed_sq = jp.sum(jp.square(body_vel[:2])) + jp.square(body_angvel[2])
        stand_still = is_stand * jp.exp(-body_speed_sq / cfg.stand_still_sigma)

        # -- penalties ----------------------------------------------------------
        lin_vel_z = jp.square(body_vel[2])
        ang_vel_xy = jp.sum(jp.square(body_angvel[:2]))
        action_rate = jp.sum(jp.square(action - info["last_act"]))
        torques = jp.sum(jp.square(pipeline_state.qfrc_actuator[6:]))
        dof_acc = jp.sum(jp.square((joint_vel - info["last_vel"]) / self._dt))

        # Position rows only -- the wheel rows of _soft_* are velocity bounds.
        out_lo = -jp.clip(joint_angles - self._soft_lowers, None, 0.0)
        out_hi = jp.clip(joint_angles - self._soft_uppers, 0.0, None)
        dof_pos_limits = jp.sum(self._pos_mask * (out_lo + out_hi))

        return {
            "tracking_lin_vel": tracking_lin_vel,
            "tracking_ang_vel": tracking_ang_vel,
            "orientation": orientation,
            "torso_height": torso_height,
            "stance_pose": stance_pose,
            "wheels_on_ground": wheels_on_ground,
            "stand_still": stand_still,
            "lin_vel_z": lin_vel_z,
            "ang_vel_xy": ang_vel_xy,
            "action_rate": action_rate,
            "torques": torques,
            "dof_acc": dof_acc,
            "dof_pos_limits": dof_pos_limits,
        }

    # -------------------------------------------------------------------- obs
    def _get_obs(
        self, pipeline_state: base.State, info: dict[str, Any], obs_history: jax.Array, rng: jax.Array
    ) -> jax.Array:
        dr = self._config.dr
        ang_rng, grav_rng, jpos_rng, jvel_rng, lact_rng = jax.random.split(rng, 5)

        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        ang_vel = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        ang_vel = ang_vel + jax.random.uniform(ang_rng, (3,), minval=-1.0, maxval=1.0) * dr.angular_velocity_noise

        gravity = math.rotate(jp.array([0.0, 0.0, -1.0]), inv_torso_rot)
        gravity = gravity + jax.random.uniform(grav_rng, (3,), minval=-1.0, maxval=1.0) * dr.gravity_noise

        # Mixed joint observation: ANGLE (rel. to default) on the position rows,
        # SPEED on the wheel rows. A free-spinning wheel's angle is unbounded and
        # wraps, which would be a meaningless input; its speed is what matters
        # and is what the real encoder reports usefully for a drive wheel.
        jpos = (pipeline_state.q[7:] - self._default_pose) * self._pos_mask
        jpos = jpos + jax.random.uniform(jpos_rng, (12,), minval=-1.0, maxval=1.0) * dr.motor_angle_noise * self._pos_mask
        # Sign-corrected the same way the ctrl is, so a wheel rolling the robot
        # forward always reads positive regardless of which side it is on.
        jvel = pipeline_state.qd[6:] * self._wheel_mask * self._ctrl_sign
        jvel = jvel + jax.random.uniform(jvel_rng, (12,), minval=-1.0, maxval=1.0) * dr.wheel_velocity_noise * self._wheel_mask
        # Wheel speeds are normalized to the same O(1) range as the angle rows so
        # neither dominates the (shared) observation normalizer.
        joint_obs = jpos + jvel / configs.WHEEL_MAX_SPEED

        last_act = info["last_act"]
        last_act = last_act + jax.random.uniform(lact_rng, (12,), minval=-1.0, maxval=1.0) * dr.last_action_noise

        obs = jp.concatenate([
            ang_vel,             # 3
            gravity,             # 3
            info["command"],     # 3  (vx, vy, yaw rate)
            joint_obs,           # 12 (rad on position rows, normalized rad/s on wheel rows)
            last_act,            # 12
        ])
        obs = jp.clip(obs, -100.0, 100.0)
        return jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

    def render(
        self,
        trajectory: List[base.State],
        camera: Optional[str] = None,
        height: int = 480,
        width: int = 640,
    ) -> Sequence[np.ndarray]:
        return super().render(trajectory, camera=camera or "tracking_cam", height=height, width=width)
