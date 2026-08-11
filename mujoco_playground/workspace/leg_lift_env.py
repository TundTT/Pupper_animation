"""Pupper V3 leg-lift environment (MJX / brax).

Trains ONE policy that, on command, raises a single leg and holds it in the air
while balancing on the other three, then lowers it when the command changes.
Structurally this mirrors the providers' `pupperv3_mjx.environment.PupperV3Env`
(action = position offset from the home pose; gravity/angular-velocity
proprioception; a command resampled during the episode) and the design of
mujoco_playground's go1 `Getup` task.

The command
-----------
The policy is conditioned on a discrete command = which leg is currently up
(stand / front_l / front_r / back_r / back_l), supplied as a one-hot. During
training the command is held for a random window and then switched, which teaches
the policy smooth raise/hold/lower transitions. On the robot the command is
advanced by the O button (a clockwise state machine); "hold" is simply the command
not changing, so hold duration is operator-controlled and not baked into the
policy. See README.md for the deployment wiring.

Domain randomization
--------------------
Step-time DR is applied here:
- Sensor noise: uniform noise on ang_vel, gravity, joint positions, last_action.
- Random kicks: low-probability horizontal impulse to the torso base.
- Action latency: a circular buffer samples one of the last few commanded actions to
  apply to the motors (simulates comm jitter); depth = len(dr.latency_distribution).
Physics DR (friction, kp/kd, mass, inertia, CoM) is handled by
workspace/randomize.py via brax PPO's randomization_fn.
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


# ---------------------------------------------------------------------------
# Circular-buffer helpers (adapted from pupperv3-mjx/utils.py)
# ---------------------------------------------------------------------------

def _buf_push_front(buf: jax.Array, new_val: jax.Array) -> jax.Array:
    """Roll buffer forward and place new_val at index 0 (newest slot)."""
    return jp.roll(buf, shift=1, axis=1).at[:, 0].set(new_val)


def _sample_lagged(rng: jax.Array, buf: jax.Array, new_val: jax.Array, dist: jax.Array):
    """Push new_val and sample one time-slot from `buf` according to `dist`.

    buf shape: (action_dim, buf_len); dist shape: (buf_len,); newest first.
    Returns (sampled_action (action_dim,), updated_buf).
    """
    buf = _buf_push_front(buf, new_val)
    return jax.random.choice(rng, buf, axis=1, p=dist), buf


class PupperLegLiftEnv(PipelineEnv):
    """Raise / hold / lower one commanded Pupper leg while balancing on the others."""

    def __init__(self, config: config_dict.ConfigDict, model_path: Optional[str] = None):
        self._config = config
        path = str(configs.resolve_model_path(model_path))

        sys = mjcf.load(path)
        self._dt = config.ctrl_dt
        sys = sys.tree_replace({"opt.timestep": config.sim_dt})

        # Match the deployment actuator model: position control with fixed kp and
        # damping (same override the providers' env applies for a smoother policy).
        sys = sys.replace(
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(config.position_control_kp),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1]
            .set(-config.position_control_kp)
            .at[:, 2]
            .set(-config.dof_damping),
        )
        sys.mj_model.keyframe("home").qpos[7:] = configs.DEFAULT_POSE
        # Mirror the gain override onto the underlying MjModel so the settling pass
        # below (which uses plain MuJoCo, not MJX) sees the same actuator as training.
        sys.mj_model.actuator_gainprm[:, 0] = config.position_control_kp
        sys.mj_model.actuator_biasprm[:, 1] = -config.position_control_kp
        sys.mj_model.actuator_biasprm[:, 2] = -config.dof_damping

        n_frames = int(round(self._dt / sys.opt.timestep))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self._default_pose = jp.array(configs.DEFAULT_POSE)
        # Spawn ALREADY STANDING rather than from the model's home keyframe, which
        # drops the robot from z=0.28 and lets it bounce. That landing slide moved
        # the torso ~25 mm, which would eat most of the body_drift deadband before
        # the policy did anything, and wasted ~0.4 s of every episode. Settling once
        # here (plain MuJoCo, at construction) also captures the steady-state droop
        # of position control at this kp, so the episode starts in true equilibrium.
        self._init_q = jp.array(self._settled_qpos(sys.mj_model))
        self._lowers = jp.array(configs.JOINT_LOWER_LIMITS)
        self._uppers = jp.array(configs.JOINT_UPPER_LIMITS)
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        f = config.soft_joint_pos_limit_factor
        self._soft_lowers = c - 0.5 * r * f
        self._soft_uppers = c + 0.5 * r * f
        # Per-joint (12,), not a scalar: the hip needs a wide range to actually raise
        # the leg while abduction/knee stay tight. See configs.ACTION_SCALE.
        self._action_scale = jp.array(config.action_scale)
        assert self._action_scale.shape == (12,), "action_scale must be per-joint (12,)"

        # Body / site indices.
        self._torso_idx = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, configs.TORSO_NAME)
        assert self._torso_idx != -1, "torso body not found"
        feet_ids = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in configs.FOOT_SITE_NAMES
        ]
        assert -1 not in feet_ids, "a foot site was not found"
        self._feet_site_id = np.array(feet_ids)
        self._foot_radius = 0.02
        knee_ids = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, b) for b in configs.KNEE_BODY_NAMES
        ]
        assert -1 not in knee_ids, "a knee body was not found"
        self._knee_body_id = np.array(knee_ids)
        self._nv = sys.nv

        # ---- per-command lookup tables (rows: stand, FL, FR, BR, BL) ----
        # There is deliberately no "target lifted pose" table any more. The old one
        # pinned the raised leg to a fixed joint target that fought the (unreachable)
        # height target; height is now shaped directly by a ramp on foot clearance,
        # and the only pose the policy is asked to hold is the home pose on the
        # STANCE legs. See configs.get_config()'s reward_config.
        foot_rows = [-1]        # which foot is up (-1 = none, i.e. "stand")
        stance_mask_rows = [np.ones(12)]  # standing => all 12 joints held at home
        abduction_idx = [0]     # lifted leg's abduction joint (dummy for "stand")
        knee_idx = [0]          # lifted leg's knee joint      (dummy for "stand")
        hip_idx = [0]           # lifted leg's hip joint       (dummy for "stand")
        hip_sign = [0.0]        # direction that hip must rotate to raise the leg
        for leg in configs.COMMAND_STATES[1:]:
            foot_rows.append(configs.FOOT_ROW_BY_LEG[leg])
            ji = configs.LEG_JOINT_INDICES[leg]
            mask = np.ones(12)
            mask[ji] = 0.0      # the raised leg is exempt from home-pose tracking
            stance_mask_rows.append(mask)
            abduction_idx.append(ji[0])
            knee_idx.append(ji[2])
            hip_idx.append(ji[1])
            hip_sign.append(configs.HIP_LIFT_SIGN[leg])
        self._lifted_foot_row = jp.array(foot_rows)                    # (5,)
        self._stance_joint_mask = jp.array(np.stack(stance_mask_rows))  # (5, 12)
        self._lift_abduction_idx = jp.array(abduction_idx)             # (5,)
        self._lift_knee_idx = jp.array(knee_idx)                       # (5,)
        self._lift_hip_idx = jp.array(hip_idx)                         # (5,)
        self._lift_hip_sign = jp.array(hip_sign)                       # (5,)
        self._num_commands = configs.NUM_COMMANDS
        self._knee_radius = configs.KNEE_RADIUS
        self._stand_height = configs.STAND_TORSO_HEIGHT

        # ---- observation sizing ----
        # [ang_vel(3), gravity(3), command_one_hot(5), joint_pos - default(12), last_act(12)]
        self._single_obs_dim = 3 + 3 + self._num_commands + 12 + 12
        self._obs_history = config.observation_history

        # ---- action latency buffer ----
        lat_dist = list(config.dr.latency_distribution)
        self._lat_dist = jp.array(lat_dist)
        self._lat_buf_len = len(lat_dist)

    def _forward_xy(self, pipeline_state: base.State) -> jax.Array:
        """Unit XY projection of the torso's forward axis (its heading)."""
        fwd = math.rotate(jp.array([1.0, 0.0, 0.0]), pipeline_state.x.rot[self._torso_idx - 1])[:2]
        # Well-conditioned while the robot is anywhere near upright: the projection's
        # norm is cos(tilt) >= 0.92 at the terminal tilt, so this never approaches 0.
        return fwd / jp.maximum(jp.linalg.norm(fwd), 1e-6)

    # -------------------------------------------------------------- spawn state
    @staticmethod
    def _settled_qpos(mj_model, settle_seconds: float = 2.0) -> np.ndarray:
        """Simulate the home pose to rest and return the resulting qpos.

        Runs once at construction on the (already gain-overridden) model, holding
        ctrl at DEFAULT_POSE, so episodes begin from a standing equilibrium.
        """
        data = mujoco.MjData(mj_model)
        mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        data.ctrl[:] = configs.DEFAULT_POSE
        for _ in range(int(settle_seconds / mj_model.opt.timestep)):
            mujoco.mj_step(mj_model, data)
        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError("settling the home pose diverged; check the model/gains")
        qpos = data.qpos.copy()
        qpos[:2] = 0.0  # re-center horizontally; only the height/attitude matter
        return qpos

    # ------------------------------------------------------------- commands
    def _sample_command(self, rng: jax.Array) -> jax.Array:
        """Pick a command index: 'stand' with stand_command_prob, else a leg."""
        rng_stand, rng_leg = jax.random.split(rng)
        is_stand = jax.random.uniform(rng_stand, ()) < self._config.stand_command_prob
        leg = jax.random.randint(rng_leg, (), 1, self._num_commands)
        return jp.where(is_stand, 0, leg).astype(jp.int32)

    def _sample_hold(self, rng: jax.Array) -> jax.Array:
        return jax.random.randint(
            rng, (), self._config.command_hold_steps_min, self._config.command_hold_steps_max + 1
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
            # Where the torso started, so the body_drift reward can measure "has it
            # walked/shuffled away from where it was standing" per episode. Recorded
            # per-episode rather than hard-coded because domain randomization changes
            # mass/CoM and therefore where the robot settles.
            "init_xy": pipeline_state.x.pos[self._torso_idx - 1, :2],
            # Which way the robot was facing, for the heading reward. Swinging a leg
            # up applies a yaw reaction torque to the body, and NOTHING else in this
            # reward opposes it: `orientation` is built on cos_tilt, which measures
            # tilt away from vertical and is completely blind to rotation ABOUT the
            # vertical, and body_drift only sees translation. Without this term the
            # robot slowly spins in place -- measured at 40-50 deg over a 12 s
            # showcase on policies that otherwise looked clean.
            "init_forward": self._forward_xy(pipeline_state),
        }
        obs_history = jp.zeros(self._obs_history * self._single_obs_dim)
        obs = self._get_obs(pipeline_state, info, obs_history, obs_rng)
        metrics: Dict[str, Any] = {k: 0.0 for k in self._config.reward_config.scales.keys()}
        metrics["lifted_foot_height"] = 0.0
        metrics["lifted_knee_height"] = 0.0
        metrics["body_drift_dist"] = 0.0
        metrics["torso_z"] = 0.0
        metrics["tilt_deg"] = 0.0
        metrics["yaw_deg"] = 0.0
        return State(pipeline_state, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # ------------------------------------------------------------------- step
    def step(self, state: State, action: jax.Array) -> State:
        state.info["rng"], kick_rng, lat_rng, obs_rng = jax.random.split(state.info["rng"], 4)

        # Action latency: sample which time-slot to actually apply.
        lagged_action, action_buffer = _sample_lagged(
            lat_rng, state.info["action_buffer"], action, self._lat_dist
        )
        state.info["action_buffer"] = action_buffer

        # Random horizontal kick to the torso base (low probability).
        kick_rng1, kick_rng2 = jax.random.split(kick_rng)
        kick_vec = jax.random.uniform(kick_rng1, (2,), minval=-1.0, maxval=1.0) * self._config.dr.kick_vel
        kick_applied = jax.random.bernoulli(kick_rng2, self._config.dr.kick_probability)
        kicked_qd = state.pipeline_state.qd.at[:2].set(
            state.pipeline_state.qd[:2] + kick_vec * kick_applied
        )

        # Per-joint action scale => reachable joint range is DEFAULT_POSE +/- ACTION_SCALE
        # (the policy head is tanh-squashed, so |lagged_action| <= 1).
        motor_targets = jp.clip(self._default_pose + lagged_action * self._action_scale, self._lowers, self._uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state.replace(qd=kicked_qd), motor_targets)

        command = state.info["command"]
        obs = self._get_obs(pipeline_state, state.info, state.obs, obs_rng)

        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]
        x = pipeline_state.x

        foot_z = pipeline_state.site_xpos[self._feet_site_id][:, 2] - self._foot_radius
        contact = foot_z < 1e-3
        # x.pos excludes the world body, hence the -1 (matches self._torso_idx usage below).
        knee_z = pipeline_state.x.pos[self._knee_body_id - 1][:, 2]
        lifted_row = self._lifted_foot_row[command]
        lifted_foot_height = jp.where(lifted_row >= 0, foot_z[jp.maximum(lifted_row, 0)], 0.0)
        lifted_knee_height = jp.where(lifted_row >= 0, knee_z[jp.maximum(lifted_row, 0)], 0.0)
        body_drift_dist = jp.linalg.norm(x.pos[self._torso_idx - 1, :2] - state.info["init_xy"])
        heading_cos = jp.dot(self._forward_xy(pipeline_state), state.info["init_forward"])

        up = jp.array([0.0, 0.0, 1.0])
        cos_tilt = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up)

        # Height of each knee sphere's lowest point above the floor.
        knee_clearance = knee_z - self._knee_radius

        done = cos_tilt < jp.cos(self._config.terminal_body_angle)
        done |= x.pos[self._torso_idx - 1, 2] < self._config.terminal_body_z
        # Resting a knee on the floor to prop up the lift ends the episode outright;
        # the ground_contact reward term is the gradient that steers away before this.
        done |= jp.any(knee_clearance < self._config.terminal_knee_clearance)
        # Likewise for shuffling the whole robot away from where it started, or for
        # spinning away from the heading it started on.
        done |= body_drift_dist > self._config.terminal_body_drift
        done |= heading_cos < jp.cos(self._config.terminal_body_yaw)

        rewards = self._get_reward(
            command, joint_angles, joint_vel, pipeline_state, contact, foot_z,
            knee_clearance, cos_tilt, heading_cos, action, state.info
        )
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # Advance command when the hold window elapses.
        state.info["rng"], cmd_rng, hold_rng = jax.random.split(state.info["rng"], 3)
        switch = state.info["step"] >= state.info["command_switch_step"]
        state.info["command"] = jp.where(switch, self._sample_command(cmd_rng), command)
        state.info["command_switch_step"] = jp.where(
            switch, state.info["step"] + self._sample_hold(hold_rng), state.info["command_switch_step"]
        )

        # Store raw commanded action (not lagged) in last_act — the obs and
        # action_rate penalty both reference what the policy last commanded.
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["step"] = state.info["step"] + 1

        state.metrics.update(rewards)
        state.metrics["lifted_foot_height"] = lifted_foot_height
        state.metrics["lifted_knee_height"] = lifted_knee_height
        state.metrics["body_drift_dist"] = body_drift_dist
        state.metrics["torso_z"] = x.pos[self._torso_idx - 1, 2]
        state.metrics["tilt_deg"] = jp.rad2deg(jp.arccos(jp.clip(cos_tilt, -1.0, 1.0)))
        state.metrics["yaw_deg"] = jp.rad2deg(jp.arccos(jp.clip(heading_cos, -1.0, 1.0)))

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=jp.float32(done))

    # ----------------------------------------------------------------- reward
    def _get_reward(
        self, command, joint_angles, joint_vel, pipeline_state, contact, foot_z,
        knee_clearance, cos_tilt, heading_cos, action, info
    ) -> Dict[str, jax.Array]:
        """Reward: hold the standing pose, and raise the commanded leg as high as it can.

        The shaping is deliberately split so nothing competes with anything else:
        the RAISED leg is driven only by a height ramp (plus a weak in-plane prior),
        while every other term describes "the robot is still standing where it was,
        upright, at full height, on its other three feet".
        """
        cfg = self._config.reward_config
        lifted_row = self._lifted_foot_row[command]
        a_leg_is_up = (lifted_row >= 0).astype(float)

        # -- raise the commanded leg -------------------------------------------
        # Linear ramp in foot clearance, saturating at target_lift_height. Unlike a
        # Gaussian around a fixed target this has a constant positive gradient all
        # the way up, so "higher is better" right up to the cap -- and no incentive
        # to contort past it.
        lifted_height = jp.where(a_leg_is_up > 0, foot_z[jp.maximum(lifted_row, 0)], 0.0)
        lift_height = a_leg_is_up * jp.clip(lifted_height / cfg.target_lift_height, 0.0, 1.0)

        # Dense companion to lift_height, measured on the hip ANGLE rather than the
        # foot's height off the floor. lift_height is identically zero for every
        # configuration in which the foot is still touching down, so on its own it
        # gives no signal at all for the first few degrees of rotation -- and a policy
        # that never leaves the ground never discovers the ramp. Hip angle has no such
        # dead zone: it pays from the first degree and rises continuously to a full
        # lift. Saturates at the same pose lift_height does, so the two agree.
        hip_i = self._lift_hip_idx[command]
        hip_delta = self._lift_hip_sign[command] * (joint_angles[hip_i] - self._default_pose[hip_i])
        lift_progress = a_leg_is_up * jp.clip(hip_delta / cfg.hip_lift_reference, 0.0, 1.0)

        # Weak prior keeping the raised leg's abduction and knee near home, so the
        # leg swings up in its own sagittal plane (driven by the hip) instead of
        # splaying sideways or curling. The hip is intentionally left unconstrained.
        abd_i, knee_i = self._lift_abduction_idx[command], self._lift_knee_idx[command]
        lift_prior_err = (
            jp.square(joint_angles[abd_i] - self._default_pose[abd_i])
            + jp.square(joint_angles[knee_i] - self._default_pose[knee_i])
        )
        lift_pose_prior = a_leg_is_up * jp.exp(-lift_prior_err / cfg.lift_prior_sigma)

        # -- hold the standing pose --------------------------------------------
        # The three planted legs (all four when standing) track the home pose. This
        # is the term that stops the robot from re-arranging its whole body to lift.
        stance_mask_j = self._stance_joint_mask[command]
        stance_err = jp.sum(stance_mask_j * jp.square(joint_angles - self._default_pose))
        stance_pose = jp.exp(-stance_err / cfg.stance_pose_sigma)

        # Feet that should be planted: all except the commanded one.
        rows = jp.arange(4)
        stance_mask = jp.where(a_leg_is_up > 0, rows != lifted_row, jp.ones(4, dtype=bool))
        n_stance = jp.sum(stance_mask.astype(float))
        stance_feet_contact = jp.sum(contact * stance_mask) / jp.maximum(n_stance, 1.0)

        # Upright, at full standing height, and still where it started.
        orientation = jp.exp(-(1.0 - jp.clip(cos_tilt, -1.0, 1.0)) / cfg.orientation_sigma)
        # Same shape as `orientation`, but about the vertical axis instead of away
        # from it -- keeps the robot pointing the way it started rather than slowly
        # spinning as the leg swings react through the body.
        heading = jp.exp(-(1.0 - jp.clip(heading_cos, -1.0, 1.0)) / cfg.heading_sigma)
        torso_height = pipeline_state.x.pos[self._torso_idx - 1, 2]
        torso_height_rew = jp.exp(-jp.square(torso_height - self._stand_height) / cfg.torso_height_sigma)

        # Deadbanded: free to shift within allowed_body_drift (a front-leg lift
        # genuinely requires ~12 mm of CoM shift to stay inside the support
        # triangle), penalized for anything beyond that.
        drift = jp.linalg.norm(pipeline_state.x.pos[self._torso_idx - 1, :2] - info["init_xy"])
        drift_excess = jp.clip(drift - cfg.allowed_body_drift, 0.0, None)
        body_drift = jp.exp(-jp.square(drift_excess) / cfg.body_drift_sigma)

        # -- penalties ----------------------------------------------------------
        # Any knee approaching the floor, normalized to [0, 1] per knee, summed over
        # all four. Directly targets "props a limb on the ground to lift the other".
        ground_contact = jp.sum(
            jp.clip((cfg.knee_ground_margin - knee_clearance) / cfg.knee_ground_margin, 0.0, 1.0)
        )

        action_rate = jp.sum(jp.square(action - info["last_act"]))
        torques = jp.sum(jp.square(pipeline_state.qfrc_actuator[6:]))

        # Per-joint hinge on how close each actuator is to its torque ceiling: 0 below
        # torque_soft_fraction of the limit, ramping to 1 at the limit. Keeps the lift
        # inside the envelope a REAL motor can actually deliver, which the plain
        # sum-of-squares `torques` term above is far too weak to do.
        soft = cfg.torque_soft_fraction * cfg.torque_limit_nm
        tau = jp.abs(pipeline_state.qfrc_actuator[6:])
        torque_limit = jp.sum(
            jp.clip((tau - soft) / (cfg.torque_limit_nm - soft), 0.0, 1.0)
        )
        dof_acc = jp.sum(jp.square((joint_vel - info["last_vel"]) / self._dt))
        out_lo = -jp.clip(joint_angles - self._soft_lowers, None, 0.0)
        out_hi = jp.clip(joint_angles - self._soft_uppers, 0.0, None)
        dof_pos_limits = jp.sum(out_lo + out_hi)

        return {
            "lift_progress": lift_progress,
            "lift_height": lift_height,
            "lift_pose_prior": lift_pose_prior,
            "stance_pose": stance_pose,
            "stance_feet_contact": stance_feet_contact,
            "orientation": orientation,
            "heading": heading,
            "torso_height": torso_height_rew,
            "body_drift": body_drift,
            "ground_contact": ground_contact,
            "action_rate": action_rate,
            "torque_limit": torque_limit,
            "torques": torques,
            "dof_acc": dof_acc,
            "dof_pos_limits": dof_pos_limits,
        }

    # -------------------------------------------------------------------- obs
    def _get_obs(
        self, pipeline_state: base.State, info: dict[str, Any], obs_history: jax.Array, rng: jax.Array
    ) -> jax.Array:
        dr = self._config.dr
        ang_rng, grav_rng, jpos_rng, lact_rng = jax.random.split(rng, 4)

        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        ang_vel = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        ang_vel = ang_vel + jax.random.uniform(ang_rng, (3,), minval=-1.0, maxval=1.0) * dr.angular_velocity_noise

        gravity = math.rotate(jp.array([0.0, 0.0, -1.0]), inv_torso_rot)
        gravity = gravity + jax.random.uniform(grav_rng, (3,), minval=-1.0, maxval=1.0) * dr.gravity_noise

        command_one_hot = jax.nn.one_hot(info["command"], self._num_commands)

        jpos = pipeline_state.q[7:] - self._default_pose
        jpos = jpos + jax.random.uniform(jpos_rng, (12,), minval=-1.0, maxval=1.0) * dr.motor_angle_noise

        last_act = info["last_act"]
        last_act = last_act + jax.random.uniform(lact_rng, (12,), minval=-1.0, maxval=1.0) * dr.last_action_noise

        obs = jp.concatenate([
            ang_vel,           # 3
            gravity,           # 3
            command_one_hot,   # 5
            jpos,              # 12
            last_act,          # 12
        ])
        obs = jp.clip(obs, -100.0, 100.0)
        # newest observation at the front
        return jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

    def render(
        self,
        trajectory: List[base.State],
        camera: Optional[str] = None,
        height: int = 480,
        width: int = 640,
    ) -> Sequence[np.ndarray]:
        return super().render(trajectory, camera=camera or "tracking_cam", height=height, width=width)
