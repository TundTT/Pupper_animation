"""Deployment-aligned walking environment with literature-standard gait rewards.

Builds on `pupperv3_mjx.environment.PupperV3Env` (the Stanford/Nathan Kau reference
pipeline) and keeps its observation layout, network contract and reward primitives,
but forks `reset()`/`step()` to fix three classes of problem found while trying to
get a normal-looking gait out of the plain reference config:

1. Deployment alignment (see origin/robot-info `robot_info/POLICY_INTERFACE.md` and
   `robot_info/robot_contract.json`):
     - Actions are offsets from the robot's *homed* pose (`leg_position` profile
       `homed_position`), which is what the deployed controller adds them to, instead
       of the MJCF's own ctrl=0 equilibrium (those differ by ~0.2 rad on abduction).
     - Per-joint `action_scale` (12-vector). Joint 2 (abduction) only has 0.14 rad of
       hardware headroom from home, while the reference's uniform 0.75 rad let the
       policy splay the legs ~56 mm laterally -- a major source of unnatural gait.
     - Motor targets are clipped the way the controller clips them: against absolute
       hardware position limits, after adding the home pose.
     - Observation history is initialized by tiling the first measured frame, which is
       what the controller does on activation (the reference filled it with zeros).

2. Gait quality. The reference reward has no term that can reward *starting* to lift a
   foot (`reward_feet_air_time` is identically zero for a foot that never leaves the
   ground, since `first_contact` requires prior air time), so it settles into dragging
   or standing. Added, all reward-only (no observation/contract change):
     - `foot_clearance`: penalizes horizontal foot motion at the wrong height, the
       standard dense anti-dragging/clearance term in the legged-RL literature.
     - `stance_cap`: penalizes continuous stance beyond a target, setting a gait
       frequency floor. Active from the first step, unlike `feet_air_time`.
     - `touchdown_impact`: penalizes downward foot speed at contact, so clearance
       doesn't turn into aggressive floor tapping.
     - `knee_clearance`: rewards lifting the knee ("elbow") during swing. Foot-tip
       clearance alone is satisfiable by rotating joint 3, which does not move the
       knee at all, i.e. pawing rather than walking.
     - `feet_air_time` is replaced by a *bounded* swing-duration credit: the
       reference form is unbounded above, and at a gait-shaping weight it pays for
       flight time and the policy launches itself.
     - `flight`: penalizes having no foot in contact.

3. Command coverage. The reference resamples one command per 250 steps and never
   resamples on early termination, and it samples vx/vy/yaw jointly so pure-axis
   motion (especially reverse) is rare. This samples an explicit mix of
   stand/forward/backward/lateral/turn/mixed modes, resamples on `done`, and
   resamples more often.
"""
from typing import Any

import jax
import numpy as np
from brax import base, math
from brax.envs.base import State
from jax import numpy as jp

from pupperv3_mjx import domain_randomization, environment, rewards, utils

# `leg_position` hardware profile, origin/robot-info robot_info/robot_contract.json.
CANONICAL_JOINTS = (
    'leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3',
    'leg_front_l_1', 'leg_front_l_2', 'leg_front_l_3',
    'leg_back_r_1', 'leg_back_r_2', 'leg_back_r_3',
    'leg_back_l_1', 'leg_back_l_2', 'leg_back_l_3',
)
HOMED_POSITION = np.array([
    0.990537, -0.179972, -1.045141,
    -0.990539, 0.179973, 0.950836,
    0.990537, -0.179972, -1.045141,
    -0.990539, 0.179973, 0.950836,
])
HARDWARE_POSITION_MIN = np.array([
    -1.12, -0.32, -2.69, -2.41, -3.04, -0.61,
    -1.12, -0.32, -2.69, -2.41, -3.04, -0.61,
])
HARDWARE_POSITION_MAX = np.array([
    2.41, 3.04, 0.61, 1.12, 0.32, 2.69,
    2.41, 3.04, 0.61, 1.12, 0.32, 2.69,
])
# Joint 1 sweeps the foot fore/aft (~27 mm per 0.2 rad) and joint 3 adds to that;
# joint 2 is abduction (~15 mm lateral per 0.2 rad) and is the joint with only
# 0.14 rad of headroom from home, so it gets the least authority.
ACTION_SCALE = np.array([0.8, 0.15, 1.0] * 4)


def foot_velocities(pipeline_state: base.State, feet_site_id, lower_leg_body_id):
    """World-frame velocity of each foot site (same construction the reference
    `reward_foot_slip` uses, factored out so slip/clearance/impact share it)."""
    pos = pipeline_state.site_xpos[feet_site_id]
    feet_offset = pos - pipeline_state.xpos[lower_leg_body_id]
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = lower_leg_body_id - 1  # world body is dropped in brax
    return offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel


def reward_foot_clearance(foot_height, foot_vel, target_clearance):
    """Penalize horizontal foot motion at the wrong height.

    sum_f clip((h_target - h_f)/h_target, 0, 1)^2 * ||v_f_xy||: a foot sliding along
    the floor pays ~1.0 per m/s of drag, a foot swinging at or above the target height
    pays nothing, and a planted (near-zero velocity) foot pays nothing, so this does
    not tax normal stance. This is the dense term that gives a gradient toward lifting
    from a standing start, which the reference `feet_air_time` structurally cannot.

    One-sided on purpose: only a foot *below* the target is penalized. A symmetric
    version also taxes lifting higher than the target, which fights the "high
    clearance" goal, and since the weight multiplies foot speed it would tax fast
    swing (the very motion locomotion needs) rather than just dragging. Normalized by
    the target and bounded so the weight is interpretable in units of the tracking
    reward and cannot push the (zero-clipped) step reward into a dead-gradient region.
    """
    speed = jp.sqrt(jp.sum(jp.square(foot_vel[:, :2]), axis=-1) + 1e-9)
    shortfall = jp.clip((target_clearance - foot_height) / target_clearance, 0.0, 1.0)
    return jp.sum(jp.square(shortfall) * speed)


def reward_tracking_lin_vel_scaled(commands, x, xd, variance_base, speed_gain,
                                   yaw_gain, variance_cap):
    """Velocity tracking whose tolerance scales with the commanded speed.

    The reference uses exp(-err^2 / 0.25), tuned for ~+-1 m/s commands where standing
    still under a full-speed command scores ~0.02. At this project's +-0.35 m/s
    envelope that same sigma pays a motionless robot 0.61 of the maximum, and with
    `tracking_ang_vel` and `tracking_orientation` also trivially satisfied by standing,
    ~79% of all available reward is obtainable by doing nothing -- which is what four
    successive training waves converged to. Shrinking sigma instead flattens the
    gradient and the policy never discovers moving at all.

    Scaling the variance with the command (this project's own earlier, hardware-tested
    walk pipeline did the same) keeps a usable gradient at every speed while making
    "do nothing" genuinely unrewarding: standing under a 0.35 m/s command drops to 0.17.
    """
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    variance = jp.minimum(
        variance_cap,
        variance_base + speed_gain * jp.sum(jp.square(commands[:2]))
        + yaw_gain * jp.square(commands[2]))
    return jp.exp(-error / variance)


def reward_tracking_ang_vel_scaled(commands, x, xd, variance_base, gain, variance_cap):
    """Yaw-rate tracking with the same command-scaled tolerance as the linear term."""
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    error = jp.square(commands[2] - base_ang_vel[2])
    variance = jp.minimum(variance_cap, variance_base + gain * jp.square(commands[2]))
    return jp.exp(-error / variance)


def reward_knee_clearance(knee_height, airborne, target_height):
    """Penalize a swinging leg whose knee ("elbow") stays low.

    Foot-tip clearance alone can be satisfied by rotating joint 3, which moves the
    foot site but leaves the knee body height *exactly* unchanged (measured: joint 3
    over -0.6..+0.6 rad moves the foot tip 6.8 -> 16.1 mm and the knee 58.4 -> 58.4 mm).
    That is the pawing/tapping motion rather than a walking stride. The knee can only
    rise by rotating joint 1 at the hip (58.4 -> 74.7 mm at +0.6 rad, and it rises for
    both swing directions since home sits at a local minimum), which is also what
    produces real stride length, so rewarding knee lift buys both.

    Only applied to feet that are actually airborne: a stance leg must keep its knee
    low to hold the robot up.
    """
    shortfall = jp.clip((target_height - knee_height) / target_height, 0.0, 1.0)
    return jp.sum(jp.square(shortfall) * airborne)


def reward_swing_duration(air_time, first_contact, commands, target_swing_s):
    """Reward swings that last about `target_swing_s`, bounded on both sides.

    The reference `reward_feet_air_time` is (air_time - minimum) at touchdown with no
    ceiling. That is harmless at its original 0.02 weight, but at a weight large
    enough to actually shape the gait it pays for flight time, and the policy learns
    to launch itself (measured: 95% falls, 11-19% of samples with no foot down).
    Clipping the credit means a proper swing is rewarded, a too-short shuffle is
    penalized, and hanging in the air buys nothing extra.
    """
    credit = jp.clip(air_time - target_swing_s, -target_swing_s, 0.5 * target_swing_s)
    moving = math.normalize(commands[:3])[1] > 0.05
    return jp.sum(credit * first_contact) * moving


def reward_tip_vertical(alignment, contact, allowance_cos, scale_cos):
    """Penalize a weight-bearing capsule that is not close to vertical.

    `alignment` is cos(angle between the elbow->tip capsule axis and straight down),
    so 1.0 is a perfectly vertical leg loading through the capsule's rounded tip.
    Measured on this model, the home stance is already only 6.3 deg off vertical, but
    joint 3 alone can plant the foot at 21 deg (0.4 rad) through 49 deg (0.8 rad),
    which loads the *side* of the capsule instead of the tip.

    Zero inside `allowance`, rising quadratically to 1.0 at `scale` and bounded after,
    so the policy may tilt the leg when it genuinely needs to for balance but pays for
    doing it habitually. Contact-gated: a swinging leg may point however it likes.
    """
    excess = jp.clip((allowance_cos - alignment) / (allowance_cos - scale_cos), 0.0, 2.0)
    return jp.mean(contact * jp.square(excess))


def reward_flight(contact):
    """Penalize samples with no foot on the ground.

    A walk (and a trot at these speeds) should always have support. Without this the
    clearance and swing rewards can be traded for a hopping/ballistic gait.
    """
    return jp.float32(jp.sum(contact) == 0)


def reward_stance_cap(stance_time, max_stance_s):
    """Penalize a foot planted longer than one gait period allows.

    Normalized by the cap and bounded, for the same reason as foot_clearance.
    """
    excess = jp.clip((stance_time - max_stance_s) / max_stance_s, 0.0, 2.0)
    return jp.sum(jp.square(excess))


def reward_touchdown_impact(first_contact, previous_foot_vel_z, allowance):
    """Penalize excess downward foot speed at touchdown, so the clearance reward
    does not produce hard floor tapping. Uses the pre-contact sample because the
    solver has already removed the downward velocity by the contact step."""
    excess = jp.clip(-previous_foot_vel_z - allowance, 0.0, None)
    return jp.sum(first_contact * jp.square(excess))


class PupperGaitEnv(environment.PupperV3Env):
    """See module docstring. `reset`/`step` are forks of the upstream methods."""

    def __init__(
        self,
        *args,
        home_pose=HOMED_POSITION,
        policy_lower_limits=HARDWARE_POSITION_MIN,
        policy_upper_limits=HARDWARE_POSITION_MAX,
        target_clearance_m: float = 0.025,
        target_knee_height_m: float = 0.070,
        variance_base: float = 0.0025,
        speed_gain: float = 0.55,
        yaw_gain: float = 0.27,
        variance_cap: float = 0.07,
        yaw_variance_base: float = 0.05,
        yaw_variance_gain: float = 0.8,
        yaw_variance_cap: float = 0.25,
        tip_allowance_deg: float = 15.0,
        tip_scale_deg: float = 45.0,
        target_swing_s: float = 0.18,
        max_stance_s: float = 0.35,
        touchdown_speed_allowance: float = 0.15,
        command_resample_steps: int = 100,
        stand_probability: float = 0.1,
        **kwargs,
    ):
        # Upstream adds `default_pose` to every action and overwrites the home
        # keyframe with it; we drive ctrl ourselves, so neutralize that path and
        # keep `_default_pose` purely as the observation/export reference frame.
        kwargs['default_pose'] = jp.array(home_pose)
        super().__init__(*args, **kwargs)

        mj_model = self.sys.mj_model
        # ctrl=0 in this MJCF holds biasprm[0]/kp, not the robot's homed pose, so
        # actions must be offset by the difference to land on the deployed frame.
        kp = mj_model.actuator_gainprm[:, 0]
        ctrl_zero_pose = mj_model.actuator_biasprm[:, 0] / kp
        self._ctrl_offset = jp.array(np.asarray(home_pose) - ctrl_zero_pose)
        # Capsule axis (elbow -> tip) in each lower-leg body frame, for tip verticality.
        import mujoco as _mj
        axes = []
        for gid, sid in zip(self._foot_geom_ids_for_axis(mj_model), self._feet_site_id):
            rot = np.zeros(9)
            _mj.mju_quat2Mat(rot, mj_model.geom_quat[gid])
            local_z = rot.reshape(3, 3)[:, 2]
            to_site = mj_model.site_pos[sid] - mj_model.geom_pos[gid]
            axes.append(local_z * (1.0 if np.dot(to_site, local_z) >= 0 else -1.0))
        self._capsule_axis_local = jp.array(np.array(axes))
        self._home_pose = jp.array(home_pose)
        self._action_scale_vec = jp.broadcast_to(jp.asarray(self._action_scale), (12,))
        self._policy_lower = jp.array(policy_lower_limits)
        self._policy_upper = jp.array(policy_upper_limits)

        self._target_clearance_m = target_clearance_m
        self._target_knee_height_m = target_knee_height_m
        self._variance_base = variance_base
        self._speed_gain = speed_gain
        self._yaw_gain = yaw_gain
        self._variance_cap = variance_cap
        self._yaw_variance_base = yaw_variance_base
        self._yaw_variance_gain = yaw_variance_gain
        self._yaw_variance_cap = yaw_variance_cap
        self._tip_allowance_cos = float(np.cos(np.radians(tip_allowance_deg)))
        self._tip_scale_cos = float(np.cos(np.radians(tip_scale_deg)))
        self._target_swing_s = target_swing_s
        self._max_stance_s = max_stance_s
        self._touchdown_speed_allowance = touchdown_speed_allowance
        self._command_resample_steps = command_resample_steps
        self._stand_probability = stand_probability

        # Settled qpos at the deployed home pose, so episodes do not start with a
        # transient that the reference's raw keyframe placement would introduce.
        self._init_q = jp.array(self._settled_home_qpos(mj_model))

    @staticmethod
    def _foot_geom_ids_for_axis(mj_model):
        legs = ('front_r', 'front_l', 'back_r', 'back_l')
        return [mj_model.geom(f'leg_{leg}_3_foot_collision').id for leg in legs]

    def _settled_home_qpos(self, mj_model):
        import mujoco

        data = mujoco.MjData(mj_model)
        data.qpos[:] = mj_model.key('home').qpos
        data.qpos[2] += 0.02
        data.ctrl[:] = np.asarray(self._ctrl_offset)
        for _ in range(int(2.0 / mj_model.opt.timestep)):
            mujoco.mj_step(mj_model, data)
        if not np.all(np.isfinite(data.qpos)) or np.linalg.norm(data.qvel) > 0.05:
            raise ValueError('Home pose did not settle; check home_pose/action frame.')
        return data.qpos.copy()

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Explicit mode mix so every direction (especially reverse) is trained.

        The reference sampled vx/vy/yaw jointly from uniform ranges, which makes
        pure-axis commands rare and leaves reverse walking undertrained.
        """
        keys = jax.random.split(rng, 7)
        vx = jax.random.uniform(
            keys[0], (), minval=self._linear_velocity_x_range[0],
            maxval=self._linear_velocity_x_range[1])
        vy = jax.random.uniform(
            keys[1], (), minval=self._linear_velocity_y_range[0],
            maxval=self._linear_velocity_y_range[1])
        wz = jax.random.uniform(
            keys[2], (), minval=self._angular_velocity_range[0],
            maxval=self._angular_velocity_range[1])
        # Signed magnitudes for pure-axis modes, so forward and reverse get the
        # same speed distribution instead of reverse only appearing inside mixes.
        speed = jax.random.uniform(keys[3], (), minval=0.12,
                                   maxval=self._linear_velocity_x_range[1])
        sign = jp.where(jax.random.bernoulli(keys[4]), 1.0, -1.0)
        mode = jax.random.randint(keys[5], (), 0, 5)
        zero = jp.zeros(())
        candidates = jp.stack([
            jp.array([sign * speed, zero, zero]),                  # forward/back
            jp.array([zero, sign * jp.minimum(speed, self._linear_velocity_y_range[1]), zero]),  # lateral
            jp.array([zero, zero, sign * jp.maximum(jp.abs(wz), 0.4)]),  # turn in place
            jp.array([sign * speed, zero, wz]),                    # arc
            jp.array([vx, vy, wz]),                                # mixed
        ])
        command = candidates[mode]
        # keys[6], not keys[0]: reusing the vx key made "stand" fire exactly when vx
        # fell in the lowest decile of its range, i.e. it deleted the fastest
        # reverse commands from training.
        stand = jax.random.uniform(keys[6], ()) < self._stand_probability
        return jp.where(stand, jp.zeros(3), command)

    def reset(self, rng: jax.Array) -> State:
        rng, cmd_key, orient_key, pos_key = jax.random.split(rng, 4)
        init_q = domain_randomization.randomize_qpos(
            self._init_q, self._start_position_config, rng=pos_key)
        pipeline_state = self.pipeline_init(init_q, jp.zeros(self._nv))

        state_info = {
            'rng': rng,
            'last_act': jp.zeros(12),
            'action_buffer': self.initial_action_buffer(),
            'imu_buffer': self.initial_imu_buffer(),
            'last_vel': jp.zeros(12),
            'command': self.sample_command(cmd_key),
            'last_contact': jp.zeros(4, dtype=bool),
            'feet_air_time': jp.zeros(4),
            'stance_time': jp.zeros(4),
            'last_foot_vel_z': jp.zeros(4),
            'rewards': {k: 0.0 for k in self._reward_config.rewards.scales.keys()},
            'kick': jp.zeros(2),
            'step': 0,
            'desired_world_z_in_body_frame': self.sample_body_orientation(orient_key),
        }

        # The deployed controller tiles the first measured frame across history on
        # activation; filling with zeros would train on frames that never occur.
        obs = self._get_obs(pipeline_state, state_info, jp.zeros(self._observation_history * self.observation_dim))
        obs = jp.tile(obs[: self.observation_dim], self._observation_history)

        metrics = {'total_dist': 0.0}
        for k in state_info['rewards']:
            metrics[k] = 0.0
        return State(pipeline_state, obs, jp.zeros(()), jp.zeros(()), metrics, state_info)

    def step(self, state: State, action: jax.Array) -> State:
        state.info['rng'], cmd_rng, kick_rng, bern_rng, latency_rng = jax.random.split(
            state.info['rng'], 5)

        kick = jax.random.uniform(kick_rng, (2,), minval=-1.0, maxval=1.0) * self._kick_vel
        kick *= jax.random.bernoulli(bern_rng, p=self._kick_probability, shape=(1,))
        qvel = state.pipeline_state.qvel
        qvel = qvel.at[:2].set(kick + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})

        lagged_action, state.info['action_buffer'] = utils.sample_lagged_value(
            latency_rng, state.info['action_buffer'], action, self._latency_distribution)

        # Mirror the deployed controller: target = clip(home + out * scale, limits).
        # ctrl is an offset from this MJCF's ctrl=0 pose, hence _ctrl_offset.
        absolute_target = self._home_pose + lagged_action * self._action_scale_vec
        absolute_target = jp.clip(absolute_target, self._policy_lower, self._policy_upper)
        motor_targets = absolute_target - (self._home_pose - self._ctrl_offset)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        foot_pos = pipeline_state.site_xpos[self._feet_site_id]
        foot_height = foot_pos[:, 2] - self._foot_radius
        foot_vel = foot_velocities(pipeline_state, self._feet_site_id, self._lower_leg_body_id)
        knee_height = pipeline_state.xpos[self._lower_leg_body_id][:, 2]
        capsule_axis_world = jax.vmap(math.rotate)(
            self._capsule_axis_local, x.rot[self._lower_leg_body_id - 1])
        tip_alignment = -capsule_axis_world[:, 2]  # dot with straight down
        contact = foot_height < 1e-3
        contact_filt_mm = contact | state.info['last_contact']
        contact_filt_cm = (foot_height < 3e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        state.info['feet_air_time'] += self.dt
        stance_time = jp.where(contact_filt_mm, state.info['stance_time'] + self.dt, 0.0)

        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < np.cos(self._terminal_body_angle)
        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < self._terminal_body_z

        moving = jp.linalg.norm(state.info['command'][:3]) > 0.05
        rewards_dict = {
            'tracking_lin_vel': reward_tracking_lin_vel_scaled(
                state.info['command'], x, xd, self._variance_base, self._speed_gain,
                self._yaw_gain, self._variance_cap),
            'tracking_ang_vel': reward_tracking_ang_vel_scaled(
                state.info['command'], x, xd, self._yaw_variance_base, self._yaw_variance_gain,
                self._yaw_variance_cap),
            'tracking_orientation': rewards.reward_tracking_orientation(
                state.info['desired_world_z_in_body_frame'], x,
                tracking_sigma=self._reward_config.rewards.tracking_sigma),
            'lin_vel_z': rewards.reward_lin_vel_z(xd),
            'ang_vel_xy': rewards.reward_ang_vel_xy(xd),
            'orientation': rewards.reward_orientation(x),
            'torques': rewards.reward_torques(pipeline_state.qfrc_actuator),
            'joint_acceleration': rewards.reward_joint_acceleration(
                joint_vel, state.info['last_vel'], dt=self._dt),
            'mechanical_work': rewards.reward_mechanical_work(
                pipeline_state.qfrc_actuator[6:], pipeline_state.qvel[6:]),
            'action_rate': rewards.reward_action_rate(action, state.info['last_act']),
            'stand_still': rewards.reward_stand_still(
                state.info['command'], joint_angles, self._default_pose, 0.1),
            'stand_still_joint_velocity': rewards.reward_stand_still(
                state.info['command'], joint_vel, jp.zeros(12), self._stand_still_command_threshold),
            'abduction_angle': rewards.reward_abduction_angle(
                joint_angles, desired_abduction_angles=self._desired_abduction_angles),
            'feet_air_time': reward_swing_duration(
                state.info['feet_air_time'], first_contact, state.info['command'],
                self._target_swing_s),
            'foot_slip': rewards.reward_foot_slip(
                pipeline_state, contact_filt_cm, feet_site_id=self._feet_site_id,
                lower_leg_body_id=self._lower_leg_body_id),
            'termination': rewards.reward_termination(
                done, state.info['step'], step_threshold=self._early_termination_step_threshold),
            'knee_collision': rewards.reward_geom_collision(pipeline_state, self._upper_leg_geom_ids),
            'body_collision': rewards.reward_geom_collision(pipeline_state, self._torso_geom_ids),
            'foot_clearance': reward_foot_clearance(
                foot_height, foot_vel, self._target_clearance_m) * moving,
            'stance_cap': reward_stance_cap(stance_time, self._max_stance_s) * moving,
            'flight': reward_flight(contact_filt_mm),
            'tip_vertical': reward_tip_vertical(
                tip_alignment, contact_filt_mm, self._tip_allowance_cos, self._tip_scale_cos),
            'knee_clearance': reward_knee_clearance(
                knee_height, ~contact_filt_mm, self._target_knee_height_m) * moving,
            'touchdown_impact': reward_touchdown_impact(
                first_contact, state.info['last_foot_vel_z'], self._touchdown_speed_allowance),
        }
        rewards_dict = {k: v * self._reward_config.rewards.scales[k] for k, v in rewards_dict.items()}
        reward = jp.clip(sum(rewards_dict.values()) * self.dt, 0.0, 10000.0)

        state.info['kick'] = kick
        state.info['last_act'] = action
        state.info['last_vel'] = joint_vel
        state.info['last_foot_vel_z'] = foot_vel[:, 2]
        state.info['feet_air_time'] *= ~contact_filt_mm
        state.info['stance_time'] = stance_time
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards_dict
        state.info['step'] += 1

        # Resample on interval AND on termination. The reference only resampled on
        # the interval, so an env that kept terminating early retried one command.
        resample = (state.info['step'] > self._command_resample_steps) | (done > 0)
        state.info['command'] = jp.where(resample, self.sample_command(cmd_rng), state.info['command'])
        state.info['desired_world_z_in_body_frame'] = jp.where(
            resample, self.sample_body_orientation(cmd_rng),
            state.info['desired_world_z_in_body_frame'])
        state.info['step'] = jp.where(resample, 0, state.info['step'])
        # Episode bookkeeping must not leak across an auto-reset boundary.
        state.info['feet_air_time'] = jp.where(done > 0, jp.zeros(4), state.info['feet_air_time'])
        state.info['stance_time'] = jp.where(done > 0, jp.zeros(4), state.info['stance_time'])

        state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info['rewards'])

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=jp.float32(done))
