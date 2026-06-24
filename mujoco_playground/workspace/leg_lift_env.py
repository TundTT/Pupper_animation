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

        n_frames = int(round(self._dt / sys.opt.timestep))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self._default_pose = jp.array(configs.DEFAULT_POSE)
        self._init_q = jp.array(sys.mj_model.keyframe("home").qpos)
        self._lowers = jp.array(configs.JOINT_LOWER_LIMITS)
        self._uppers = jp.array(configs.JOINT_UPPER_LIMITS)
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        f = config.soft_joint_pos_limit_factor
        self._soft_lowers = c - 0.5 * r * f
        self._soft_uppers = c + 0.5 * r * f
        self._action_scale = config.action_scale

        # Body / site indices.
        self._torso_idx = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, configs.TORSO_NAME)
        assert self._torso_idx != -1, "torso body not found"
        feet_ids = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in configs.FOOT_SITE_NAMES
        ]
        assert -1 not in feet_ids, "a foot site was not found"
        self._feet_site_id = np.array(feet_ids)
        self._foot_radius = 0.02
        self._nv = sys.nv

        # ---- command -> target pose table (rows: stand, FL, FR, BR, BL) ----
        target_rows = [configs.DEFAULT_POSE]  # stand
        foot_rows = [-1]  # which foot is up (-1 = none)
        for leg in configs.COMMAND_STATES[1:]:
            target_rows.append(configs.lifted_pose_for(leg))
            foot_rows.append(configs.FOOT_ROW_BY_LEG[leg])
        self._target_table = jp.array(np.stack(target_rows))      # (5, 12)
        self._lifted_foot_row = jp.array(foot_rows)               # (5,)
        self._num_commands = configs.NUM_COMMANDS

        # ---- observation sizing ----
        # [ang_vel(3), gravity(3), command_one_hot(5), joint_pos - default(12), last_act(12)]
        self._single_obs_dim = 3 + 3 + self._num_commands + 12 + 12
        self._obs_history = config.observation_history

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
        rng, cmd_rng, hold_rng = jax.random.split(rng, 3)
        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        info = {
            "rng": rng,
            "step": 0,
            "command": self._sample_command(cmd_rng),
            "command_switch_step": self._sample_hold(hold_rng),
            "last_act": jp.zeros(12),
            "last_vel": jp.zeros(12),
        }
        obs_history = jp.zeros(self._obs_history * self._single_obs_dim)
        obs = self._get_obs(pipeline_state, info, obs_history)
        metrics: Dict[str, Any] = {k: 0.0 for k in self._config.reward_config.scales.keys()}
        metrics["lifted_foot_height"] = 0.0
        return State(pipeline_state, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # ------------------------------------------------------------------- step
    def step(self, state: State, action: jax.Array) -> State:
        # Position-target action, identical scheme to the deployed neural_controller.
        motor_targets = jp.clip(self._default_pose + action * self._action_scale, self._lowers, self._uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)

        command = state.info["command"]
        obs = self._get_obs(pipeline_state, state.info, state.obs)

        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]
        x = pipeline_state.x

        foot_z = pipeline_state.site_xpos[self._feet_site_id][:, 2] - self._foot_radius
        contact = foot_z < 1e-3
        lifted_row = self._lifted_foot_row[command]
        lifted_foot_height = jp.where(lifted_row >= 0, foot_z[jp.maximum(lifted_row, 0)], 0.0)

        up = jp.array([0.0, 0.0, 1.0])
        cos_tilt = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up)

        done = cos_tilt < jp.cos(self._config.terminal_body_angle)
        done |= x.pos[self._torso_idx - 1, 2] < self._config.terminal_body_z

        rewards = self._get_reward(
            command, joint_angles, joint_vel, pipeline_state, contact, foot_z, cos_tilt, action, state.info
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

        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["step"] = state.info["step"] + 1

        state.metrics.update(rewards)
        state.metrics["lifted_foot_height"] = lifted_foot_height

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=jp.float32(done))

    # ----------------------------------------------------------------- reward
    def _get_reward(
        self, command, joint_angles, joint_vel, pipeline_state, contact, foot_z, cos_tilt, action, info
    ) -> Dict[str, jax.Array]:
        cfg = self._config.reward_config
        target_pose = self._target_table[command]
        lifted_row = self._lifted_foot_row[command]
        a_leg_is_up = lifted_row >= 0

        pose_err = jp.sum(jp.square(joint_angles - target_pose))
        tracking_pose = jp.exp(-pose_err / cfg.tracking_sigma)

        # Commanded foot reaches target clearance (only when a leg is commanded up).
        lifted_height = jp.where(a_leg_is_up, foot_z[jp.maximum(lifted_row, 0)], 0.0)
        clearance_err = jp.square(lifted_height - cfg.target_foot_height)
        foot_clearance = a_leg_is_up.astype(float) * jp.exp(-clearance_err / 0.0025)

        # Feet that should be planted: all except the commanded one.
        rows = jp.arange(4)
        stance_mask = jp.where(a_leg_is_up, rows != lifted_row, jp.ones(4, dtype=bool))
        n_stance = jp.sum(stance_mask.astype(float))
        stance_feet_contact = jp.sum(contact * stance_mask) / jp.maximum(n_stance, 1.0)

        orientation = jp.clip(cos_tilt, 0.0, 1.0)
        torso_height = pipeline_state.x.pos[self._torso_idx - 1, 2]
        torso_height_rew = jp.exp(-jp.square(torso_height - 0.14) / 0.01)

        action_rate = jp.sum(jp.square(action - info["last_act"]))
        torques = jp.sum(jp.square(pipeline_state.qfrc_actuator[6:]))
        dof_acc = jp.sum(jp.square((joint_vel - info["last_vel"]) / self._dt))
        out_lo = -jp.clip(joint_angles - self._soft_lowers, None, 0.0)
        out_hi = jp.clip(joint_angles - self._soft_uppers, 0.0, None)
        dof_pos_limits = jp.sum(out_lo + out_hi)

        return {
            "tracking_pose": tracking_pose,
            "foot_clearance": foot_clearance,
            "stance_feet_contact": stance_feet_contact,
            "orientation": orientation,
            "torso_height": torso_height_rew,
            "action_rate": action_rate,
            "torques": torques,
            "dof_acc": dof_acc,
            "dof_pos_limits": dof_pos_limits,
        }

    # -------------------------------------------------------------------- obs
    def _get_obs(self, pipeline_state: base.State, info: dict[str, Any], obs_history: jax.Array) -> jax.Array:
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        ang_vel = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        gravity = math.rotate(jp.array([0.0, 0.0, -1.0]), inv_torso_rot)
        command_one_hot = jax.nn.one_hot(info["command"], self._num_commands)

        obs = jp.concatenate([
            ang_vel,                                    # 3
            gravity,                                    # 3
            command_one_hot,                            # 5
            pipeline_state.q[7:] - self._default_pose,  # 12
            info["last_act"],                           # 12
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
