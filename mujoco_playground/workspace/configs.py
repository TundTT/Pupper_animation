"""Configuration for the Pupper V3 leg-lift policy.

Single source of truth for: the canonical 12-joint order (must match the robot's
`neural_controller` config.yaml and the pupperv3-mjx training env), joint limits,
the default standing pose, the per-leg "lifted" targets, the reward weights, and
the PPO hyperparameters.

Design (see README.md): ONE policy, conditioned on a command = which leg is
currently lifted (stand / FL / FR / BR / BL). "Hold" is just the command staying
constant, so hold duration is operator-controlled on the robot (each press of the
O button advances a clockwise state machine) and is NOT baked into the policy.

Conventions (see workspace-root CLAUDE.md): no silent fallbacks. Paths that don't
resolve raise; we never quietly substitute a default model.
"""

from pathlib import Path

import numpy as np
from ml_collections import config_dict

# ---------------------------------------------------------------------------
# Robot model
# ---------------------------------------------------------------------------

# The MJX training model the providers' pipeline uses: full body, ground plane,
# foot sites, position actuators, and a "home" keyframe. It lives in the separate
# pupper_v3_description checkout; we reference it in place (its meshes resolve
# relative to the xml) rather than copying assets across repos.
_WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    _WORKSPACE_DIR.parent.parent
    / "Stanford"
    / "training"
    / "pupper_v3_description"
    / "description"
    / "mujoco_xml"
    / "pupper_v3_complete.mjx.position.xml"
)


def resolve_model_path(path=None) -> Path:
    """Resolve the Pupper MJX model path, raising if it does not exist."""
    p = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Pupper MJX model not found at {p}. Pass --model_path or fix "
            f"configs.DEFAULT_MODEL_PATH to point at pupper_v3_complete.mjx.position.xml."
        )
    return p


# ---------------------------------------------------------------------------
# Joints  (canonical order — DO NOT reorder; the robot pins this exact order)
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    "leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
    "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
    "leg_back_r_1", "leg_back_r_2", "leg_back_r_3",
    "leg_back_l_1", "leg_back_l_2", "leg_back_l_3",
]  # fmt: skip

# Each leg's (abduction=_1, hip=_2, knee=_3) joint indices.
LEG_JOINT_INDICES = {
    "front_r": [0, 1, 2],
    "front_l": [3, 4, 5],
    "back_r": [6, 7, 8],
    "back_l": [9, 10, 11],
}

FOOT_SITE_NAMES = [
    "leg_front_r_3_foot_site",
    "leg_front_l_3_foot_site",
    "leg_back_r_3_foot_site",
    "leg_back_l_3_foot_site",
]
FOOT_ROW_BY_LEG = {"front_r": 0, "front_l": 1, "back_r": 2, "back_l": 3}
TORSO_NAME = "base_link"

# Command states the policy is conditioned on. Index 0 = stand (no leg up). The
# clockwise lift order (FL -> FR -> BR -> BL) is enforced by the ON-ROBOT state
# machine, not here — the policy only needs to know which leg is up right now.
COMMAND_STATES = ["stand", "front_l", "front_r", "back_r", "back_l"]
NUM_COMMANDS = len(COMMAND_STATES)

# Standing "home" pose, identical to the locomotion policy's default_joint_pos.
DEFAULT_POSE = np.array(
    [0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 0.26, 0.0, -0.52, -0.26, 0.0, 0.52]
)

# Joint limits, copied from pupperv3-mjx PupperV3Env (same robot, same order).
JOINT_LOWER_LIMITS = np.array(
    [-1.220, -0.420, -2.790, -2.510, -3.140, -0.710,
     -1.220, -0.420, -2.790, -2.510, -3.140, -0.710]
)  # fmt: skip
JOINT_UPPER_LIMITS = np.array(
    [2.510, 3.140, 0.710, 1.220, 0.420, 2.790,
     2.510, 3.140, 0.710, 1.220, 0.420, 2.790]
)  # fmt: skip

# Per-leg joint delta (abduction, hip, knee) added to that leg's DEFAULT_POSE to
# put the foot in the air. PLACEHOLDER values — these MUST be tuned in sim (load
# the model, apply the target, confirm the foot clears the ground by ~target
# height and the body stays balanced). Signs mirror the L/R joint convention.
LIFT_DELTAS = {
    "front_r": np.array([0.0, 0.8, -0.9]),
    "front_l": np.array([0.0, -0.8, 0.9]),
    "back_r": np.array([0.0, 0.8, -0.9]),
    "back_l": np.array([0.0, -0.8, 0.9]),
}


def lifted_pose_for(leg: str) -> np.ndarray:
    """The 12-joint target pose with `leg` raised and the others at home."""
    pose = DEFAULT_POSE.astype(float).copy()
    idx = LEG_JOINT_INDICES[leg]
    pose[idx] += LIFT_DELTAS[leg]
    return np.clip(pose, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS)


def get_config() -> config_dict.ConfigDict:
    """Returns the full leg-lift training config."""
    return config_dict.create(
        # ---- command sampling during training ----
        # Hold a command for a random number of steps, then switch (this teaches
        # the policy smooth raise/hold/lower transitions). On the robot the same
        # transitions are driven by O-button presses instead.
        command_hold_steps_min=50,    # 1.0 s at 50 Hz
        command_hold_steps_max=150,   # 3.0 s
        stand_command_prob=0.25,      # fraction of commands that are "stand"

        # ---- timestepping ----
        ctrl_dt=0.02,   # 50 Hz policy, matches deployment repeat_action=10 @ 500Hz
        sim_dt=0.004,
        action_scale=0.3,
        position_control_kp=5.0,
        dof_damping=0.25,
        observation_history=1,  # set >1 to stack frames like the locomotion policy
        soft_joint_pos_limit_factor=0.95,

        # ---- episode / termination ----
        episode_length=600,        # 12 s; several command switches per episode
        terminal_body_angle=0.6,   # rad of tilt before we call it a fall
        terminal_body_z=0.08,      # torso too low => fell

        # ---- reward weights ----
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_pose=2.0,          # track the commanded target joint pose
                foot_clearance=1.0,         # commanded foot reaches target height
                stance_feet_contact=0.5,    # the other feet stay planted
                orientation=1.0,            # torso upright
                torso_height=0.5,           # torso near standing height
                action_rate=-0.01,          # smoothness (protect the polymer link)
                torques=-2e-4,
                dof_acc=-2.5e-7,
                dof_pos_limits=-1.0,
            ),
            tracking_sigma=0.25,
            target_foot_height=0.06,  # meters off the ground when a leg is up
        ),

        # ---- PPO (brax) ----
        ppo=config_dict.create(
            num_timesteps=150_000_000,
            num_evals=10,
            episode_length=600,  # kept in sync with episode_length above by train.py
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=32,
            num_updates_per_batch=4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            num_envs=8192,
            batch_size=256,
            seed=0,
        ),
        policy=config_dict.create(
            hidden_layer_sizes=(128, 128, 128),
            activation="swish",
        ),

        # ---- domain randomization (sensor noise, kicks, action latency) ----
        # Physics DR ranges live in workspace/randomize.py. Only the step-time
        # terms (noise added to the obs, random torso kicks, motor lag) live here.
        dr=config_dict.create(
            # Obs noise (uniform ±scale added per-step; from notebook cell 21)
            angular_velocity_noise=0.1,   # rad/s
            gravity_noise=0.05,           # unit vector components
            motor_angle_noise=0.05,       # rad
            last_action_noise=0.01,       # normalized action units

            # Random horizontal impulse kicks applied to the torso
            kick_probability=0.04,
            kick_vel=0.10,                # m/s, each component drawn from ±1 * kick_vel

            # Action latency: probability weights for the circular buffer, newest
            # element first. len(latency_distribution) = buffer depth.
            # [0.8, 0.2] => 80 % current action, 20 % one-step-old action.
            latency_distribution=(0.8, 0.2),
        ),
    )
