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

# Body owning each leg's knee joint (joint "_3", defined at that body's own
# origin — so this body's world position IS the knee pivot location). Same
# row order as FOOT_SITE_NAMES/FOOT_ROW_BY_LEG. Used for the knee_clearance
# reward term (recovered from a wandb run's logged config; the run's actual
# leg_lift_env.py implementation was lost with the machine it trained on —
# this body-position choice is a reconstruction, not the original code).
KNEE_BODY_NAMES = [
    "leg_front_r_3",
    "leg_front_l_3",
    "leg_back_r_3",
    "leg_back_l_3",
]

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

# Per-JOINT action scale (rad of commanded offset from DEFAULT_POSE at |action|=1).
#
# This is the single most important change vs. the earlier runs. The policy head is
# tanh-squashed, so actions live in (-1, 1) and the reachable joint range is exactly
# DEFAULT_POSE +/- ACTION_SCALE. The old uniform 0.3 capped EVERY joint at 0.3 rad
# from home, which made the commanded lift physically unreachable by leg motion
# (a 0.08 m foot clearance needs ~1.2 rad of hip rotation) -- so the only way to earn
# the clearance reward was to move the BODY. That is precisely the "shifts back /
# sits / drops another limb" behavior we are fixing here.
#
# Roles per leg: _1 = abduction (sideways splay), _2 = hip (this is what swings the
# leg up), _3 = knee. The hip gets a large range so the leg can actually be raised;
# abduction and knee stay tight so the leg lifts in its own sagittal plane and the
# three stance legs keep precise control authority near the home pose.
#
# Verified against the model: hip delta 2.0 rad reaches ~0.19 m of foot clearance and
# is inside every hip joint limit, giving headroom above the 0.15 m the reward
# saturates at. The deployment side already supports a 12-element action_scale array
# (neural_controller.cpp's set_param_from_json_mixed), so this needs no C++ change.
ACTION_SCALE_ABDUCTION = 0.5
ACTION_SCALE_HIP = 2.0
ACTION_SCALE_KNEE = 1.0
ACTION_SCALE = np.array(
    [ACTION_SCALE_ABDUCTION, ACTION_SCALE_HIP, ACTION_SCALE_KNEE] * 4
)

# Torso height (m) the robot actually settles at while standing in DEFAULT_POSE under
# position control at position_control_kp. MEASURED from the model, not guessed: the
# previous config's 0.14 target was below the true standing height, so the height
# reward was actively paying the policy to sit down.
STAND_TORSO_HEIGHT = 0.1556

# Radius of the knee collision sphere on each leg_*_2 body. Its center coincides with
# the origin of the corresponding leg_*_3 body (KNEE_BODY_NAMES), so a knee's height
# above the floor is exactly (that body's world z) - KNEE_RADIUS. Used to detect and
# penalize "resting a knee on the ground to cheat the lift".
KNEE_RADIUS = 0.025


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
        action_scale=tuple(float(a) for a in ACTION_SCALE),  # per-joint, see ACTION_SCALE
        position_control_kp=5.0,
        dof_damping=0.25,
        observation_history=1,  # set >1 to stack frames like the locomotion policy
        soft_joint_pos_limit_factor=0.95,

        # ---- episode / termination ----
        # Tightened from (0.6 rad, 0.08 m): the failure mode being fixed here was a
        # stable ~26 deg (0.45 rad) lean with the body dropped, which the old
        # thresholds happily tolerated for a full episode. Now it ends the episode.
        episode_length=600,        # 12 s; several command switches per episode
        terminal_body_angle=0.4,   # rad of tilt (~23 deg) before we call it a fall
        terminal_body_z=0.10,      # torso too low => sat down
        terminal_knee_clearance=0.005,  # a knee touching the floor ends the episode

        # ---- reward weights ----
        # REDESIGNED for "hold the standing pose, raise the commanded leg high".
        # The previous weights (recovered from wandb run leg_lift_2026-06-24_20-04-18)
        # are deliberately NOT carried over: they tracked a fixed lifted-pose target
        # that conflicted with an unreachable foot/knee height target, which is what
        # taught the policy to lean and sit. See ACTION_SCALE above for the actuation
        # half of that bug.
        reward_config=config_dict.create(
            scales=config_dict.create(
                # -- raise the commanded leg --
                # Weighted to clearly beat "just keep standing and never risk it":
                # the posture terms below sum to 8.5 and are earned whether or not a
                # leg goes up, so lift_height is the ONLY term that discriminates.
                # At 4.0 a clean lift is worth ~50% more than refusing to lift. It is
                # safe to push this high because posture is protected by episode
                # TERMINATION (tilt / torso height / knee-on-floor), not just weights.
                lift_height=4.0,            # ramp: higher foot = more reward, "as high as possible"
                lift_pose_prior=0.5,        # lifted leg's abduction+knee stay home => lifts in-plane
                # -- everything else holds the standing pose --
                stance_pose=2.5,            # the three planted legs stay AT the home pose
                stance_feet_contact=1.0,    # ...and their feet stay on the ground
                orientation=2.0,            # torso upright (no leaning to fake height)
                torso_height=1.5,           # torso at true standing height (no sitting)
                body_drift=1.5,             # torso does not translate away from where it started
                # -- penalties --
                ground_contact=-3.0,        # no knee/limb resting on the floor
                action_rate=-0.01,          # smoothness (protect the polymer link)
                torques=-2e-4,
                dof_acc=-2.5e-6,
                dof_pos_limits=-1.0,
            ),
            # Foot clearance (m) that earns full lift credit. The reward ramps
            # linearly to this and then flats, so the policy is pushed to lift as
            # high as it can without being paid to contort past a useful height.
            # Reachable envelope is ~0.19 m at the hip action limit, so this leaves
            # headroom rather than pinning the joint against its stop.
            target_lift_height=0.15,
            stance_pose_sigma=0.25,     # 9 stance joints, squared-error sum
            lift_prior_sigma=0.10,      # lifted leg's abduction + knee only
            orientation_sigma=0.02,     # on (1 - cos tilt): ~0.83 at 5 deg, ~0.006 at 26 deg
            torso_height_sigma=0.0004,  # ~0.78 at 1 cm off, ~0.37 at 2 cm off
            # Lifting a FRONT leg leaves the CoM ~12 mm outside the remaining 3-foot
            # support triangle (measured), so a small body shift is physically
            # MANDATORY -- demanding zero drift would be asking for the impossible.
            # This is a deadband: free movement up to allowed_body_drift, penalized
            # beyond it.
            allowed_body_drift=0.035,
            body_drift_sigma=0.0025,
            # A knee closer than this to the floor starts losing reward, well before
            # it actually touches (stance knees sit ~0.065 m up at the home pose).
            knee_ground_margin=0.04,
        ),

        # ---- PPO (brax) ----
        ppo=config_dict.create(
            num_timesteps=200_000_000,
            # More evals than before (was 10) purely for auditability: each eval
            # renders a rollout video, so this gives a finer-grained view of how the
            # lift behavior develops over training.
            num_evals=15,
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
            # "elu", not "swish": the vendored RTNeural build in neural_controller's
            # model_loader.h only implements tanh/relu/sigmoid/softmax/elu -- an
            # unrecognized activation string silently yields a null layer that
            # segfaults on load. "elu" is what the already-deployed locomotion
            # policy uses (see policy_latest.json), so it's proven to work.
            activation="elu",
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
