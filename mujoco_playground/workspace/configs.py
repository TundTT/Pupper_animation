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
# Verified against the model: hip delta 1.6 rad reaches ~0.145 m of foot clearance and
# is inside every hip joint limit, giving headroom above the ~1.4 rad the lift reward
# saturates at. The deployment side already supports a 12-element action_scale array
# (neural_controller.cpp's set_param_from_json_mixed), so this needs no C++ change.
#
# The hip was briefly set to 2.0 and pulled back: a wider range multiplies the policy's
# own exploration noise into violent leg swings that just tip the robot over, which
# teaches it to hold still rather than to lift. 1.6 still clears far more than the
# behavior needs.
ACTION_SCALE_ABDUCTION = 0.5
ACTION_SCALE_HIP = 1.6
ACTION_SCALE_KNEE = 1.0
ACTION_SCALE = np.array(
    [ACTION_SCALE_ABDUCTION, ACTION_SCALE_HIP, ACTION_SCALE_KNEE] * 4
)

# Direction each leg's hip joint (_2) must rotate to raise that leg. Left and right
# legs mirror. Used by the dense `lift_progress` reward -- see get_config().
HIP_LIFT_SIGN = {"front_r": 1.0, "front_l": -1.0, "back_r": 1.0, "back_l": -1.0}

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

# Per-step cap (action-space units, |a|<=1) on how far the ACTIVELY-LIFTED leg's
# hip action may move from what was actually applied last step. Models an
# actuator max-velocity limit so the lift ramps in instead of snapping to the
# commanded height in one step (observed on-hardware 2026-08-17/18 -- see
# workspace/README.md Status -- as "snaps to target, then seems to work on
# balance" rather than lifting while continuously balancing).
#
# Deliberately scoped to ONLY the lifted leg's hip (leg_lift_env.py selects it
# per-step via the same per-command lookup as _lift_hip_idx), not applied
# body-wide: the 9 stance-leg joints are exactly what corrects balance while a
# leg is in the air, and a uniform cap would blunt that responsiveness along
# with the snap. See leg_lift_env.py step() for how this is applied: it clamps
# only the PHYSICS consequence (what reaches pipeline_step); action_rate/
# dof_acc/obs last_act are computed on the raw action, so exceeding this limit
# costs reward for no physical gain, which should pull the policy's raw output
# toward already respecting it -- i.e. this is expected (not yet hardware-
# verified) to transfer to the deployed, unclamped policy without a
# neural_controller.cpp change.
#
# 0.05 action units = 0.08 rad/step at ACTION_SCALE_HIP=1.6, reaching the full
# hip_lift_reference=1.4 rad target in ~18 steps (~0.35 s) -- gradual, but with
# large margin inside the 1.0 s minimum command hold (command_hold_steps_min).
#
# CONFIRMED on hardware (2026-08-19): the lift raises gradually rather than
# snapping, and the raw-action mechanism above (clamp physics only, penalize
# the raw action via action_rate) transferred to the deployed policy with no
# neural_controller.cpp change needed, as hoped.
LIFT_HIP_MAX_ACTION_DELTA = 0.05

# How many steps the SAME rate limit above continues to apply to a leg's hip
# after the command switches AWAY from it (i.e. on the way back down), timed
# from the step the switch takes effect. Without this, the lowering leg falls
# out of the (raise-only) mask above the instant the command changes, and its
# hip is free to snap home at full speed -- confirmed as a real on-hardware
# problem 2026-08-20 (see workspace/README.md Status): the raise-side fix
# above did not cover this, since it only ever tracked whichever leg is UP
# NOW, not whichever leg WAS just up. Same order of magnitude as the ~18 steps
# the raise itself takes, plus a small margin for whatever fraction of the
# full lift was actually reached (a leg lowered from a partial lift has less
# distance to cover, so this is a safe upper bound, not a precise one).
LOWER_HIP_RATE_LIMIT_STEPS = 20


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
        lift_hip_max_action_delta=LIFT_HIP_MAX_ACTION_DELTA,  # see constant above
        lift_hip_max_action_delta_lower_steps=LOWER_HIP_RATE_LIMIT_STEPS,  # see constant above
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
        # Walking away from the spawn point ends the episode too. Added after a run
        # that lifted well and stood upright (tilt 3.7 deg, torso height spot on) but
        # translated the torso ~0.14 m doing it, and would NOT come back down from
        # that on reward shaping alone -- body_drift had already bottomed out, so
        # there was no gradient left to pull it in. Termination is what worked for
        # tilt and sitting; same lever here. 0.09 m is ~7x the ~12 mm of CoM shift a
        # front-leg lift actually requires, so it constrains sloppiness, not physics.
        terminal_body_drift=0.09,
        # Spinning in place ends the episode too (rad, ~29 deg). Added after finding
        # that policies which scored well on every other posture measure still yawed
        # 40-50 deg over a 12 s showcase -- nothing in the reward looked at yaw, so
        # nothing stopped it.
        terminal_body_yaw=0.5,

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
                # These must clearly beat "just keep standing and never risk it": the
                # posture terms below sum to 8.5 and are earned whether or not a leg
                # goes up, so these two are the ONLY terms that discriminate. A first
                # attempt at lift_height=4.0 with no lift_progress term collapsed into
                # exactly that failure -- a policy that stood flawlessly (tilt 0.8 deg,
                # drift 17 mm) and never once raised a foot, because standing still
                # already banked ~102 of a ~144 maximum at zero risk.
                #
                # lift_progress is the fix for the DEAD ZONE that caused it:
                # lift_height clips to 0 whenever the foot is on the ground, so there
                # was no gradient at all rewarding the first few degrees of hip
                # rotation. lift_progress is measured on the hip ANGLE, so it pays out
                # from the very first degree and rises continuously all the way to a
                # full lift -- a smooth path out of the standing local optimum.
                # Both saturate at the same configuration (~1.4 rad of hip), so they
                # reinforce rather than compete.
                lift_progress=3.0,          # dense: hip rotating toward "up" (no dead zone)
                lift_height=8.0,            # the real objective: actual foot clearance
                # It is safe to weight these heavily because posture is protected by
                # episode TERMINATION (tilt / torso height / knee-on-floor), not by
                # out-weighing them.
                lift_pose_prior=0.5,        # lifted leg's abduction+knee stay home => lifts in-plane
                # -- everything else holds the standing pose --
                # stance_pose and body_drift raised (from 2.5 / 1.5) after a run that
                # nailed the lift but held the stance legs ~9 deg off home and let the
                # torso wander 0.14 m. Both describe "don't rearrange yourself to lift".
                stance_pose=3.5,            # the three planted legs stay AT the home pose
                stance_feet_contact=1.0,    # ...and their feet stay on the ground
                orientation=2.0,            # torso upright (no leaning to fake height)
                heading=2.0,                # torso keeps FACING the way it started
                torso_height=1.5,           # torso at true standing height (no sitting)
                body_drift=2.5,             # torso does not translate away from where it started
                # -- penalties --
                ground_contact=-3.0,        # no knee/limb resting on the floor
                # Raised from -0.01, then again from -0.05 (2026-08-18) to chase
                # whole-body oscillation observed while a leg is held up on hardware
                # (not just the lifted leg -- the stance legs/body wobble too, so this
                # stays a body-wide penalty rather than being scoped like the hip rate
                # limit above). Smoothness is a sim-to-real requirement here, not just
                # a nicety: it is what stops the policy commanding position steps the
                # real motor cannot follow. Also protects the polymer link. A starting
                # point, not a tuned value -- check the eval video/oscillation by eye
                # and adjust.
                action_rate=-0.1,
                # Keep the lift OFF the motor's torque ceiling. Measured on the first
                # policy: steady hold needs only ~0.11 Nm, but the raise transient
                # pinned the lifting hip at exactly 3.000 Nm -- the model's forcerange
                # -- i.e. saturated. A saturated ideal position actuator still tracks
                # in sim; a real motor at saturation hits voltage/current limits and
                # backlash and does not, so anything relying on it transfers badly.
                # The old `torques` term could not prevent this: at -2e-4, a fully
                # saturated joint cost -0.0018/step against ~20/step of reward.
                torque_limit=-2.0,
                torques=-2e-4,
                # Raised 4x from -2.5e-6 (2026-08-18), alongside action_rate above,
                # to damp the whole-body oscillation seen on hardware while a leg is
                # held up. Same "starting point, not tuned" caveat applies.
                dof_acc=-1e-5,
                dof_pos_limits=-1.0,
            ),
            # Foot clearance (m) that earns full lift credit. The reward ramps
            # linearly to this and then flats, so the policy is pushed to lift as
            # high as it can without being paid to contort past a useful height.
            # Reachable envelope is ~0.145 m at the hip action limit, so this leaves
            # headroom rather than pinning the joint against its stop.
            target_lift_height=0.12,
            # Hip rotation (rad, in the lifting direction) that earns full
            # lift_progress credit. Measured to correspond to ~0.12 m of foot
            # clearance, i.e. the same configuration target_lift_height describes.
            hip_lift_reference=1.4,
            # 9 stance joints, squared-error sum. Tightened from 0.25: at that width a
            # ~9 deg/joint deviation still scored ~0.4, i.e. the reward barely cared.
            stance_pose_sigma=0.15,
            lift_prior_sigma=0.10,      # lifted leg's abduction + knee only
            orientation_sigma=0.02,     # on (1 - cos tilt): ~0.83 at 5 deg, ~0.006 at 26 deg
            heading_sigma=0.02,         # same curve, applied to yaw away from the start heading
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
            # Motor torque ceiling (Nm), matching the model's actuator forcerange.
            # The torque_limit penalty is zero below torque_soft_fraction of this and
            # ramps to 1 per joint at the ceiling, so ordinary effort is free and only
            # approaching saturation costs anything.
            torque_limit_nm=3.0,
            torque_soft_fraction=0.6,
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
            kick_probability=0.05,
            kick_vel=0.15,                # m/s, each component drawn from ±1 * kick_vel

            # Action latency: probability weights for the circular buffer, newest
            # element first. len(latency_distribution) = buffer depth.
            # Deepened to 3 steps (up to 60 ms at 50 Hz) from a 2-step [0.8, 0.2].
            # The real path is ROS2 -> CAN -> motor controller, which is both slower
            # and more variable than one control period, and a policy tuned on
            # near-zero latency is exactly the kind that oscillates on hardware.
            latency_distribution=(0.5, 0.3, 0.2),
        ),
    )
