"""Configuration for the Pupper V3 WHEELED locomotion policy (`wheel` branch).

Single source of truth for: the canonical 12-joint order (must match the robot's
`neural_controller` config.yaml and the pupperv3-mjx training env), joint/ctrl
limits, the default splayed wheeled pose, the reward weights, and the PPO
hyperparameters.

Wheeled robot layout
--------------------
Still 12 joints, same names as the quadruped. Per leg: `_1` = abduction and
`_2` = hip are POSITION controlled as before, while `_3` -- formerly the knee --
is now the WHEEL's continuous spin joint and is VELOCITY controlled (ctrl =
target wheel speed in rad/s, no meaningful target angle). See
`pupper_v3_complete.mjx.position.xml`, whose `_3` joints are `limited="false"`
with `<velocity>` actuators.

That mixed actuation is the one thing an env MUST respect here: the providers'
`pupperv3_mjx.environment.PupperV3Env` (and our `leg_lift_env.py`, which mirrors
it) overwrite EVERY actuator's gains with one position-PD setting at load time,
which would silently corrupt the wheel actuators. `wheel_env.py` overrides the
position rows and the wheel rows separately -- see POSITION_ACTUATOR_ROWS /
WHEEL_ACTUATOR_ROWS below.

Leg-lift legacy
---------------
The leg-lift constants and `get_config()` below are retained from the fork but
are NOT used by the wheeled pipeline and no longer describe this branch's model
(the knee joints they assume are wheels now). The live leg-lift task lives on
`master`. Wheeled training uses `get_wheel_config()`.

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

# Each leg's (abduction=_1, hip=_2, wheel=_3) joint indices. `_3` was the knee on
# the quadruped; on this branch it is the wheel's continuous spin joint.
LEG_JOINT_INDICES = {
    "front_r": [0, 1, 2],
    "front_l": [3, 4, 5],
    "back_r": [6, 7, 8],
    "back_l": [9, 10, 11],
}

# The two actuation groups. Rows index BOTH the 12-joint vectors above and the 12
# actuators (the model declares one actuator per joint, in the same order -- an
# invariant wheel_env.py asserts at construction).
#
# Position rows are angle-controlled (rad); wheel rows are velocity-controlled
# (rad/s). Everything downstream that treats "the action vector" uniformly has to
# know which entries mean what, so these are the single place that split lives.
WHEEL_ACTUATOR_ROWS = [2, 5, 8, 11]
POSITION_ACTUATOR_ROWS = [0, 1, 3, 4, 6, 7, 9, 10]

# Wheel geometry, from the CAD/mesh (see
# Stanford/training/pupper_v3_description/WHEEL_MASS_LOG.md). The collision
# cylinder in the MJCF uses this same radius; the reward code needs it to convert
# between commanded body velocity and wheel angular velocity.
WHEEL_RADIUS = 0.048

# Top ground speed the wheels may be commanded to (m/s), and the wheel angular
# velocity that corresponds to (rad/s) -- the latter is what actually goes to the
# actuators, derived rather than hand-written so the two can never disagree.
#
# Capped at 1 m/s deliberately. An earlier 30 rad/s (~1.44 m/s) cap was tested in
# sim and the robot FLIPS at full command: driving all four wheels at action 1.0
# pitched it to 90 deg of tilt within ~250 steps. 1 m/s keeps full-scale command
# inside what the chassis can actually take.
#
# WHEEL_MAX_SPEED is mirrored by the `<velocity ctrlrange>` in the MJCF (which is
# set fractionally wider so this config, not the model, is the binding limit) --
# keep the two in sync.
WHEEL_MAX_LINEAR_SPEED = 1.0
WHEEL_MAX_SPEED = WHEEL_MAX_LINEAR_SPEED / WHEEL_RADIUS  # ~20.83 rad/s

# Sign that turns "drive forward" into a signed wheel-joint command, per wheel,
# in WHEEL_ACTUATOR_ROWS order (front_r, front_l, back_r, back_l).
#
# The left and right legs' frames are MIRRORED, so the two sides' wheel joints
# spin about OPPOSITE world axes: measured on the model at the home pose, the
# right wheels' spin axis is world -Y and the left wheels' is world +Y. A wheel
# rolling forward (+x) needs world +Y angular velocity, so the same positive
# joint command drives the right wheels backwards and the left wheels forwards.
# Left uncorrected, commanding all four wheels the same way makes them fight each
# other and the robot does not move at all -- confirmed in sim before this was
# added (wheels spinning at ~14 rad/s, body drifting <3 cm in 2 s).
#
# Applying it here means +1 action = "this wheel drives the robot forward" on all
# four, so the policy sees one consistent convention. The deployment side needs
# the same sign convention applied to the real motors.
WHEEL_FORWARD_SIGN = np.array([-1.0, 1.0, -1.0, 1.0])

# Wheel centre offset along the wheel body's own local z (= the spin axis), in m.
# The `_3` body's origin sits at the knee joint / motor output; the wheel's
# collision cylinder is centred WHEEL_CENTER_LOCAL_Z further out along that axis
# (0.0136 mount standoff + 0.01675 half-width). Used to find the wheel centre in
# world coordinates for ground-contact checks -- its lowest point is
# (centre z - WHEEL_RADIUS) while the spin axis is horizontal.
WHEEL_CENTER_LOCAL_Z = 0.03035

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

# Default wheeled pose = the ctrl vector the policy's actions are offsets FROM.
# Units are mixed, matching the mixed actuation: the `_1`/`_2` entries are joint
# ANGLES (rad), the `_3` (wheel) entries are wheel SPEEDS (rad/s), and 0 there
# means "wheels stopped" -- a continuously-spinning joint has no home angle.
#
# The +-1 rad abduction splay is the stance chosen in the MuJoCo viewer and
# committed as the model's `home` keyframe: it swings all four wheels out from
# under the body so they sit upright on the ground. Signs mirror left/right
# (front_r/back_r = +1, front_l/back_l = -1) because those two abduction joints
# have mirrored axes -- see the mirrored joint ranges below.
#
# Verified against the model: holding this ctrl under position control settles
# upright and stationary (see STAND_TORSO_HEIGHT).
DEFAULT_POSE = np.array(
    [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0]
)

# Per-row ctrl limits, mixed units to match DEFAULT_POSE. The `_1`/`_2` rows are
# the model's real joint angle limits (unchanged from the quadruped, copied from
# pupperv3-mjx PupperV3Env). The `_3` rows are NOT angle limits -- the wheel
# joints are `limited="false"` and spin freely -- they are the velocity ctrl range
# +-WHEEL_MAX_SPEED, matching the MJCF's `<velocity ctrlrange>`.
JOINT_LOWER_LIMITS = np.array(
    [-1.220, -0.420, -WHEEL_MAX_SPEED, -2.510, -3.140, -WHEEL_MAX_SPEED,
     -1.220, -0.420, -WHEEL_MAX_SPEED, -2.510, -3.140, -WHEEL_MAX_SPEED]
)  # fmt: skip
JOINT_UPPER_LIMITS = np.array(
    [2.510, 3.140, WHEEL_MAX_SPEED, 1.220, 0.420, WHEEL_MAX_SPEED,
     2.510, 3.140, WHEEL_MAX_SPEED, 1.220, 0.420, WHEEL_MAX_SPEED]
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
#
# WHEELED VALUES (this branch). The comment block above is the leg-lift
# derivation, kept because the mechanism is identical -- tanh-squashed head, so
# the reachable envelope is exactly DEFAULT_POSE +/- ACTION_SCALE -- but the
# numbers themselves are re-chosen for wheels and are UNTUNED starting points.
#
# Abduction (0.5 rad): lets the stance widen/narrow around the +-1 rad splay for
# balance and turning lean, while staying well inside the abduction limits
# (front_r_1 spans -1.22..2.51, so 1.0 +- 0.5 has margin on both sides).
#
# Hip (0.4 rad): deliberately small. The hip's own limits are asymmetric and
# mirrored (front_r_2 spans -0.42..3.14, front_l_2 spans -3.14..0.42), so the
# widest envelope that is symmetric about the 0.0 default AND legal on both
# sides is +-0.42. 0.4 sits just inside that, which keeps the reachable range
# identical left-to-right -- a larger value would clip asymmetrically and quietly
# give the two sides different authority. The legs are suspension/posture here,
# not the thing producing locomotion, so this does not need to be large.
#
# Wheel (WHEEL_MAX_SPEED): full-scale action = full wheel speed, i.e. the wheel
# entries of the action vector ARE the normalized drive command. This is the
# term that actually moves the robot.
ACTION_SCALE_ABDUCTION = 0.5
ACTION_SCALE_HIP = 0.4
ACTION_SCALE_WHEEL = WHEEL_MAX_SPEED
ACTION_SCALE = np.array(
    [ACTION_SCALE_ABDUCTION, ACTION_SCALE_HIP, ACTION_SCALE_WHEEL] * 4
)

# Direction each leg's hip joint (_2) must rotate to raise that leg. Left and right
# legs mirror. Used by the dense `lift_progress` reward -- see get_config().
HIP_LIFT_SIGN = {"front_r": 1.0, "front_l": -1.0, "back_r": 1.0, "back_l": -1.0}

# Torso height (m) the robot settles at resting on its wheels in DEFAULT_POSE under
# position control at position_control_kp. MEASURED from the wheeled model (settle
# the home ctrl for 3 s and read base_link z), not guessed -- the leg-lift value
# this replaces (0.1556) was measured on the quadruped standing on feet and is
# ~24 mm too high for the wheeled stance, which would pay the policy to hoist
# itself.
STAND_TORSO_HEIGHT = 0.1313

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


# Bodies carrying each wheel. `_3` is the wheel body now (the old shin/knee body,
# with the wheel's mass and collision cylinder attached directly to it), so these
# are the same names KNEE_BODY_NAMES lists -- kept separate because the wheeled
# code means "the wheel", not "the knee". Same row order as FOOT_ROW_BY_LEG.
WHEEL_BODY_NAMES = [
    "leg_front_r_3",
    "leg_front_l_3",
    "leg_back_r_3",
    "leg_back_l_3",
]


def get_config() -> config_dict.ConfigDict:
    """LEGACY leg-lift config -- not used by the wheeled pipeline.

    Retained from the fork for reference only. It describes a knee joint that no
    longer exists on this branch's model (it is a wheel now), so it will NOT
    produce a working leg-lift run here; the live leg-lift task is on `master`.
    Wheeled training uses get_wheel_config().
    """
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


def get_wheel_config() -> config_dict.ConfigDict:
    """Returns the full WHEELED-locomotion training config.

    Task: drive to a commanded body velocity (vx, vy, yaw rate) on four wheels,
    keeping the torso upright, level and at its resting height, with the legs
    holding the splayed DEFAULT_POSE stance.

    Everything here is a first pass, not a tuned configuration -- no wheeled run
    has been trained yet. The reward weights follow the shape of the providers'
    locomotion reward (tracking terms positive and dominant, everything else a
    small regularizer) rather than the leg-lift reward, which was built around a
    completely different objective.
    """
    return config_dict.create(
        # ---- velocity command sampling ----
        # Commands are resampled mid-episode so one policy learns the whole
        # command space rather than a single gait. Ranges are modest relative to
        # the ~1.44 m/s the wheels can actually reach (WHEEL_MAX_SPEED), leaving
        # the policy headroom to exceed the command while correcting.
        command_resample_steps_min=100,   # 2.0 s at 50 Hz
        command_resample_steps_max=250,   # 5.0 s
        # Kept strictly INSIDE WHEEL_MAX_LINEAR_SPEED (1.0 m/s): a command at the
        # actuation cap would leave the policy no wheel authority left to steer or
        # correct with, since it would already be saturated just to hold speed.
        lin_vel_x_range=(-0.5, 0.8),      # m/s, forward-biased
        lin_vel_y_range=(-0.2, 0.2),      # m/s, lateral (wheels can't strafe;
                                          # this mostly teaches it to refuse)
        ang_vel_yaw_range=(-1.5, 1.5),    # rad/s
        # Fraction of commands that are exactly zero. Standing still on wheels is
        # its own skill (and the most common real command), and without explicit
        # zero commands a velocity-tracking policy tends to creep.
        zero_command_prob=0.15,
        # Below this speed a command counts as "stand still" for the
        # stand_still reward term.
        stand_still_threshold=0.05,

        # ---- timestepping ----
        ctrl_dt=0.02,   # 50 Hz policy, matches deployment repeat_action=10 @ 500Hz
        sim_dt=0.004,
        action_scale=tuple(float(a) for a in ACTION_SCALE),  # per-joint, mixed units
        position_control_kp=5.0,   # applied to POSITION_ACTUATOR_ROWS only
        dof_damping=0.25,          # ditto
        # Velocity-actuator gain for WHEEL_ACTUATOR_ROWS. Mirrors the MJCF's
        # `<velocity kv>`; wheel_env.py writes it back over the loaded model so
        # this file stays the single source of truth.
        wheel_kv=0.35,
        observation_history=1,  # set >1 to stack frames like the locomotion policy
        soft_joint_pos_limit_factor=0.95,

        # ---- episode / termination ----
        episode_length=600,        # 12 s; several command resamples per episode
        terminal_body_angle=0.6,   # rad of tilt (~34 deg) before we call it a fall.
                                   # Looser than leg-lift's 0.4: a wheeled robot
                                   # legitimately leans under acceleration.
        terminal_body_z=0.08,      # torso on the floor => chassis is down

        # ---- reward weights ----
        reward_config=config_dict.create(
            scales=config_dict.create(
                # -- the task --
                tracking_lin_vel=2.0,       # match commanded body-frame xy velocity
                tracking_ang_vel=1.0,       # match commanded yaw rate
                # -- posture: hold the wheeled stance --
                orientation=1.0,            # torso upright
                torso_height=1.0,           # torso at its resting height
                stance_pose=1.0,            # the 8 leg joints stay at DEFAULT_POSE
                wheels_on_ground=1.0,       # all four wheels stay in contact
                stand_still=0.5,            # zero command => actually stop
                # -- penalties --
                lin_vel_z=-0.5,             # no bouncing
                ang_vel_xy=-0.05,           # no roll/pitch rate
                action_rate=-0.05,          # smooth commands (sim-to-real)
                torques=-2e-4,
                dof_acc=-1e-6,
                dof_pos_limits=-1.0,        # position joints only, see wheel_env.py
            ),
            # Gaussian widths for the tracking terms, same exp(-err^2/sigma) shape
            # the providers' locomotion reward uses.
            tracking_lin_vel_sigma=0.10,   # (m/s)^2
            tracking_ang_vel_sigma=0.25,   # (rad/s)^2
            # Posture widths, carried over from the leg-lift config where the same
            # quantity is being measured the same way.
            orientation_sigma=0.05,        # on (1 - cos tilt); looser than leg-lift
            torso_height_sigma=0.0004,     # ~0.78 at 1 cm off
            # Only the 8 position joints are scored (a wheel has no home angle).
            stance_pose_sigma=0.15,
            # Squared body speed at which the stand_still bonus has fully decayed.
            stand_still_sigma=0.02,
        ),

        # ---- PPO (brax) ----
        # Same shape as the leg-lift run, which trained successfully on this
        # hardware; velocity tracking is a better-conditioned objective than the
        # leg-lift reward, so this is a reasonable starting point.
        ppo=config_dict.create(
            num_timesteps=200_000_000,
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
            # model_loader.h only implements tanh/relu/sigmoid/softmax/elu.
            activation="elu",
        ),

        # ---- domain randomization (sensor noise, kicks, action latency) ----
        dr=config_dict.create(
            angular_velocity_noise=0.1,   # rad/s
            gravity_noise=0.05,           # unit vector components
            motor_angle_noise=0.05,       # rad, on the position joints
            # Wheel rows of the joint observation are SPEEDS, not angles, so they
            # get their own (much larger, in rad/s) noise scale.
            wheel_velocity_noise=0.5,     # rad/s
            last_action_noise=0.01,       # normalized action units
            kick_probability=0.05,
            kick_vel=0.15,                # m/s
            latency_distribution=(0.5, 0.3, 0.2),
        ),
    )
