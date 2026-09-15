"""Stationary, operator-paced leg-to-wheel policy settings."""
from ml_collections import config_dict
from workspace.walk_config import get_config as walk_config

COMMAND_STATES = ('stand', 'FL', 'FR', 'BR', 'BL')
# Geometry/actuator order is FR, FL, BR, BL.
COMMAND_TO_FOOT = (-1, 1, 0, 2, 3)


def get_config():
    c = walk_config()
    for name in ('command_low', 'command_high', 'command_hold_steps', 'swing_clearance_target'):
        del c[name]
    c.command_states = COMMAND_STATES
    c.command_hold_seconds = (1., 10.)
    c.stand_probability = .25
    # MJCF order is abduction (_1), lifting hip (_2), knee/hub (_3).
    # Verified by finite differences of actual capsule-bottom height.
    c.action_scale = (.5, 1.0, .8)*4
    c.lift_progress_reference = .5
    c.lift_clearance_target = .025
    c.lift_clearance_gate = .015
    c.allowed_body_drift = .04
    c.touchdown_warmup_seconds = .2  # ignore spawn settling, before any command switch
    c.soft_touchdown_speed = .10  # m/s, pre-impact downward speed
    c.converted_probability = .5
    c.leg_shortening_range = (0., .010)  # independent per limb, metres
    # Touchdown is an impulse reward (paid once), whereas missing stance contact
    # costs every step. Weight 2 favored fast descent; 100 makes a gentle landing
    # worth the extra airborne time. Physical gates stay fixed at 0.10 m/s.
    c.reward_scales = config_dict.create(
        lift_height=3., lift_progress=.75, stance_contact=-5., soft_touchdown=-100., body_drift=-3.,
        upright=20., height=-8., vertical_velocity=-.3, roll_pitch_velocity=-.1,
        foot_slip=-1., ring_side=-1., ring_bottom=-1., ring_rub=-2.,
        unwanted_contact=-2., action_rate=-.1, torques=-.0002,
        joint_limits=-1.5, termination=-10.,
    )
    c.ppo.reward_scaling = .5  # preserve critic scale when refining stronger balance costs
    return c
