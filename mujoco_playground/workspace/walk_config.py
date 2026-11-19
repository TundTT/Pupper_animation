"""Walking settings; the XML is authoritative for home pose, gains, and limits."""
from ml_collections import config_dict

def get_config():
    return config_dict.create(
        ctrl_dt=.02, episode_length=600, observation_history=4,
        # Zero preserves XML friction; positive values override floor sliding friction.
        floor_friction=0., friction_range=(.8, 2.5),
        foot_model='rigid_flush',
        # Scale the upstream leg offset around the newly extended capsule/ring
        # assembly. Both move together, so the flush fit survives length DR.
        leg_length_common_range=(.90, .98), leg_length_per_leg_range=(.96, 1.04),
        # Extra knee flexion makes a centered, smooth swing reachable. These
        # command bounds remain within the authoritative XML joint limits.
        action_scale=(.5, .25, 1.1)*4,
        smoothness_reference_scale=(.5, .25, .5)*4,
        moving_height_offset=-.003,
        # Symmetric range: uniform sampling then gives backward and forward equal
        # magnitude coverage instead of forward getting 2x the top speed and 5x the
        # at-speed sample density.
        command_low=(-.35, -.15, -.8), command_high=(.35, .15, .8),
        command_hold_steps=150, stand_probability=.2,
        # Of all commands: 20% stand, 20% forward, 20% reverse, 10% lateral,
        # 10% turn in place, 20% mixed. Pure axes prevent direction forgetting.
        straight_probability=.5, axis_probability=.25, straight_min_speed=.10,
        # Grade short swings instead of making everything below 80 ms equally
        # unrewarded. Persistent contact is unaffected; only touchdown is scored.
        air_time_floor=-.08,
        touchdown_speed_allowance=.15,
        # Keep ~.07 variance at .35 m/s, but make ignoring a slow command costly.
        tracking_variance_base=.0025, tracking_variance_speed_gain=.55,
        tracking_variance_yaw_gain=.27,
        minimum_swing_clearance=.012,
        planned_swing_duration=.32, planned_swing_height=.016,
        planned_swing_bonus=.25,
        # The front feet under-lifted under equal weighting despite reachable
        # geometry. Strengthen their trajectory error, keeping bonus unchanged.
        planned_swing_foot_weights=(3.,3.,1.,1.),
        # Permit ordinary rounded-tip rolling; discourage leaning on the foot.
        tip_tilt_allowance_deg=15., tip_tilt_scale_deg=45.,
        # Legacy diagnostic only: bottom compression contributes no reward.
        bottom_allowance=.009, side_clearance=.001,
        joint_limit_margin=.15,
        # Height-area reward cap, not a prescribed apex. The audit found short,
        # rounded swings below this cap and limited centered-swing reachability.
        swing_clearance_target=.028,
        reset_joint_noise=.01, reset_xy_noise=.01, reset_yaw_noise=.1,
        sensor_noise=.01, latency_probability=.2,
        push_probability=.02, push_velocity=.3,
        terminal_tilt=.65, terminal_height=.085,
        reward_scales=config_dict.create(
            tracking_linear=6., tracking_yaw=3., upright=.3,
            height=-12., vertical_velocity=-.3, roll_pitch_velocity=-.05,
            foot_slip=-.6, air_time=1.5, touchdown=-1.5, tip_support=-.5,
            clearance_shortfall=-.5,
            # The old height-area term is retained for old-checkpoint evaluation;
            # the selected policy uses the full-duration curve and credit below.
            swing_clearance=0., stand_pose=-3., stance_feet=-.4,
            ring_side=-3., ring_bottom=0., ring_rub=-2.,
            ring_side_contact=-6.,
            unwanted_contact=-2., action_rate=-.30, torques=-.0002,
            # Fixed physical normalization keeps wider commands from reducing
            # the cost of an equally abrupt joint-target change.
            action_acceleration=-1.6,
            planned_swing=-6.,
            joint_limits=-1.5, termination=-2.,
        ),
        ppo=config_dict.create(
            num_timesteps=200_000_000, num_envs=8192, num_evals=15,
            num_eval_envs=128, unroll_length=20, batch_size=256,
            num_minibatches=32, num_updates_per_batch=4,
            discounting=.97, learning_rate=3e-4, entropy_cost=.01,
            normalize_observations=True, reward_scaling=1., seed=0,
        ),
        hidden_layer_sizes=(128,128,128), activation='elu',
    )
