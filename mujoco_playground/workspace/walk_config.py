"""Walking settings; the XML is authoritative for home pose, gains, and limits."""
from ml_collections import config_dict

def get_config():
    return config_dict.create(
        ctrl_dt=.02, episode_length=600, observation_history=4,
        # Joint order per leg is (hip pitch, abduction, knee); hip pitch drives the
        # fore/aft swing that both walking directions need most, so it gets the
        # largest authority, not abduction.
        action_scale=(.5, .25, .5)*4,
        # Symmetric range: uniform sampling then gives backward and forward equal
        # magnitude coverage instead of forward getting 2x the top speed and 5x the
        # at-speed sample density.
        command_low=(-.35, -.15, -.8), command_high=(.35, .15, .8),
        command_hold_steps=150, stand_probability=.2,
        bottom_allowance=.006, side_clearance=.001,
        joint_limit_margin=.15,
        # 20mm of capsule-bottom height ~= 17mm of visible mesh clearance (the
        # visual foot hangs ~2.9mm below the collision capsule); reachable at zero
        # ring-clearance cost within the action range via coordinated hip+knee.
        swing_clearance_target=.020,
        reset_joint_noise=.01, reset_xy_noise=.01, reset_yaw_noise=.1,
        sensor_noise=.01, latency_probability=.2,
        push_probability=.02, push_velocity=.3,
        terminal_tilt=.65, terminal_height=.085,
        reward_scales=config_dict.create(
            tracking_linear=2., tracking_yaw=1., upright=.3,
            height=-2., vertical_velocity=-.3, roll_pitch_velocity=-.05,
            foot_slip=-.6, air_time=.3,
            # Smallest weight at which lift's marginal reward beats the ring's
            # marginal per-mm cost along the pure-knee lift direction (~.0066/s per
            # mm) -- a weaker weight (.5 realized as 0.047% of total reward) cannot
            # move the gait at all, regardless of target.
            swing_clearance=6., stand_pose=-3., stance_feet=-.4,
            ring_side=-1., ring_bottom=-1., ring_rub=-2.,
            unwanted_contact=-2., action_rate=-.05, torques=-.0002,
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
