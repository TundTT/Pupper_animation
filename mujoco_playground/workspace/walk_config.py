"""Walking settings; the XML is authoritative for home pose, gains, and limits."""
from ml_collections import config_dict

def get_config():
    return config_dict.create(
        ctrl_dt=.02, episode_length=600, observation_history=4,
        action_scale=(.35, .5, .5)*4,
        command_low=(-.2, -.1, -.6), command_high=(.4, .1, .6),
        command_hold_steps=150, stand_probability=.2,
        bottom_allowance=.006, side_clearance=.001,
        reset_joint_noise=.01, reset_xy_noise=.01, reset_yaw_noise=.1,
        sensor_noise=.01, latency_probability=.2,
        terminal_tilt=.65, terminal_height=.085,
        reward_scales=config_dict.create(
            tracking_linear=2., tracking_yaw=1., upright=.3,
            height=-2., vertical_velocity=-.3, roll_pitch_velocity=-.05,
            foot_slip=-.3, air_time=.3, stand_pose=-.5,
            ring_side=-2., ring_bottom=-1., ring_rub=-2.,
            unwanted_contact=-2., action_rate=-.05, torques=-.0002,
            joint_limits=-1., termination=-2.,
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
