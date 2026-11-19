"""Walking trainer targeting a normal, deployable gait.

Uses `workspace.gait_env.PupperGaitEnv` (deployment-aligned action frame + gait
rewards) with the reference PPO setup, randomizes foot-capsule length per leg, and
exports an RTNeural JSON carrying the full metadata set that
origin/robot-info `robot_info/validate_policy.py` checks.

Run detached (SSH sessions can drop mid-run):
    nohup .venv/bin/python -m workspace.train_gait --run_tag baseline > gait.log 2>&1 &
    disown
"""
import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if subprocess.run(['nvidia-smi'], capture_output=True).returncode != 0:
    raise RuntimeError('No GPU visible to nvidia-smi; this trainer needs a CUDA GPU.')

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_triton_gemm_any=True'

import matplotlib
matplotlib.use('Agg')

import jax
from jax import numpy as jp
import mujoco
import numpy as np
from brax import envs
from brax.io import model as brax_model
from brax.training.agents.ppo import train as ppo, networks as ppo_networks, losses as ppo_losses
from ml_collections import config_dict
import wandb

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / 'Stanford/training/pupperv3-mjx'))
from pupperv3_mjx import domain_randomization, export, utils  # noqa: E402

from workspace import gait_env  # noqa: E402
from workspace.train_reference import (  # noqa: E402
    MODEL_PATH, build_terrain_model, combined_domain_randomize, foot_capsule_frame,
)

CONTRACT_SCHEMA_VERSION = 1


def build_config(args, model_path=MODEL_PATH):
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    foot_frame = foot_capsule_frame(mj_model)

    simulation = config_dict.ConfigDict()
    simulation.model_path = str(model_path)
    simulation.upper_leg_body_names = ['leg_front_r_2', 'leg_front_l_2', 'leg_back_r_2', 'leg_back_l_2']
    simulation.lower_leg_body_names = ['leg_front_r_3', 'leg_front_l_3', 'leg_back_r_3', 'leg_back_l_3']
    simulation.foot_site_names = [
        'leg_front_r_3_foot_site', 'leg_front_l_3_foot_site',
        'leg_back_r_3_foot_site', 'leg_back_l_3_foot_site',
    ]
    simulation.torso_name = 'base_link'
    simulation.foot_radius = 0.012
    # Termination limits stay on the MJCF envelope (0.1 rad wider than hardware);
    # motor targets are separately clipped to the hardware envelope in gait_env, so
    # a target at the hardware limit cannot spuriously terminate the episode.
    simulation.model_joint_lower = mj_model.jnt_range[1:, 0].tolist()
    simulation.model_joint_upper = mj_model.jnt_range[1:, 1].tolist()
    simulation.physics_dt = 0.004

    training = config_dict.ConfigDict()
    training.environment_dt = 0.02
    training.ppo = config_dict.ConfigDict()
    training.ppo.num_timesteps = args.num_timesteps
    training.ppo.episode_length = 500
    training.ppo.num_evals = args.num_evals
    training.ppo.reward_scaling = 1
    training.ppo.normalize_observations = True
    training.ppo.action_repeat = 1
    training.ppo.unroll_length = 20
    training.ppo.num_minibatches = 32
    training.ppo.num_updates_per_batch = 4
    training.ppo.discounting = 0.97
    training.ppo.learning_rate = args.learning_rate
    training.ppo.entropy_cost = args.entropy_cost
    training.ppo.num_envs = args.num_envs
    training.ppo.batch_size = 256

    # Moderate, hardware-plausible envelope. The reference's +-0.75 m/s on a 14 cm
    # tall robot is ~5 body lengths/s and pushes the policy into a frantic gait.
    training.lin_vel_x_range = [-args.max_vx, args.max_vx]
    training.lin_vel_y_range = [-args.max_vy, args.max_vy]
    training.ang_vel_yaw_range = [-args.max_wz, args.max_wz]
    training.command_resample_steps = 100
    training.stand_probability = 0.1
    training.stand_still_command_threshold = 0.05
    training.maximum_pitch_command = 0.0
    training.maximum_roll_command = 0.0
    training.desired_world_z_in_body_frame = (0.0, 0.0, 1.0)

    training.terminal_body_z = 0.05
    training.terminal_body_angle = 0.70
    training.early_termination_step_threshold = 250

    training.dof_damping = 0.25
    training.position_control_kp = 5.0
    training.home_pose = gait_env.HOMED_POSITION.tolist()
    training.action_scale = (args.action_scale or gait_env.ACTION_SCALE.tolist())
    training.desired_abduction_angles = [0.0] * 4

    training.kick_probability = 0.04
    training.kick_vel = 0.10
    training.angular_velocity_noise = 0.1
    training.gravity_noise = 0.05
    training.motor_angle_noise = 0.05
    training.last_action_noise = 0.01
    training.position_control_kp_multiplier_range = (0.6, 1.1)
    training.position_control_kd_multiplier_range = (0.8, 1.5)
    training.latency_distribution = [0.2, 0.8]
    training.imu_latency_distribution = [0.5, 0.5]
    training.body_com_x_shift_range = (-0.02, 0.03)
    training.body_com_y_shift_range = (-0.005, 0.005)
    training.body_com_z_shift_range = (-0.005, 0.005)
    training.body_mass_scale_range = (0.9, 1.3)
    training.body_inertia_scale_range = (0.9, 1.3)
    training.friction_range = (0.6, 1.4)
    training.leg_length_range_m = (args.leg_length_min_m, args.leg_length_max_m)

    # Gait shaping targets. Small quadrupeds trot at roughly 2-3 Hz stride, i.e. a
    # 0.33-0.5 s period; with a ~0.5 duty factor that is ~0.18 s swing and no more
    # than ~0.35 s of continuous stance while a move command is active.
    training.target_clearance_m = args.target_clearance_m
    training.target_knee_height_m = args.target_knee_height_m
    training.variance_base = args.variance_base
    training.speed_gain = args.speed_gain
    training.variance_cap = args.variance_cap
    training.tip_allowance_deg = args.tip_allowance_deg
    training.tip_scale_deg = args.tip_scale_deg
    training.target_swing_s = args.target_swing_s
    training.max_stance_s = args.max_stance_s
    training.touchdown_speed_allowance = 0.15

    policy = config_dict.ConfigDict()
    policy.use_imu = True
    policy.observation_history = 4
    policy.hidden_layer_sizes = (256, 128, 128, 128)
    policy.activation = 'elu'

    reward = config_dict.ConfigDict()
    reward.rewards = config_dict.ConfigDict()
    reward.rewards.scales = config_dict.ConfigDict()
    reward.rewards.scales.tracking_lin_vel = args.w_track
    reward.rewards.scales.tracking_ang_vel = args.w_track_ang
    # Lowered from the reference 0.5: an upright bonus is fully earned by standing
    # still, so it inflates the do-nothing baseline this trainer is fighting.
    reward.rewards.scales.tracking_orientation = args.w_track_orient
    reward.rewards.scales.lin_vel_z = -0.1
    reward.rewards.scales.ang_vel_xy = -0.002
    reward.rewards.scales.orientation = -0.0
    reward.rewards.scales.torques = -0.025
    reward.rewards.scales.joint_acceleration = -1e-6
    reward.rewards.scales.mechanical_work = 0
    reward.rewards.scales.action_rate = -0.1
    reward.rewards.scales.stand_still = -0.00
    reward.rewards.scales.stand_still_joint_velocity = -0.2
    reward.rewards.scales.abduction_angle = -0.01
    reward.rewards.scales.termination = -100.0
    reward.rewards.scales.knee_collision = -10.0
    reward.rewards.scales.body_collision = -0.5
    reward.rewards.scales.feet_air_time = args.w_air_time
    reward.rewards.scales.foot_slip = -0.2
    reward.rewards.scales.foot_clearance = args.w_clearance
    reward.rewards.scales.stance_cap = args.w_stance_cap
    reward.rewards.scales.touchdown_impact = args.w_touchdown
    reward.rewards.scales.knee_clearance = args.w_knee
    reward.rewards.scales.flight = args.w_flight
    reward.rewards.scales.tip_vertical = args.w_tip
    # The reference's 0.25 was tuned for ~+-1 m/s commands: with exp(-err^2/sigma),
    # standing still under a 0.25 m/s command still scores 0.78 of the maximum, so
    # there is almost no gradient toward actually moving. Scale it to this command
    # envelope (sigma ~ 0.25*(max_vx/1.0)^2) so full-speed error scores near zero.
    reward.rewards.tracking_sigma = args.tracking_sigma

    config = config_dict.ConfigDict()
    config.simulation = simulation
    config.training = training
    config.policy = policy
    config.reward = reward
    return config_dict.FrozenConfigDict(config), foot_frame


def make_env_kwargs(CONFIG):
    return dict(
        path=CONFIG.simulation.model_path,
        action_scale=jp.array(CONFIG.training.action_scale),
        observation_history=CONFIG.policy.observation_history,
        joint_lower_limits=jp.array(CONFIG.simulation.model_joint_lower),
        joint_upper_limits=jp.array(CONFIG.simulation.model_joint_upper),
        dof_damping=CONFIG.training.dof_damping,
        position_control_kp=CONFIG.training.position_control_kp,
        foot_site_names=CONFIG.simulation.foot_site_names,
        torso_name=CONFIG.simulation.torso_name,
        upper_leg_body_names=CONFIG.simulation.upper_leg_body_names,
        lower_leg_body_names=CONFIG.simulation.lower_leg_body_names,
        resample_velocity_step=CONFIG.training.command_resample_steps,
        linear_velocity_x_range=CONFIG.training.lin_vel_x_range,
        linear_velocity_y_range=CONFIG.training.lin_vel_y_range,
        angular_velocity_range=CONFIG.training.ang_vel_yaw_range,
        zero_command_probability=0.0,  # gait_env.sample_command handles stand mode
        stand_still_command_threshold=CONFIG.training.stand_still_command_threshold,
        maximum_pitch_command=CONFIG.training.maximum_pitch_command,
        maximum_roll_command=CONFIG.training.maximum_roll_command,
        start_position_config=domain_randomization.StartPositionRandomization(
            x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, z_min=0.148, z_max=0.175),
        desired_abduction_angles=jp.array(CONFIG.training.desired_abduction_angles),
        reward_config=CONFIG.reward,
        angular_velocity_noise=CONFIG.training.angular_velocity_noise,
        gravity_noise=CONFIG.training.gravity_noise,
        motor_angle_noise=CONFIG.training.motor_angle_noise,
        last_action_noise=CONFIG.training.last_action_noise,
        kick_vel=CONFIG.training.kick_vel,
        kick_probability=CONFIG.training.kick_probability,
        terminal_body_z=CONFIG.training.terminal_body_z,
        early_termination_step_threshold=CONFIG.training.early_termination_step_threshold,
        terminal_body_angle=CONFIG.training.terminal_body_angle,
        foot_radius=CONFIG.simulation.foot_radius,
        environment_timestep=CONFIG.training.environment_dt,
        physics_timestep=CONFIG.simulation.physics_dt,
        latency_distribution=jp.array(CONFIG.training.latency_distribution),
        imu_latency_distribution=jp.array(CONFIG.training.imu_latency_distribution),
        desired_world_z_in_body_frame=jp.array(CONFIG.training.desired_world_z_in_body_frame),
        use_imu=CONFIG.policy.use_imu,
        home_pose=np.array(CONFIG.training.home_pose),
        target_clearance_m=CONFIG.training.target_clearance_m,
        target_knee_height_m=CONFIG.training.target_knee_height_m,
        variance_base=CONFIG.training.variance_base,
        speed_gain=CONFIG.training.speed_gain,
        variance_cap=CONFIG.training.variance_cap,
        tip_allowance_deg=CONFIG.training.tip_allowance_deg,
        tip_scale_deg=CONFIG.training.tip_scale_deg,
        target_swing_s=CONFIG.training.target_swing_s,
        max_stance_s=CONFIG.training.max_stance_s,
        touchdown_speed_allowance=CONFIG.training.touchdown_speed_allowance,
        command_resample_steps=CONFIG.training.command_resample_steps,
        stand_probability=CONFIG.training.stand_probability,
    )


def export_policy(params, CONFIG, model_path, out_path):
    """RTNeural export plus the metadata robot_info/validate_policy.py requires."""
    payload = export.convert_params(
        jax.block_until_ready(params),
        activation=CONFIG.policy.activation,
        action_scale=list(CONFIG.training.action_scale),
        kp=CONFIG.training.position_control_kp,
        kd=CONFIG.training.dof_damping,
        default_pose=np.array(CONFIG.training.home_pose),
        joint_upper_limits=gait_env.HARDWARE_POSITION_MAX,
        joint_lower_limits=gait_env.HARDWARE_POSITION_MIN,
        use_imu=CONFIG.policy.use_imu,
        observation_history=CONFIG.policy.observation_history,
        maximum_pitch_command=CONFIG.training.maximum_pitch_command,
        maximum_roll_command=CONFIG.training.maximum_roll_command,
        final_activation='tanh',
    )
    payload['action_scale'] = list(CONFIG.training.action_scale)
    payload['behavior'] = 'locomotion'
    payload['hardware_profile'] = 'leg_position'
    payload['robot_contract_schema_version'] = CONTRACT_SCHEMA_VERSION
    payload['joint_names'] = list(gait_env.CANONICAL_JOINTS)
    payload['action_types'] = ['position'] * 12
    payload['observation_layout'] = [
        'body_angular_velocity_xyz[3]',
        'projected_gravity_xyz[3]',
        'command_vx_vy_yaw_rate[3]',
        'desired_world_z_in_body_xyz[3]',
        'joint_position_minus_default[12]',
        'previous_faded_normalized_action[12]',
    ]
    payload['command_low'] = [CONFIG.training.lin_vel_x_range[0],
                              CONFIG.training.lin_vel_y_range[0],
                              CONFIG.training.ang_vel_yaw_range[0]]
    payload['command_high'] = [CONFIG.training.lin_vel_x_range[1],
                               CONFIG.training.lin_vel_y_range[1],
                               CONFIG.training.ang_vel_yaw_range[1]]
    payload['orientation_command'] = list(CONFIG.training.desired_world_z_in_body_frame)
    payload['training_control_dt'] = CONFIG.training.environment_dt
    payload['training_model_sha256'] = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    try:
        payload['source_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(_REPO_ROOT)).decode().strip()
    except subprocess.CalledProcessError:
        payload['source_commit'] = 'unknown'
    Path(out_path).write_text(json.dumps(payload))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_timesteps', type=int, default=300_000_000)
    parser.add_argument('--num_envs', type=int, default=8192)
    parser.add_argument('--num_evals', type=int, default=11)
    parser.add_argument('--learning_rate', type=float, default=3.0e-4)
    parser.add_argument('--entropy_cost', type=float, default=1e-2)
    parser.add_argument('--wandb_project', default='Walk')
    parser.add_argument('--wandb_entity', default='QuadMorph')
    parser.add_argument('--run_tag', default='gait')
    parser.add_argument('--flat_ground', action='store_true', default=True)
    parser.add_argument('--terrain', dest='flat_ground', action='store_false',
                         help='Enable heightfield terrain (off by default).')
    parser.add_argument('--terrain_seed', type=int, default=0)
    parser.add_argument('--leg_length_min_m', type=float, default=0.056)
    parser.add_argument('--leg_length_max_m', type=float, default=0.060)
    parser.add_argument('--max_vx', type=float, default=0.4)
    parser.add_argument('--max_vy', type=float, default=0.2)
    parser.add_argument('--max_wz', type=float, default=1.0)
    parser.add_argument('--target_clearance_m', type=float, default=0.025)
    # Knee sits at 58.4 mm standing; ~16 mm of lift is reachable via joint 1.
    parser.add_argument('--target_knee_height_m', type=float, default=0.070)
    parser.add_argument('--target_swing_s', type=float, default=0.18)
    parser.add_argument('--max_stance_s', type=float, default=0.35)
    parser.add_argument('--w_clearance', type=float, default=-2.0)
    parser.add_argument('--w_stance_cap', type=float, default=-0.15)
    parser.add_argument('--w_touchdown', type=float, default=-3.0)
    parser.add_argument('--w_air_time', type=float, default=2.0)
    parser.add_argument('--tracking_sigma', type=float, default=0.05)
    parser.add_argument('--w_knee', type=float, default=-3.0)
    parser.add_argument('--w_flight', type=float, default=-1.0)
    parser.add_argument('--w_tip', type=float, default=-1.5)
    # Home stance is 6.3 deg off vertical; joint 3 alone reaches 49 deg at 0.8 rad.
    parser.add_argument('--tip_allowance_deg', type=float, default=15.0)
    parser.add_argument('--tip_scale_deg', type=float, default=45.0)
    parser.add_argument('--w_track', type=float, default=1.5)
    parser.add_argument('--w_track_ang', type=float, default=0.8)
    parser.add_argument('--w_track_orient', type=float, default=0.2)
    parser.add_argument('--variance_base', type=float, default=0.0025)
    parser.add_argument('--speed_gain', type=float, default=0.55)
    # variance_cap >= variance_base with speed_gain 0 reproduces the reference's
    # fixed-sigma tracking, for curriculum stage 1 (learn to translate at all).
    parser.add_argument('--variance_cap', type=float, default=0.07)
    parser.add_argument('--action_scale', type=float, nargs=12, default=None)
    parser.add_argument('--warm_start_params_path', default=None)
    args = parser.parse_args()

    jax.config.update('jax_default_matmul_precision', 'high')

    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_folder = _REPO_ROOT / 'training_runs' / f'gait_{args.run_tag}_{stamp}'
    output_folder.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_PATH
    if not args.flat_ground:
        model_path = output_folder / 'model_with_terrain.xml'
        build_terrain_model(MODEL_PATH, model_path, seed=args.terrain_seed)

    CONFIG, foot_frame = build_config(args, model_path=model_path)
    env_kwargs = make_env_kwargs(CONFIG)

    envs.register_environment('pupper_gait', gait_env.PupperGaitEnv)
    env = envs.get_environment('pupper_gait', **env_kwargs)
    eval_env = envs.get_environment('pupper_gait', **env_kwargs)
    jit_reset, jit_step = jax.jit(eval_env.reset), jax.jit(eval_env.step)

    wandb.init(
        entity=args.wandb_entity, project=args.wandb_project,
        config={**CONFIG.to_dict(), 'run_tag': args.run_tag,
                'leg_length_range_m': list(CONFIG.training.leg_length_range_m)},
        name=f'gait-{args.run_tag}-{stamp}', save_code=True,
        settings={'_service_wait': 90, 'init_timeout': 90},
    )

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=CONFIG.policy.hidden_layer_sizes,
        activation=utils.activation_fn_map(CONFIG.policy.activation),
    )
    train_fn = functools.partial(
        ppo.train,
        **CONFIG.training.ppo.to_dict(),
        network_factory=make_networks_factory,
        randomization_fn=functools.partial(
            combined_domain_randomize,
            leg_length_range_m=tuple(CONFIG.training.leg_length_range_m),
            foot_frame=foot_frame,
            friction_range=CONFIG.training.friction_range,
            kp_multiplier_range=CONFIG.training.position_control_kp_multiplier_range,
            kd_multiplier_range=CONFIG.training.position_control_kd_multiplier_range,
            body_com_x_shift_range=CONFIG.training.body_com_x_shift_range,
            body_com_y_shift_range=CONFIG.training.body_com_y_shift_range,
            body_com_z_shift_range=CONFIG.training.body_com_z_shift_range,
            body_mass_scale_range=CONFIG.training.body_mass_scale_range,
            body_inertia_scale_range=CONFIG.training.body_inertia_scale_range,
        ),
        seed=28,
    )

    x_data, y_data, ydataerr, times = [], [], [], [datetime.now()]

    def policy_params_fn(current_step, make_policy, params):
        normalizer, policy_params, value_params = params
        compat = (normalizer, ppo_losses.PPONetworkParams(policy=policy_params, value=value_params))
        # visualize_policy defaults to vx=0.5, vy=0.4, wz=1.5, which is OUTSIDE this
        # trainer's command envelope (vy especially, at 2x the trained maximum), so the
        # review videos were showing the policy commanded beyond anything it trained on.
        # Drive it inside the envelope instead.
        utils.visualize_policy(
            current_step=current_step, make_policy=make_policy, params=compat,
            eval_env=eval_env, jit_step=jit_step, jit_reset=jit_reset,
            output_folder=str(output_folder),
            vx=0.9 * CONFIG.training.lin_vel_x_range[1],
            vy=0.9 * CONFIG.training.lin_vel_y_range[1],
            wz=0.8 * CONFIG.training.ang_vel_yaw_range[1])
        utils.save_checkpoint(
            current_step=current_step, make_policy=make_policy, params=compat,
            checkpoint_path=output_folder)
        brax_model.save_params(str(output_folder / f'mjx_params_{current_step}'), params)

    warm = {}
    if args.warm_start_params_path:
        warm['restore_params'] = brax_model.load_params(args.warm_start_params_path)
        print(f'Warm-starting from {args.warm_start_params_path}')

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=functools.partial(
            utils.progress, times=times, x_data=x_data, y_data=y_data, ydataerr=ydataerr,
            num_timesteps=CONFIG.training.ppo.num_timesteps, min_y=0, max_y=40),
        eval_env=eval_env, policy_params_fn=policy_params_fn, **warm)

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

    params_path = output_folder / 'mjx_params_final'
    brax_model.save_params(str(params_path), params)
    policy_path = output_folder / f'policy_{args.run_tag}.json'
    export_policy(params, CONFIG, model_path, policy_path)
    wandb.log_model(path=str(policy_path), name=f'policy_{args.run_tag}_{stamp}.json')
    wandb.finish()
    print(f'Exported policy: {policy_path}')
    print(f'Final params: {params_path}')


if __name__ == '__main__':
    main()
