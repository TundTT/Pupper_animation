"""Export a saved train_reference/shaped_env checkpoint to the robot contract JSON.

train_reference exports only at the end of a training run. Shipping a policy from an
EARLIER run needs the same export applied to its saved `mjx_params_*`, and critically
with that run's OWN command envelope rather than whatever the current build_config
defaults happen to be -- `lin_vel_x_range` has been retuned several times, so the
default at export time is not necessarily the range the checkpoint was trained on.
Stamping the wrong `command_low`/`command_high` into the contract would let the
deployed controller command speeds the policy never saw.

    .venv/bin/python -m workspace.export_reference_policy \
        --params training_runs/<run>/mjx_params_<ts> --out policy_walk_v2.json \
        --vx_range -0.35 0.35 --vy_range 0.0 0.0 --wz_range -2.0 2.0
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jax
from brax.io import model as brax_model

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / 'Stanford/training/pupperv3-mjx'))
from pupperv3_mjx import export  # noqa: E402

from workspace import gait_env as _ge  # noqa: E402
from workspace import train_reference as tr  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--params', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--vx_range', type=float, nargs=2, required=True,
                   help="The checkpoint's OWN training range, not today's default.")
    p.add_argument('--vy_range', type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument('--wz_range', type=float, nargs=2, default=[-2.0, 2.0])
    p.add_argument('--source_run', default='')
    args = p.parse_args()

    cfg_args = argparse.Namespace(num_timesteps=1, num_envs=1, learning_rate=3e-4)
    CONFIG, _ = tr.build_config(cfg_args)

    params = brax_model.load_params(args.params)
    out = export.convert_params(
        jax.block_until_ready(params),
        activation=CONFIG.policy.activation,
        action_scale=CONFIG.policy.action_scale,
        kp=CONFIG.training.position_control_kp,
        kd=CONFIG.training.dof_damping,
        default_pose=CONFIG.training.default_pose,
        # Hardware limits, not the MJCF envelope (0.1 rad wider per joint) -- the
        # deployed controller clips to these and validate_policy.py rejects a wider one.
        joint_upper_limits=_ge.HARDWARE_POSITION_MAX,
        joint_lower_limits=_ge.HARDWARE_POSITION_MIN,
        use_imu=CONFIG.policy.use_imu,
        observation_history=CONFIG.policy.observation_history,
        maximum_pitch_command=CONFIG.training.maximum_pitch_command,
        maximum_roll_command=CONFIG.training.maximum_roll_command,
        final_activation='tanh',
    )
    out['behavior'] = 'locomotion'
    out['hardware_profile'] = 'leg_position'
    out['robot_contract_schema_version'] = 1
    out['joint_names'] = list(_ge.CANONICAL_JOINTS)
    out['action_types'] = ['position'] * 12
    out['action_scale'] = [float(CONFIG.policy.action_scale)] * 12
    out['observation_layout'] = [
        'body_angular_velocity_xyz[3]', 'projected_gravity_xyz[3]',
        'command_vx_vy_yaw_rate[3]', 'desired_world_z_in_body_xyz[3]',
        'joint_position_minus_default[12]', 'previous_faded_normalized_action[12]']
    out['command_low'] = [args.vx_range[0], args.vy_range[0], args.wz_range[0]]
    out['command_high'] = [args.vx_range[1], args.vy_range[1], args.wz_range[1]]
    out['orientation_command'] = list(CONFIG.training.desired_world_z_in_body_frame)
    out['training_control_dt'] = CONFIG.training.environment_dt
    out['training_model_sha256'] = hashlib.sha256(
        Path(CONFIG.simulation.model_path).read_bytes()).hexdigest()
    try:
        out['source_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(_REPO_ROOT), text=True).strip()
    except Exception:
        out['source_commit'] = 'unknown'
    if args.source_run:
        out['source_run'] = args.source_run

    # indent=2 to match the formatting of the policy JSONs already on
    # robot-code; a single-line dump makes the git diff 54k deletions and
    # unreviewable.
    Path(args.out).write_text(json.dumps(out, indent=2) + '\n')
    print(f'wrote {args.out}')
    print(f"  command_low  {out['command_low']}")
    print(f"  command_high {out['command_high']}")
    print(f"  in_shape {out['in_shape']}  layers {len(out['layers'])}")


if __name__ == '__main__':
    main()
