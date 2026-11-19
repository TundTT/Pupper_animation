"""Render a review video of a trained gait policy over an in-envelope command suite.

`pupperv3_mjx.utils.visualize_policy` defaults to vx=0.5, vy=0.4, wz=1.5, which is
outside this project's trained command envelope, so its videos show the policy being
driven harder than anything it saw in training. This renders the same kind of
sequence but inside the envelope, with a camera that tracks the robot.

  .venv/bin/python -m workspace.render_gait --params <mjx_params_file> --out review.mp4
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import jax
import mediapy as media
import mujoco
import numpy as np
from jax import numpy as jp

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / 'Stanford/training/pupperv3-mjx'))

from brax import envs  # noqa: E402
from brax.io import model as brax_model  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from pupperv3_mjx import utils  # noqa: E402

from workspace import evaluate_gait, gait_env  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    parser.add_argument('--env', choices=['gait', 'reference'], default='reference',
                        help='Must match how the policy was trained. A reference-trained '
                             'policy rendered in gait_env gets mismatched observations '
                             'and looks frozen (and vice versa).')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seconds_per_command', type=float, default=2.5)
    parser.add_argument('--leg_length_min_m', type=float, default=0.056)
    parser.add_argument('--leg_length_max_m', type=float, default=0.060)
    parser.add_argument('--max_vx', type=float, default=0.4)
    parser.add_argument('--max_vy', type=float, default=0.2)
    parser.add_argument('--max_wz', type=float, default=1.0)
    parser.add_argument('--target_clearance_m', type=float, default=0.025)
    parser.add_argument('--target_swing_s', type=float, default=0.18)
    parser.add_argument('--max_stance_s', type=float, default=0.35)
    parser.add_argument('--action_scale', type=float, nargs=12, default=None)
    args = parser.parse_args()

    env, CONFIG = (evaluate_gait.build_reference_eval_env(args) if args.env == 'reference'
                   else evaluate_gait.build_eval_env(args))
    params = brax_model.load_params(args.params)
    network = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        # MUST match training: these policies were trained with
        # normalize_observations=True, so the network needs the running-statistics
        # preprocessor. An identity fn here feeds raw observations to a policy that
        # expects normalized ones, which makes every policy look frozen.
        running_statistics.normalize,
        policy_hidden_layer_sizes=CONFIG.policy.hidden_layer_sizes,
        activation=utils.activation_fn_map(CONFIG.policy.activation))
    inference_fn = ppo_networks.make_inference_fn(network)((params[0], params[1]),
                                                          deterministic=True)
    jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
    jit_policy = jax.jit(inference_fn)

    dt = float(CONFIG.training.environment_dt)
    steps_per_command = int(args.seconds_per_command / dt)
    suite = [(vx, vy, wz, label) for vx, vy, wz, label in evaluate_gait.COMMAND_SUITE]

    mj_model = env.sys.mj_model
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    data = mujoco.MjData(mj_model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 0.75, 120, -12

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    frames = []
    terminated = 0
    for vx, vy, wz, label in suite:
        cmd = jp.array([vx, vy, wz])
        for _ in range(steps_per_command):
            state.info['command'] = cmd
            rng, key = jax.random.split(rng)
            action, _ = jit_policy(state.obs, key)
            state = jit_step(state, action)
            if bool(state.done):
                terminated += 1
                rng, rkey = jax.random.split(rng)
                state = jit_reset(rkey)
            data.qpos[:] = np.asarray(state.pipeline_state.qpos)
            mujoco.mj_forward(mj_model, data)
            cam.lookat[:] = data.xpos[env._torso_idx]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
        print(f'  {label}: cmd=({vx:+.2f},{vy:+.2f},{wz:+.2f})')

    media.write_video(args.out, frames, fps=int(1.0 / dt))
    print(f'wrote {args.out} ({len(frames)} frames, {len(frames)*dt:.1f}s)')
    print(f'terminations (falls) during render: {terminated}')


if __name__ == '__main__':
    main()
