"""Train the Pupper V3 WHEELED locomotion policy with brax PPO (MJX backend).

Mirrors the providers' pupperv3-mjx training call and mujoco_playground's
`learning/train_jax_ppo.py`: brax `ppo.train` over an MJX env, observation
normalization on, MLP policy. Designed to run on the CUDA workstation.

Trains `wheel_env.PupperWheelEnv` against `configs.get_wheel_config()`: drive at
a commanded body velocity (vx, vy, yaw rate) on four wheels while holding the
splayed stance upright. The leg-lift task this branch forked from lives on
`master`; `leg_lift_env.py` is retained here as a reference but is NOT what this
script trains.

Weights & Biases logging is opt-in (`--use_wandb`). When on, training metrics are
logged each eval along with a rollout VIDEO stepping through a fixed command
showcase (stop / forward / arcs / spin / reverse), so runs are comparable
frame-for-frame. A final video is always rendered to the run's output dir.

IMPORTANT -- run with the venv python directly, NOT `uv run`:
    cd mujoco_playground
    .venv/bin/python -m workspace.train --num_timesteps 100000000 --use_wandb
`uv run` re-syncs jax/jaxlib to uv.lock's 0.6.2, which mismatches the installed
jax_cuda12_plugin 0.5.0, silently disables the GPUs and segfaults on model load.

For a long run, detach it so an SSH drop cannot kill it:
    nohup .venv/bin/python -m workspace.train ... > train.log 2>&1 & disown

The trained brax params are saved to workspace/output/<run>/mjx_params; convert
to the robot's neural_controller JSON with workspace/export_policy.py.
"""

import argparse
import functools
import os
from datetime import datetime

# NOTE: headless rendering (MUJOCO_GL=egl) is set in workspace/__init__.py, which
# runs before this module and before `mujoco` is first imported. Setting it here
# would be too late -- see the comment there.

import jax  # noqa: E402
import mediapy as media  # noqa: E402
from brax.io import model  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402

from workspace import configs, wheel_visualize  # noqa: E402
from workspace.randomize import domain_randomize_wheeled  # noqa: E402
from workspace.wheel_env import PupperWheelEnv  # noqa: E402

_ACTIVATIONS = {"swish": jax.nn.swish, "relu": jax.nn.relu, "tanh": jax.nn.tanh, "elu": jax.nn.elu}


def main() -> None:
    p = argparse.ArgumentParser(description="Train the Pupper wheeled-locomotion policy.")
    p.add_argument("--num_timesteps", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model_path", default=None, help="override Pupper MJX xml path")
    p.add_argument(
        "--init_params", default=None,
        help="warm-start from a previous run's saved brax params (path to an output/<run>/mjx_params dir)",
    )
    p.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "output"))
    p.add_argument("--use_wandb", action="store_true", help="log metrics + rollout videos to W&B")
    p.add_argument("--wandb_project", default="pupper-wheel")
    # Default to the QuadMorph team so runs and their videos land somewhere the whole
    # team can see, not in whoever-launched-it's personal workspace. Slug is
    # case-sensitive and was read back from the W&B API, not guessed from the UI
    # label. Override with --wandb_entity for a personal scratch run.
    p.add_argument("--wandb_entity", default="QuadMorph")
    p.add_argument("--no_eval_videos", action="store_true", help="skip the per-eval W&B video")
    p.add_argument(
        "--com_x_shift_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="override randomize.py's body_com_x_shift_range for this run, without touching its committed default",
    )
    p.add_argument(
        "--com_y_shift_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="override randomize.py's body_com_y_shift_range for this run, without touching its committed default",
    )
    args = p.parse_args()

    config = configs.get_wheel_config()
    if args.num_timesteps is not None:
        config.ppo.num_timesteps = args.num_timesteps
    if args.num_envs is not None:
        config.ppo.num_envs = args.num_envs
    if args.seed is not None:
        config.ppo.seed = args.seed
    # Each episode spans several commanded velocity changes.
    config.ppo.episode_length = config.episode_length

    run_name = f"wheel_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=config.to_dict()
        )

    env = PupperWheelEnv(config, model_path=args.model_path)
    eval_env = PupperWheelEnv(config, model_path=args.model_path)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(config.policy.hidden_layer_sizes),
        activation=_ACTIVATIONS[config.policy.activation],
    )
    restore_params = None
    if args.init_params is not None:
        print(f"Warm-starting from {args.init_params}")
        restore_params = model.load_params(args.init_params)

    randomization_fn = domain_randomize_wheeled
    com_overrides = {}
    if args.com_x_shift_range is not None:
        com_overrides["body_com_x_shift_range"] = tuple(args.com_x_shift_range)
    if args.com_y_shift_range is not None:
        com_overrides["body_com_y_shift_range"] = tuple(args.com_y_shift_range)
    if com_overrides:
        print(f"Overriding CoM shift range(s) for this run only: {com_overrides}")
        randomization_fn = functools.partial(domain_randomize_wheeled, **com_overrides)

    ppo_kwargs = dict(config.ppo)
    train_fn = functools.partial(
        ppo.train, **ppo_kwargs, network_factory=network_factory, randomization_fn=randomization_fn,
        restore_params=restore_params,
    )

    times = [datetime.now()]

    def progress(step: int, metrics: dict) -> None:
        times.append(datetime.now())
        reward = metrics.get("eval/episode_reward", float("nan"))
        # Print the tracking errors alongside the reward, not just the reward: a
        # policy that simply parks and never moves still banks every posture term,
        # so vel_err is the number that says whether it is doing the TASK.
        # brax reports episode metrics as SUMS, so divide by the episode length.
        n = metrics.get("eval/avg_episode_length", 0.0) or 1.0
        d = lambda k: metrics.get(f"eval/episode_{k}", float("nan")) / n
        print(
            f"[{step:>12,}] reward={reward:8.2f} eplen={n:5.0f} "
            f"vel_err={d('lin_vel_error'):.4f} yaw_err={d('ang_vel_error'):.4f} "
            f"tilt={d('tilt_deg'):5.2f}deg torso_z={d('torso_z'):.4f}m "
            f"contacts={d('wheel_contacts'):.2f} "
            f"| trk_lin={d('tracking_lin_vel'):.3f} trk_ang={d('tracking_ang_vel'):.3f} "
            f"stance={d('stance_pose'):.3f} still={d('stand_still'):.3f}",
            flush=True,  # unbuffered: this goes to a redirected log that is tailed live
        )
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

    def _log_video(step: int, frames, fps: int) -> None:
        path = os.path.join(out_dir, f"rollout_step_{step}.mp4")
        media.write_video(path, frames, fps=fps)
        print(f"  wrote video -> {path}")
        if wandb_run is not None:
            import wandb

            wandb_run.log({"eval/video": wandb.Video(path, fps=fps, format="mp4")}, step=step)

    def policy_params_fn(step: int, make_policy, params) -> None:
        # Called by brax after each eval. params[1] is PPONetworkParams here, so the
        # policy params are params[1].policy. Render best-effort: a video hiccup must
        # not kill a multi-hour run, but we surface it loudly (no silent swallow).
        if args.no_eval_videos:
            return
        try:
            # params = (normalizer_params, policy_params, value_params); make_policy
            # uses params[0:2], so passing the full tuple is correct and matches the
            # final-video call below.
            inference_fn = make_policy(params, deterministic=True)
            frames, fps = wheel_visualize.render(eval_env, inference_fn, jax.random.PRNGKey(0))
            _log_video(step, frames, fps)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: eval video render failed at step {step}: {e!r}")

    print(
        f"Training WHEELED locomotion policy: "
        f"vx={tuple(config.lin_vel_x_range)} vy={tuple(config.lin_vel_y_range)} "
        f"yaw={tuple(config.ang_vel_yaw_range)}, wheel cap "
        f"{configs.WHEEL_MAX_LINEAR_SPEED:.2f} m/s ({configs.WHEEL_MAX_SPEED:.2f} rad/s), "
        f"episode_length={config.episode_length} steps "
        f"({config.episode_length * config.ctrl_dt:.1f}s)"
    )
    make_inference_fn, params, _ = train_fn(
        environment=env, eval_env=eval_env, progress_fn=progress, policy_params_fn=policy_params_fn
    )

    params_path = os.path.join(out_dir, "mjx_params")
    model.save_params(params_path, params)

    print(f"time to jit:   {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"Saved brax params -> {params_path}")

    # Always render a final video from the trained policy (final params[1] is the
    # policy params dict, so make_inference_fn(params) is the right call here).
    print("Rendering final rollout video...")
    inference_fn = make_inference_fn(params, deterministic=True)
    frames, fps = wheel_visualize.render(eval_env, inference_fn, jax.random.PRNGKey(1))
    final_path = os.path.join(out_dir, "rollout_final.mp4")
    media.write_video(final_path, frames, fps=fps)
    print(f"Final video -> {final_path}")
    if wandb_run is not None:
        import wandb

        wandb_run.log({"eval/video_final": wandb.Video(final_path, fps=fps, format="mp4")})
        wandb_run.finish()

    print(f"Next: python -m workspace.export_policy --params {params_path}")


if __name__ == "__main__":
    main()
