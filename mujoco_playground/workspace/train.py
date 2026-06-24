"""Train the Pupper V3 leg-lift policy with brax PPO (MJX backend).

Mirrors the providers' pupperv3-mjx training call and mujoco_playground's
`learning/train_jax_ppo.py`: brax `ppo.train` over an MJX env, observation
normalization on, MLP policy. Designed to run on the CUDA workstation (RTX 5090),
not the laptop used for authoring.

Weights & Biases logging is opt-in (`--use_wandb`). When on, training metrics are
logged each eval and a rollout VIDEO is logged that steps the command through the
O-button sequence (stand -> FL -> FR -> BR -> BL), so you can watch the policy lift
each leg. A final video is always rendered to the run's output dir regardless.

Usage (on the workstation, inside the playground venv with jax[cuda12]):
    python -m workspace.train --num_timesteps 100000000 --use_wandb
The trained brax params are saved to workspace/output/<run>/mjx_params; convert
to the robot's neural_controller JSON with workspace/export_policy.py.
"""

import argparse
import functools
import os
from datetime import datetime

# Headless MuJoCo rendering for the rollout videos (matches train_jax_ppo.py).
os.environ.setdefault("MUJOCO_GL", "egl")

import jax  # noqa: E402
import mediapy as media  # noqa: E402
from brax.io import model  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402

from workspace import configs, visualize  # noqa: E402
from workspace.leg_lift_env import PupperLegLiftEnv  # noqa: E402
from workspace.randomize import domain_randomize  # noqa: E402

_ACTIVATIONS = {"swish": jax.nn.swish, "relu": jax.nn.relu, "tanh": jax.nn.tanh}


def main() -> None:
    p = argparse.ArgumentParser(description="Train the Pupper leg-lift policy.")
    p.add_argument("--num_timesteps", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model_path", default=None, help="override Pupper MJX xml path")
    p.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "output"))
    p.add_argument("--use_wandb", action="store_true", help="log metrics + rollout videos to W&B")
    p.add_argument("--wandb_project", default="pupper-leg-lift")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--no_eval_videos", action="store_true", help="skip the per-eval W&B video")
    args = p.parse_args()

    config = configs.get_config()
    if args.num_timesteps is not None:
        config.ppo.num_timesteps = args.num_timesteps
    if args.num_envs is not None:
        config.ppo.num_envs = args.num_envs
    if args.seed is not None:
        config.ppo.seed = args.seed
    # Each episode spans several commanded raise/hold/lower transitions.
    config.ppo.episode_length = config.episode_length

    run_name = f"leg_lift_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=config.to_dict()
        )

    env = PupperLegLiftEnv(config, model_path=args.model_path)
    eval_env = PupperLegLiftEnv(config, model_path=args.model_path)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(config.policy.hidden_layer_sizes),
        activation=_ACTIVATIONS[config.policy.activation],
    )
    ppo_kwargs = dict(config.ppo)
    train_fn = functools.partial(
        ppo.train, **ppo_kwargs, network_factory=network_factory, randomization_fn=domain_randomize
    )

    times = [datetime.now()]

    def progress(step: int, metrics: dict) -> None:
        times.append(datetime.now())
        reward = metrics.get("eval/episode_reward", float("nan"))
        print(f"[{step:>12,}] eval reward={reward:.3f}")
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
            frames, fps = visualize.render(eval_env, inference_fn, jax.random.PRNGKey(0))
            _log_video(step, frames, fps)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: eval video render failed at step {step}: {e!r}")

    print(f"Training leg-lift policy: {configs.NUM_COMMANDS} commands "
          f"({', '.join(configs.COMMAND_STATES)}), episode_length={config.episode_length} steps "
          f"({config.episode_length * config.ctrl_dt:.1f}s)")
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
    frames, fps = visualize.render(eval_env, inference_fn, jax.random.PRNGKey(1))
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
