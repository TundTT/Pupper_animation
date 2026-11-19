"""Train Pupper wheel locomotion or QuadMorph alignment with Brax PPO (MJX).

Use --task wheel-align for full multi-leg lift/rotate/lower/hold sequences. This
selects get_wheel_align_config(), complete task-state episode resets, and the
existing W&B/video path with the QuadMorph alignment project enabled by default.

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
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

# NOTE: headless rendering (MUJOCO_GL=egl) is set in workspace/__init__.py, which
# runs before this module and before `mujoco` is first imported. Setting it here
# would be too late -- see the comment there.
import jax  # noqa: E402
import mediapy as media  # noqa: E402
from brax.io import model  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402

from workspace import configs, wheel_align_visualize, wheel_visualize  # noqa: E402
from workspace.randomize import domain_randomize_wheeled  # noqa: E402
from workspace.wheel_align_env import PupperWheelAlignEnv, domain_randomize_wheel_align
from workspace.wheel_align_direct_env import PupperWheelAlignDirectEnv
from workspace.wheel_align_evaluate import make_evaluator  # noqa: E402
from workspace.wheel_align_training import make_alignment_networks, wrap_for_alignment  # noqa: E402
from workspace.wheel_env import PupperWheelEnv
from workspace import supervised_align_ppo

_ACTIVATIONS = {
    "swish": jax.nn.swish,
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "elu": jax.nn.elu,
}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train the Pupper wheeled-locomotion policy."
    )
    p.add_argument(
        "--task",
        choices=("wheel", "wheel-align", "wheel-align-direct"),
        default="wheel",
    )
    p.add_argument("--num_timesteps", type=int, default=None)
    p.add_argument("--num_evals", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument(
        "--audit_episodes",
        type=int,
        default=16,
        help="full-sequence audit size, also used for direct clone readiness",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model_path", default=None, help="override Pupper MJX xml path")
    p.add_argument(
        "--reference_path",
        default=None,
        help="direct alignment's phase-pose demonstration bank",
    )
    p.add_argument(
        "--supervision_path",
        default=None,
        help="phase-balanced direct-action demonstrations for protected PPO",
    )
    p.add_argument(
        "--init_params",
        default=None,
        help="warm-start from a previous run's saved brax params (path to an output/<run>/mjx_params dir)",
    )
    p.add_argument(
        "--output_dir", default=os.path.join(os.path.dirname(__file__), "output")
    )
    p.add_argument(
        "--use_wandb", action="store_true", help="log metrics + rollout videos to W&B"
    )
    p.add_argument("--wandb_project", default=None)
    # Default to the QuadMorph team so runs and their videos land somewhere the whole
    # team can see, not in whoever-launched-it's personal workspace. Slug is
    # case-sensitive and was read back from the W&B API, not guessed from the UI
    # label. Override with --wandb_entity for a personal scratch run.
    p.add_argument("--wandb_entity", default="QuadMorph")
    p.add_argument(
        "--privileged_critic",
        action="store_true",
        help="training-only phase/contact/drift value inputs; actor remains sensor-only",
    )
    p.add_argument('--encoder_hold', action='store_true',
                   help='encoder-only persistent hold after five settled samples; learned lift and rotation')
    p.add_argument('--encoder_gate', action='store_true',
                   help='8 learned position outputs and4 learned enables for encoder-controlled rotation')
    p.add_argument('--gate_capture_goal', action='store_true',
                   help='keep the final wheel goal once the slew reference reaches it')
    p.add_argument('--reference_reset_path', type=str)
    p.add_argument('--reference_reset_probability', type=float, default=0.0)
    p.add_argument(
        "--direct_clearance_upper",
        type=float,
        help="direct-only upper edge of a small clearance reward band; success gates unchanged",
    )
    p.add_argument(
        "--direct_hold_weight",
        type=float,
        help="direct-only negative held-angle/speed penalty weight",
    )
    p.add_argument(
        "--no_eval_videos", action="store_true", help="skip the per-eval W&B video"
    )
    p.add_argument(
        "--com_x_shift_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="override randomize.py's body_com_x_shift_range for this run, without touching its committed default",
    )
    p.add_argument(
        "--com_y_shift_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="override randomize.py's body_com_y_shift_range for this run, without touching its committed default",
    )
    args = p.parse_args()
    if args.encoder_gate:
        args.encoder_hold = True
    if args.gate_capture_goal and not args.encoder_gate:
        p.error('--gate_capture_goal requires --encoder_gate')
    if args.audit_episodes < 16:
        p.error("--audit_episodes must be at least 16")

    direct = args.task == "wheel-align-direct"
    align = args.task in ("wheel-align", "wheel-align-direct")
    config = (
        configs.get_wheel_align_direct_config()
        if direct
        else configs.get_wheel_align_config()
        if align
        else configs.get_wheel_config()
    )
    env_type = (
        PupperWheelAlignDirectEnv
        if direct
        else PupperWheelAlignEnv
        if align
        else PupperWheelEnv
    )
    if args.direct_clearance_upper is not None or args.direct_hold_weight is not None:
        if not direct:
            p.error("Direct reward overrides require --task wheel-align-direct")
    if args.privileged_critic and not direct:
        p.error("Task-state critic requires --task wheel-align-direct")
    if (args.encoder_hold or args.reference_reset_probability) and not direct:
        p.error('Encoder holding and reference resets require wheel-align-direct')
    if args.reference_reset_probability and (not args.encoder_hold or not args.reference_reset_path):
        p.error('Reference resets require --encoder_hold and --reference_reset_path')
    if args.encoder_hold:
        from workspace.wheel_align_refine_env import PupperWheelAlignRefineEnv
        env_type = PupperWheelAlignRefineEnv
        config.direct.encoder_hold = True
        config.direct.control_convention = 'default_plus_scaled_position_with_encoder_hold'
        config.direct.reference_reset_path = args.reference_reset_path or ''
        config.direct.reference_reset_probability = args.reference_reset_probability
        if args.encoder_gate:
            from workspace.wheel_align_gate_env import PupperWheelAlignGateEnv
            env_type = PupperWheelAlignGateEnv
            config.direct.encoder_gate = True
            config.direct.capture_finished_reference = args.gate_capture_goal
            config.direct.control_convention = 'eight_position_targets_four_wheel_enables'
            config.direct.policy_output_types = ('position_offset','position_offset','rotation_enable') * 4
    if direct:
        config.direct.privileged_critic = args.privileged_critic
    if args.direct_clearance_upper is not None:
        if args.direct_clearance_upper <= config.align.clearance_min:
            p.error("Clearance upper edge must exceed the existing unloading threshold")
        config.direct.lift_clearance_upper = args.direct_clearance_upper
    if args.direct_hold_weight is not None:
        if args.direct_hold_weight >= 0:
            p.error("Held-angle weight must be negative")
        config.reward_config.scales.hold = args.direct_hold_weight
    if direct:
        if args.init_params is None:
            p.error(
                "wheel-align-direct requires --init_params from teacher behavior cloning"
            )
        with open(os.path.join(os.path.dirname(args.init_params), "config.json")) as f:
            initialization_config = json.load(f)
        if initialization_config['direct'].get('encoder_gate', False) and not args.encoder_gate:
            p.error('This checkpoint requires --encoder_gate; its wheel outputs are enable signals')
        if initialization_config['direct'].get('capture_finished_reference', False) and not args.gate_capture_goal:
            p.error('This checkpoint requires --gate_capture_goal')
        config.policy.hidden_layer_sizes = tuple(
            initialization_config["policy"]["hidden_layer_sizes"]
        )
        config.direct.observation_frame_size = initialization_config["direct"][
            "observation_frame_size"
        ]
        if "history_taps" in initialization_config["direct"]:
            config.direct.history_taps = tuple(
                initialization_config["direct"]["history_taps"]
            )
        config.direct.reference_path = initialization_config["direct"]["reference_path"]
        for key in ("motor_angle_noise", "gravity_noise", "angular_velocity_noise"):
            config.dr[key] = initialization_config["dr"][key]
        config.direct.supervision_path = (
            args.supervision_path or initialization_config["direct"]["supervision_path"]
        )
        if config.direct.observation_frame_size == 64:
            for key in ("position_velocity_scale", "position_velocity_noise"):
                config.direct[key] = initialization_config["direct"][key]
        if args.reference_path is not None:
            config.direct.reference_path = args.reference_path
    visualizer = wheel_align_visualize if align else wheel_visualize
    if align:
        args.use_wandb = True
    if args.wandb_project is None:
        args.wandb_project = (
            "wheel-leg lift and align triangle base" if align else "pupper-wheel"
        )
    if args.num_timesteps is not None:
        config.ppo.num_timesteps = args.num_timesteps
    if args.num_evals is not None:
        if args.num_evals < 2:
            p.error("--num_evals must be at least two")
        config.ppo.num_evals = args.num_evals
    if args.num_envs is not None:
        config.ppo.num_envs = args.num_envs
    if direct:
        if config.ppo.num_envs % config.ppo.num_minibatches:
            p.error("Direct PPO requires num_envs divisible by num_minibatches")
        config.ppo.batch_size = config.ppo.num_envs // config.ppo.num_minibatches
    if args.seed is not None:
        config.ppo.seed = args.seed
    # Each episode spans several commanded velocity changes.
    config.ppo.episode_length = config.episode_length

    run_name = f"{args.task}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    if direct:
        source_dir = os.path.join(out_dir, "source")
        os.makedirs(source_dir)
        hashes = {}
        for filename in (
            "train.py",
            "configs.py",
            "supervised_align_ppo.py",
            "wheel_align_direct_env.py",
            "wheel_align_critic.py",
            "wheel_align_refine_env.py",
            "wheel_align_gate_env.py",
            "wheel_align_env.py",
            "wheel_env.py",
            "randomize.py",
            "wheel_align_training.py",
            "wheel_align_evaluate.py",
            "wheel_align_visualize.py",
        ):
            source = os.path.join(os.path.dirname(__file__), filename)
            shutil.copy2(source, os.path.join(source_dir, filename))
            with open(source, "rb") as f:
                hashes[filename] = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(out_dir, "provenance.json"), "w") as f:
            json.dump(
                dict(
                    argv=sys.argv,
                    cwd=os.getcwd(),
                    source_sha256=hashes,
                    initialization=args.init_params,
                ),
                f,
                indent=2,
            )

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=config.to_dict(),
        )

    if args.privileged_critic:
        from workspace.wheel_align_critic import PupperWheelAlignCriticEnv
        if args.encoder_hold:
            class RefineCriticEnv(PupperWheelAlignCriticEnv, env_type):
                pass
            env = RefineCriticEnv(config, model_path=args.model_path)
        else:
            env = PupperWheelAlignCriticEnv(config, model_path=args.model_path)
    else:
        env = env_type(config, model_path=args.model_path)
    evaluation_config = config.copy_and_resolve_references()
    if args.encoder_hold:
        evaluation_config.direct.reference_reset_probability = 0.0
    eval_env = env_type(evaluation_config, model_path=args.model_path)
    video_config = evaluation_config.copy_and_resolve_references()
    if align:
        video_config.automatic_commands = False
        video_config.align.initial_completed_max = 0
    video_env = env_type(video_config, model_path=args.model_path)
    if wandb_run is not None:
        with open(os.path.join(out_dir, "wandb_run.json"), "w") as f:
            json.dump({"id": wandb_run.id, "url": wandb_run.url}, f)
        print(f"W&B run: {wandb_run.url}", flush=True)

    network_options = (
        {
            "value_hidden_layer_sizes": tuple(config.policy.hidden_layer_sizes)
            if direct
            else (128, 128),
            "initial_std": config.policy.initial_std,
        }
        if align
        else {}
    )
    network_factory = functools.partial(
        make_alignment_networks if align else ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(config.policy.hidden_layer_sizes),
        activation=_ACTIVATIONS[config.policy.activation],
        **network_options,
    )
    restore_params = None
    if args.init_params is not None:
        print(f"Warm-starting from {args.init_params}")
        restore_params = model.load_params(args.init_params)

    randomization_fn = (
        domain_randomize_wheel_align if align else domain_randomize_wheeled
    )
    com_overrides = {}
    if args.com_x_shift_range is not None:
        com_overrides["body_com_x_shift_range"] = tuple(args.com_x_shift_range)
    if args.com_y_shift_range is not None:
        com_overrides["body_com_y_shift_range"] = tuple(args.com_y_shift_range)
    if com_overrides:
        print(f"Overriding CoM shift range(s) for this run only: {com_overrides}")
        randomization_fn = functools.partial(randomization_fn, **com_overrides)

    ppo_kwargs = dict(config.ppo)
    if direct:
        ppo_kwargs.update(
            supervision_path=config.direct.supervision_path,
            action_scale=config.action_scale,
            supervision_weight=config.direct.supervision_weight,
            supervision_error_scales=config.direct.supervision_error_scales,
            supervision_final_weight=config.direct.supervision_final_weight,
            supervision_anneal_steps=config.direct.supervision_anneal_steps,
            kl_budget=config.direct.kl_budget,
        )
    train_fn = functools.partial(
        supervised_align_ppo.train if direct else ppo.train,
        **ppo_kwargs,
        network_factory=network_factory,
        randomization_fn=randomization_fn,
        restore_params=restore_params,
        wrap_env_fn=wrap_for_alignment if align else None,
    )

    times = [datetime.now()]
    audit = None
    best_score = None

    def save_metrics(step, kind, metrics):
        if align:
            with open(os.path.join(out_dir, "metrics.jsonl"), "a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": int(step),
                            "kind": kind,
                            **{k: float(v) for k, v in metrics.items()},
                        }
                    )
                    + "\n"
                )

    def progress(step: int, metrics: dict) -> None:
        times.append(datetime.now())
        reward = metrics.get("eval/episode_reward", float("nan"))
        # Print the tracking errors alongside the reward, not just the reward: a
        # policy that simply parks and never moves still banks every posture term,
        # so vel_err is the number that says whether it is doing the TASK.
        # brax reports episode metrics as SUMS, so divide by the episode length.
        n = metrics.get("eval/avg_episode_length", 0.0) or 1.0
        d = lambda k: metrics.get(f"eval/episode_{k}", float("nan")) / n
        if align:
            print(
                f"[{step:>12,}] reward={reward:.2f} eplen={n:.0f} "
                f"completed_avg={d('completed_legs'):.2f} drift={d('body_drift_dist'):.4f}m "
                f"outside_20mm={d('drift_outside_tolerance'):.2%} "
                f"hold_err={d('hold_angle_error'):.3f}rad held_err_avg={d('held_angle_error'):.3f}rad "
                f"lower_err_avg={d('lower_angle_error'):.3f}rad scrub={d('active_scrub_speed'):.3f}rad/s",
                flush=True,
            )
            metrics.update(
                {
                    f"eval/mean_{k}": d(k)
                    for k in (
                        "completed_legs",
                        "body_drift_dist",
                        "drift_outside_tolerance",
                        "hold_angle_error",
                        "held_angle_error",
                        "lower_angle_error",
                        "active_scrub_speed",
                        "active_angle_error",
                        "active_clearance",
                    )
                }
            )
            metrics["eval/new_completions_per_episode"] = metrics[
                "eval/episode_new_completions"
            ]
            held_samples = metrics["eval/episode_held_samples"]
            metrics["eval/held_error_conditional"] = (
                metrics["eval/episode_held_error_sum"] / held_samples
                if held_samples > 0
                else 0.0
            )
        else:
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
            # Initial evaluation precedes its policy callback; later evaluations
            # follow it. Commit once both metrics and video/audit are present.
            wandb_run.log(metrics, step=step, commit=(step != 0) if align else None)
        if align and metrics.get("training/kl_mean", 0.0) > 0.15:
            print(f"  PPO KL warning: {metrics['training/kl_mean']:.4g}", flush=True)
        save_metrics(step, "ppo", metrics)

    def _log_video(step: int, frames, fps: int) -> None:
        path = os.path.join(out_dir, f"rollout_step_{step}.mp4")
        media.write_video(path, frames, fps=fps)
        print(f"  wrote video -> {path}")
        if wandb_run is not None:
            import wandb

            wandb_run.log(
                {"eval/video": wandb.Video(path, fps=fps, format="mp4")},
                step=step,
                commit=False if align else None,
            )

    def policy_params_fn(step: int, make_policy, params):
        nonlocal audit, best_score
        # Save every evaluated policy before rendering; surface failures directly.
        model.save_params(os.path.join(out_dir, "mjx_params_latest"), params)
        if align:
            model.save_params(os.path.join(out_dir, f"mjx_params_{step}"), params)
            if audit is None:
                audit = make_evaluator(
                    config,
                    make_policy,
                    randomization_fn,
                    args.model_path,
                    count=args.audit_episodes,
                    env_type=env_type,
                )
            audit_metrics = audit(params)
            # Once success/completion/survival tie, prefer less body drift.
            # Tiny differences in already-small hub errors should not decide it.
            score = (
                audit_metrics["audit/all_four_success_fraction"],
                audit_metrics["audit/mean_completed_legs"],
                audit_metrics["audit/survival_fraction"],
                -audit_metrics["audit/drift_p95"],
                -audit_metrics["audit/held_error_p95"],
            )
            if direct:
                # A deployment candidate must be fall-free before comparing
                # completion counts; retain the teacher's historical selection.
                score = (audit_metrics["audit/survival_fraction"] == 1.,) + score
            if best_score is None or score > best_score:
                best_score = score
                model.save_params(os.path.join(out_dir, "mjx_params_best"), params)
                with open(os.path.join(out_dir, "best_checkpoint.json"), "w") as f:
                    json.dump(
                        {"step": int(step), "score": score, **audit_metrics},
                        f,
                        indent=2,
                    )
            save_metrics(step, "full_sequence", audit_metrics)
            wandb_run.log(audit_metrics, step=step, commit=False)
            print(
                f"  full-sequence audit [{step:,}]: {json.dumps(audit_metrics)}",
                flush=True,
            )
        if args.no_eval_videos:
            if align and step == 0:
                wandb_run.log({}, step=step, commit=not direct)
            return audit_metrics if align else None
        inference_fn = make_policy(params, deterministic=True)
        frames, fps = visualizer.render(video_env, inference_fn, jax.random.PRNGKey(0))
        _log_video(step, frames, fps)
        if align and step == 0:
            wandb_run.log({}, step=step, commit=not direct)
        return audit_metrics if align else None

    if align:
        print(
            f"Training QuadMorph alignment: {config.ppo.num_timesteps:,} steps, "
            f"{config.ppo.num_envs} envs, full sequences, {config.episode_length * config.ctrl_dt:.0f}s episodes, "
            f"wheel_control_mode={config.align.wheel_control_mode}",
            flush=True,
        )
    else:
        print(
            f"Training WHEELED locomotion policy: "
            f"vx={tuple(config.lin_vel_x_range)} vy={tuple(config.lin_vel_y_range)} "
            f"yaw={tuple(config.ang_vel_yaw_range)}, wheel cap "
            f"{configs.WHEEL_MAX_LINEAR_SPEED:.2f} m/s ({configs.WHEEL_MAX_SPEED:.2f} rad/s), "
            f"episode_length={config.episode_length} steps "
            f"({config.episode_length * config.ctrl_dt:.1f}s)"
        )
    make_inference_fn, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
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
    frames, fps = visualizer.render(video_env, inference_fn, jax.random.PRNGKey(1))
    final_path = os.path.join(out_dir, "rollout_final.mp4")
    media.write_video(final_path, frames, fps=fps)
    print(f"Final video -> {final_path}")
    if wandb_run is not None:
        import wandb

        wandb_run.log(
            {"eval/video_final": wandb.Video(final_path, fps=fps, format="mp4")}
        )
        wandb_run.finish()

    if not align:
        print(f"Next: python -m workspace.export_policy --params {params_path}")
    else:
        print(
            "Alignment uses a new observation/command contract; the legacy drive exporter is not compatible."
        )


if __name__ == "__main__":
    main()
