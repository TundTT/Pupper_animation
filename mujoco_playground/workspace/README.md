# Pupper V3 leg-lift — RL training

Training pipeline for the Pupper leg-lift behavior, built on this repo's MJX / brax
RL stack. One policy learns to raise a single commanded leg, hold it up while
balancing on the other three, and lower it when the command changes.

This `workspace/` is **our** code; the rest of `mujoco_playground/` is the upstream
library we build on (brax PPO, MJX env patterns; go1 `getup.py` is the closest
reference task). The trained policy deploys to the robot exactly like the locomotion
policy — an exported JSON MLP loaded by `neural_controller` (see the monorepo).

## Design

- **One policy, command-conditioned.** The policy observes a 5-way one-hot command
  = which leg is up (`stand`, `front_l`, `front_r`, `back_r`, `back_l`). The target
  is the home pose with that leg's foot raised; the policy tracks it while keeping
  the torso upright and the other three feet planted.
- **Hold is operator-timed, not baked in.** "Hold" is just the command staying
  constant, so the hold duration is however long the operator waits between button
  presses — no fixed duration in the policy, no retrain to change it. During
  training the command is held for a random window then switched, which teaches
  smooth raise/hold/lower transitions.
- **O-button state machine lives on the robot, not in the policy.** Each press of O
  advances a clockwise sequence (`stand → front_l → front_r → back_r → back_l → …`);
  the press lowers the current leg and raises the next by changing the command fed
  to the policy. The policy itself is order-agnostic — it only ever sees "which leg
  is up now."

## Files

| File | Purpose |
|---|---|
| `configs.py` | Joint order/limits, home pose, per-leg lifted targets, reward weights, PPO hyperparameters, model path. **Single source of truth.** |
| `leg_lift_env.py` | `PupperLegLiftEnv` (brax `PipelineEnv`, MJX): reset/step/obs/reward, command sampling. |
| `train.py` | brax PPO training entry; saves brax params to `output/<run>/mjx_params`. Optional W&B logging + rollout videos. |
| `visualize.py` | Rolls the policy out through the O-button sequence and renders a `tracking_cam` video — what you watch to judge the policy. |
| `export_policy.py` | Converts brax params → `neural_controller` JSON (normalization folded into layer 0). |

The Pupper MJX model is referenced in place from the `pupper_v3_description` checkout
(`pupper_v3_complete.mjx.position.xml`); nothing is copied across repos.

## Setup & run — on the CUDA workstation (RTX 5090), not the laptop

Training needs the GPU; author on the laptop, run here. The 5090 is Blackwell
(sm_120), so use a **recent** JAX + CUDA 12 build.

```sh
cd mujoco_playground
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
uv --no-config sync --all-extras            # installs the playground + brax
uv pip install wandb                        # optional, for --use_wandb
python -c "import jax; print(jax.default_backend())"   # -> gpu

# smoke test first (JITs the env, a few PPO iters, ~1 min) before a long run
python -m workspace.train --num_timesteps 200000 --num_envs 1024

# full run (run from the mujoco_playground/ dir so `workspace` is importable)
wandb login                                 # once, if using W&B
python -m workspace.train --num_timesteps 150000000 --use_wandb
# export the trained policy to the robot's JSON format
python -m workspace.export_policy --params workspace/output/<run>/mjx_params
```

`num_envs` defaults to 8192 (fits the 5090's 32 GB); lower it with `--num_envs` if VRAM
is tight.

### Watching the policy

Every eval, training renders a rollout that steps the command through the O-button
sequence (`stand → FL → FR → BR → BL`) and logs it to W&B as `eval/video` (plus a
final `eval/video_final`). Videos are also written to `workspace/output/<run>/*.mp4`
regardless of W&B. Rendering is headless via EGL (`MUJOCO_GL=egl`, set automatically).
Flags: `--use_wandb`, `--wandb_project`, `--wandb_entity`, `--no_eval_videos` (skip the
per-eval video if it slows things down — the final video still renders).

## Status / what still needs doing

- **`LIFT_DELTAS` in `configs.py` are placeholders.** Tune them in sim so each foot
  clears the ground by ~`target_foot_height` without tipping the body. Joint signs
  mirror L/R — verify against the model.
- **Reward weights are a starting point**, not tuned. Expect to iterate on
  `tracking_pose` vs balance terms.
- **Untrained / unvalidated.** None of this has been run yet; treat as the scaffold
  to iterate on, not a finished policy.

## Deployment (monorepo side)

The exported JSON carries `observation_layout`, `command_states`, and
`button_sequence` metadata. To run it on the robot:

1. Copy the JSON into `neural_controller/launch/` and add a third controller instance
   in `config.yaml` with its `model_path` (mirror `neural_controller_three_legged`).
2. Spawn it in `launch.py` and add it to `joy_util_node`'s `controller_names`.
3. **New integration work:** feed the policy its command. Unlike the locomotion
   policy (driven by `cmd_vel`), this one needs a small state machine that, while the
   controller is active, increments the command index on each O press and supplies
   the one-hot to the observation. This is the main on-robot task and does not exist
   yet — `neural_controller`'s observation builder currently has no command-of-this-shape.
