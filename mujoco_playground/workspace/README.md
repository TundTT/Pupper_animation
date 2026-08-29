# Pupper V3 legged locomotion — RL training

Training pipeline for a new **general legged locomotion** (walking) policy, forked
from this repo's leg-lift training pipeline (see `master` branch) to reuse its MJX /
brax RL stack, project structure, and deployment path rather than starting from zero.

This `workspace/` is **our** code; the rest of `mujoco_playground/` is the upstream
library we build on (brax PPO, MJX env patterns; go1 `getup.py` is the closest
upstream reference task). A trained policy deploys to the robot the same way the
existing locomotion/leg-lift policies do — an exported JSON MLP loaded by
`neural_controller` (see the monorepo).

## Starting point (fork of the leg-lift branch)

This branch is not yet a general-locomotion pipeline — it is the **leg-lift
pipeline, unmodified**, kept as a hardware-verified reference to build from:

- `configs.py` and `leg_lift_env.py` still describe the leg-lift behavior (5-way
  per-leg lift command, leg-lift reward terms, leg-lift termination conditions) and
  still reference the quadruped MJCF
  (`Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml`),
  which is unchanged and applicable to legged locomotion as-is (same robot geometry).
- Note the existing locomotion policy already deployed on the robot
  (`neural_controller`'s `policy_latest.json`, see CLAUDE.md's "Deployment side") was
  trained via `Stanford/training/pupperv3-mjx/` (a separate, older MJX pipeline), not
  through this `workspace/`. That pipeline's `environment.py`/`rewards.py` are a
  useful Pupper-specific reference for velocity-tracking locomotion reward design,
  even though this branch's actual training will run through `workspace/`.
- What still needs building, in rough order: a new env (velocity/heading-command
  conditioned, not the leg-lift per-leg one-hot) and a new reward design (tracking a
  commanded walking velocity/gait rather than lift height). The model itself likely
  does not need to change — same quadruped, different behavior.
- The full leg-lift design rationale (reward-shaping history, local optima hit during
  tuning, the yaw-drift limitation, hardware test results) lives on `master`'s copy of
  this README — read it before writing the new reward, since several of those lessons
  (e.g. "reward weights shape behavior inside the feasible set; termination is what
  removes a strategy") generalize past leg-lift.

## Files

| File | Purpose |
|---|---|
| `configs.py` | Joint order/limits, home pose, per-joint action scale, reward weights, PPO hyperparameters, model path. **Single source of truth.** |
| `leg_lift_env.py` | `PupperLegLiftEnv` (brax `PipelineEnv`, MJX): reset/step/obs/reward, command sampling. |
| `train.py` | brax PPO training entry; saves brax params to `output/<run>/mjx_params`. Optional W&B logging + rollout videos. |
| `visualize.py` | Rolls the policy out through the O-button sequence and renders a `tracking_cam` video — what you watch to judge the policy. |
| `export_policy.py` | Converts brax params → `neural_controller` JSON (normalization folded into layer 0). |

The Pupper MJX model is referenced in place from the `pupper_v3_description` checkout
(`pupper_v3_complete.mjx.position.xml`); nothing is copied across repos.

## Setup & run — on a CUDA workstation

Training needs a GPU. **Check `nvidia-smi` first rather than assuming which machine
you're on** — this has been run on more than one. The original RTX 5090 workstation
and the current host (2× RTX PRO 6000 Blackwell, 96 GB each) are both Blackwell
(sm_120), so either way use a **recent** JAX + CUDA 12 build.

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
CUDA_VISIBLE_DEVICES=0 python -m workspace.train --use_wandb
# export the trained policy to the robot's JSON format
python -m workspace.export_policy --params workspace/output/<run>/mjx_params
```

`num_timesteps` defaults to 200M and `num_envs` to 8192 (fits the 5090's 32 GB with
room to spare on bigger cards); lower `--num_envs` if VRAM is tight. Pin
`CUDA_VISIBLE_DEVICES` on a multi-GPU host — brax would otherwise pmap across every
visible device, which brings its own `num_envs`/`batch_size` divisibility constraints
for no gain at this scale.

### Watching the policy

Every eval, training renders a rollout that steps the command through the O-button
sequence (`stand → FL → FR → BR → BL`) and logs it to W&B as `eval/video` (plus a
final `eval/video_final`). Videos are also written to `workspace/output/<run>/*.mp4`
regardless of W&B. Rendering is headless via EGL (`MUJOCO_GL=egl`, set in
`workspace/__init__.py` — it has to be set before anything imports `mujoco`, which is
too early for `train.py`; getting this wrong aborts the process rather than raising).
Flags: `--use_wandb`, `--wandb_project`, `--wandb_entity`, `--no_eval_videos` (skip the
per-eval video if it slows things down — the final video still renders).

Alongside the reward terms, these diagnostic metrics are logged so you can tell *how*
a policy is earning its reward rather than just how much:
`lifted_foot_height` (how high the commanded foot actually gets), `body_drift_dist`,
`torso_z`, and `tilt_deg`. **brax reports these as per-episode sums**, so divide by
`eval/avg_episode_length` to read them back as averages.

## Status / what still needs doing

This branch is a fresh fork (2026-08-29) from the leg-lift `master` branch, set up
purely as a starting scaffold for general legged-locomotion (walking) work — no
locomotion-specific training has happened here yet. The full leg-lift status/changelog
(reward redesign history, CoM-correction hardware-testing rounds, the
`front_l`/lowering-snap issues, etc.) lives on `master`'s copy of this file and in
`Stanford/pupperv3-monorepo/LEG_LIFT_TESTING.md` — consult it for lessons that
generalize (reward-shaping pitfalls, warm-starting, detached-training process notes),
but none of those specific numbers/fixes apply to a walking policy.

**Next steps for this branch, in order:**
1. Write a new env (likely velocity/heading-command-conditioned, not the leg-lift
   5-way per-leg one-hot) — can start from `leg_lift_env.py` as a structural template
   for how reset/step/obs/reward/command-sampling wire together, but the observation,
   action space, and reward almost certainly need to change. `Stanford/training/pupperv3-mjx/pupperv3_mjx/environment.py`
   and `rewards.py` (the existing, already-deployed locomotion policy's training
   pipeline) are a useful Pupper-specific reference for velocity-tracking reward
   design, even though this branch trains through `workspace/` rather than that
   pipeline.
2. Design the reward/termination for tracking a commanded walking velocity/gait
   rather than lifting a leg, update `configs.py` accordingly (new
   `COMMAND_STATES`/command space, new reward weights, new termination thresholds).
3. Update `train.py`/`evaluate.py`/`export_policy.py`/`visualize.py`'s hardcoded
   `PupperLegLiftEnv`/leg-lift-command references once the new env exists.

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
