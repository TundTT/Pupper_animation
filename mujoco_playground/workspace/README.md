# Pupper V3 wheeled locomotion — RL training

Training pipeline for a new **wheeled locomotion** policy, forked from this repo's
leg-lift training pipeline (see `master` branch) to reuse its MJX / brax RL stack,
project structure, and deployment path rather than starting from zero.

This `workspace/` is **our** code; the rest of `mujoco_playground/` is the upstream
library we build on (brax PPO, MJX env patterns; go1 `getup.py` is the closest
upstream reference task). A trained policy deploys to the robot the same way the
existing locomotion/leg-lift policies do — an exported JSON MLP loaded by
`neural_controller` (see the monorepo).

## The wheeled robot

The MJCF
(`Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml`)
is now a **four-wheeled** robot, converted in place from the quadruped:

- Each leg's **`_3` joint IS the wheel** — the old knee joint, repurposed as the
  wheel's continuous spin joint (`limited="false"`), with the wheel's mesh, mass and
  collision cylinder attached directly to that body. The old shin/foot geometry is
  gone. **Still 12 joints, not 16.**
- So each leg is `_1` abduction (position) + `_2` hip (position) + `_3` wheel
  (**velocity**: ctrl is a target speed in rad/s, not an angle). That **mixed
  actuation** is the thing to be careful about — see below.
- Wheel mass/inertia come from real scale measurements, not CAD estimates — see
  `Stanford/training/pupper_v3_description/WHEEL_MASS_LOG.md`.
- The `home` keyframe is the **splayed stance** (abduction at ±1 rad) that puts all
  four wheels on the ground; `configs.DEFAULT_POSE` matches it and `wheel_env.py`
  asserts the two agree.

### The mixed-actuation trap

Both the providers' `pupperv3_mjx.environment.PupperV3Env` and this repo's
`leg_lift_env.py` stamp **one position-PD gain set over every actuator row**:

```python
sys.replace(actuator_gainprm=...at[:, 0].set(kp),
            actuator_biasprm=...at[:, 1].set(-kp).at[:, 2].set(-kd))
```

On this robot that silently overwrites the wheels' velocity gains with position
gains — the model still loads and still trains, just against wrong physics.
`wheel_env.py` and `randomize.domain_randomize_wheeled` override the two groups
separately, indexed by `configs.POSITION_ACTUATOR_ROWS` / `WHEEL_ACTUATOR_ROWS`.
Anything new that touches actuator gains must do the same.

Two more consequences of the split, both handled in `wheel_env.py`:
- The joint **observation** reports angle for the position rows but **speed** for the
  wheel rows (a free-spinning joint's angle is unbounded and wraps), and
  `stance_pose`/`dof_pos_limits` score the position rows only.
- `configs.WHEEL_FORWARD_SIGN` flips the wheel commands per side. The left and right
  legs' frames are **mirrored**, so the two sides' wheels spin about opposite world
  axes; without this the sides fight each other and the robot **does not move at all**
  (measured: <3 cm in 2 s with wheels at ~14 rad/s). The real motors need the same
  convention.

The full leg-lift design rationale (reward-shaping history, local optima hit during
tuning, hardware test results) lives on `master`'s copy of this README — worth reading
for the lessons that generalize (e.g. "reward weights shape behavior inside the
feasible set; termination is what removes a strategy").

## Files

| File | Purpose |
|---|---|
| `configs.py` | Joint order/limits, default pose, per-joint action scale, actuator row splits, wheel geometry, reward weights, PPO hyperparameters, model path. **Single source of truth.** `get_wheel_config()` is the live one; `get_config()` is legacy leg-lift. |
| `wheel_env.py` | `PupperWheelEnv` (brax `PipelineEnv`, MJX): the wheeled task — reset/step/obs/reward, velocity-command sampling. |
| `train.py` | brax PPO training entry for the wheeled policy; saves brax params to `output/<run>/mjx_params`. Optional W&B logging + rollout videos. |
| `wheel_visualize.py` | Rolls the policy through a fixed command showcase (stop / forward / arcs / spin / reverse) and renders a `tracking_cam` video. |
| `randomize.py` | Physics domain randomization. Use `domain_randomize_wheeled`; the plain `domain_randomize` is the leg-lift (all-position-actuator) version. |
| `export_policy.py` | Converts brax params → `neural_controller` JSON (normalization folded into layer 0). |
| `leg_lift_env.py`, `visualize.py` | **Legacy leg-lift reference, not used by the wheeled pipeline.** Kept untouched; the live leg-lift task is on `master`. |

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

# smoke test first (JITs the env, a few PPO iters) before a long run
.venv/bin/python -m workspace.train --num_timesteps 300000 --num_envs 512

# full run (run from the mujoco_playground/ dir so `workspace` is importable)
wandb login                                 # once, if using W&B
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -m workspace.train --use_wandb \
    > train.log 2>&1 & disown
# export the trained policy to the robot's JSON format
python -m workspace.export_policy --params workspace/output/<run>/mjx_params
```

> **Do NOT use `uv run` in this project — use `.venv/bin/python`.** `uv run` re-syncs
> jax/jaxlib to `uv.lock`'s 0.6.2, which mismatches the installed
> `jax_cuda12_plugin` 0.5.0: that silently disables both GPUs and makes
> `brax.io.mjcf.load` **segfault** (exit 139). If you hit that, re-pin with
> `uv pip install "jax==0.5.0" "jaxlib==0.5.0" "orbax-checkpoint==0.11.1" "flax==0.10.4" "optax==0.2.4" "numpy==2.2.6"`
> and check `jax.devices()` lists two `CudaDevice`s. (`uv pip install` is fine; it is
> `uv run`'s implicit sync that breaks things.) Long runs should be detached with
> `nohup ... & disown` so an SSH drop cannot kill them.

`num_timesteps` defaults to 200M and `num_envs` to 8192 (fits the 5090's 32 GB with
room to spare on bigger cards); lower `--num_envs` if VRAM is tight. Pin
`CUDA_VISIBLE_DEVICES` on a multi-GPU host — brax would otherwise pmap across every
visible device, which brings its own `num_envs`/`batch_size` divisibility constraints
for no gain at this scale.

### Watching the policy

Every eval, training renders a rollout through a fixed velocity-command showcase
(stop → forward → forward fast → arc left → arc right → spin in place → reverse →
stop; see `wheel_visualize._SHOWCASE`) and logs it to W&B as `eval/video` (plus a
final `eval/video_final`). The schedule is fixed so runs are comparable
frame-for-frame. Videos are also written to `workspace/output/<run>/*.mp4`
regardless of W&B. Rendering is headless via EGL (`MUJOCO_GL=egl`, set in
`workspace/__init__.py` — it has to be set before anything imports `mujoco`, which is
too early for `train.py`; getting this wrong aborts the process rather than raising).
Flags: `--use_wandb`, `--wandb_project`, `--wandb_entity`, `--no_eval_videos` (skip the
per-eval video if it slows things down — the final video still renders).

Alongside the reward terms, these diagnostic metrics are logged so you can tell *how*
a policy is earning its reward rather than just how much: `lin_vel_error` and
`ang_vel_error` (is it actually tracking the command, or just parking and banking the
posture terms?), `torso_z`, `tilt_deg`, and `wheel_contacts`. **brax reports these as
per-episode sums**, so divide by `eval/avg_episode_length` to read them back as
averages — `train.py`'s progress line already does this.

## Status / what still needs doing

**First wheeled policy trained and sim-verified (2026-08-31):** run
`wheel_2026-08-31_18-53-25`
([W&B](https://wandb.ai/QuadMorph/pupper-wheel/runs/h6yhfsxe)), 200M steps in
**14m27s** on one Blackwell GPU (~230k steps/s), params at
`workspace/output/wheel_2026-08-31_18-53-25/mjx_params`.

Final eval: reward 79.67, episode length 600/600 (never falls), tilt 0.29°, all four
wheels in contact, `vel_err` 0.0097, `yaw_err` 0.0965. It converged by ~43M steps and
then drifted up slowly; the remaining ~150M bought about +0.5 reward.

Measured on the trained policy (deterministic, steady state) — commanded vs achieved:

| commanded | achieved vx | achieved yaw | tilt |
|---|---|---|---|
| stop | −0.003 | +0.009 | 0.1° |
| vx 0.40 | +0.387 | −0.002 | 0.0° |
| vx 0.80 | +0.783 | −0.047 | 0.6° |
| vx −0.40 | −0.416 | +0.011 | 0.0° |
| yaw +1.0 | −0.020 | +0.931 | 0.2° |
| yaw +2.0 | −0.029 | +1.879 | 0.3° |
| yaw −2.0 | −0.032 | −1.885 | 0.4° |
| vx 0.5 + yaw 1.0 | +0.512 | +1.000 | 0.0° |

Within 2–7% on every axis with no meaningful cross-coupling. **Nothing here has
touched hardware**, and the reward weights / gains are still a first pass, not tuned.

Built and verified in sim (2026-08-31):
- Wheeled MJCF: four wheels on repurposed knee joints, real measured mass/inertia,
  splayed home keyframe.
- `configs.get_wheel_config()`, `wheel_env.PupperWheelEnv`,
  `randomize.domain_randomize_wheeled`, `wheel_visualize`, and `train.py` wired
  together.
- Sim checks: settles upright on all four wheels; drives straight forward and reverse
  (0.694 m/s measured vs. 0.72 theoretical at that command — minimal slip); spins in
  place on a differential command; random-action rollouts give finite rewards with no
  spurious terminations.

Known open items:
- **No intermediate checkpointing.** `train.py` saves params only after `ppo.train`
  returns, and no `checkpoint_logdir` is set — killing a run mid-flight loses the
  weights entirely. Worth adding before any run long enough that you'd want to
  early-stop it.
- Yaw is the weaker axis (`trk_ang` plateaued ~0.80 of 1.0 vs `trk_lin` 0.97). If
  yaw accuracy matters, the lever is raising `tracking_ang_vel` (1.0) toward
  `tracking_lin_vel` (2.0) — linear tracking has margin to give up.
- **Wheel speed is capped at 1 m/s** (`configs.WHEEL_MAX_LINEAR_SPEED`). At the
  earlier 1.44 m/s cap the robot **flipped** when all four wheels were commanded to
  full scale. Command ranges are kept strictly inside the cap so the policy has
  authority left to steer and correct.
- `wheel_kv` (0.35), the reward weights, and the DR ranges are untuned starting
  points. The wheeled DR CoM ranges are symmetric placeholders — the leg-lift ones
  encode quadruped-on-feet hardware findings that do not transfer.
- **Deployment is not done.** `export_policy.py` still emits leg-lift metadata
  (`command_states`/`button_sequence`), and the robot side has no velocity-command
  path or wheel velocity-control mode. The on-robot motors also need
  `WHEEL_FORWARD_SIGN` applied.
- `evaluate.py` still references the leg-lift env.

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
