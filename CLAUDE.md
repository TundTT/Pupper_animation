# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this workspace is

A **working area for one task**: build a leg-lift behavior for the **Pupper V3** quadruped
that raises one leg, holds it up stably on the other three legs for a fixed window (so heat
can be applied to a smart-polymer link while it's off the ground), then lowers it. See
[.notes/goal.md](.notes/goal.md) for the research context. This task owns **mechanical
motion only** — not heating or polymer sensing.

> **Approach: this is an RL policy, NOT a scripted/keyframe animation.** Despite the
> "animation" wording in goal.md, the chosen approach is to **train a reinforcement-learning
> policy** (the same kind of artifact as Pupper's locomotion policy) and deploy it to the
> robot. The CSV-keyframe `animation_controller_py` package in the monorepo is therefore
> **not** the relevant subsystem here — ignore it for this work.

**This directory *is* a single git repo** (`Pupper_animation`, origin
`github.com/TundTT/Pupper_animation.git`, branch `master`). Despite the layout below looking
like nested checkouts, `mujoco_playground/`, `Stanford/pupperv3-monorepo/`,
`Stanford/training/pupperv3-mjx/`, and `Stanford/training/pupper_v3_description/` are **not**
separate git repos — none of them has its own `.git` dir. Their content is vendored as regular
tracked files in this one repo's history (confirm with `git ls-files -s <dir> | grep 160000` —
no output means no gitlinks). They started as gitlinked submodules but commit `06945aa`
("Fix Stanford/training subfolders tracked as gitlinks instead of real files") converted them
to plain vendored copies because GitHub was showing them as empty linked repos. There is no
`.gitmodules` and no per-directory remote/history to preserve — a normal `git status`/`git log`
at the repo root covers everything, including these subtrees.

| Path | What it is | Our use |
|---|---|---|
| `mujoco_playground/` | DeepMind's MJX RL suite, vendored — **our editable training base.** | **Where we develop the leg-lift training pipeline.** Our code lives in `mujoco_playground/workspace/`; the rest is the upstream library we build on. Edit here. |
| `Stanford/pupperv3-monorepo/` | The code that **runs on the robot** (ROS2), vendored. | The **deployment target**. We read it to see what's deployed and write the code that runs/binds the new policy. Edit here. |
| `Stanford/training/pupperv3-mjx/` | The providers' **RL training pipeline for Pupper** (MJX/Brax env), vendored. | **Reference** for Pupper-specific details (env structure, joint order, `export.py` JSON format). Not edited. |
| `Stanford/training/pupper_v3_description/` | The Pupper V3 MJCF/URDF model, vendored. | Source of the MJX model the training env loads. Not edited. |
| `.notes/goal.md` | The task definition. | — |

The Pupper MJX model itself lives in `Stanford/training/pupper_v3_description/description/mujoco_xml/`
(`pupper_v3_complete.mjx.position.xml`); the workspace references it in place. Even though
they now share one `.git`, keep treating these subtrees as logically separate upstreams (each
still carries its own `LICENSE`/`README.md`) — don't merge their code together or edit as if
they're one codebase; just commit/push through the single top-level repo.

## Project decisions made so far

- **One command-conditioned policy.** A single RL policy, separate from locomotion, observes
  a 5-way one-hot command = which leg is up (`stand`, `front_l`, `front_r`, `back_r`, `back_l`)
  and raises/holds/lowers that leg while balancing on the other three.
- **O button steps a clockwise sequence.** On the robot, each press of O advances
  `stand → front_l → front_r → back_r → back_l → …`, lowering the current leg and raising the
  next. This state machine lives **on the robot** (`joy_util_node`'s `EStopController`, see
  Deployment side below), not in the policy — the policy is order-agnostic and only sees
  "which leg is up now."
- **Hold is operator-timed.** "Hold" = the command not changing, so duration is however long
  the operator waits between presses. No fixed duration baked into the policy; no retrain to
  change it. (Supersedes the earlier "fixed duration, retrain" decision.)
- **Status: retrained, exported, deployed, and sim-validated end-to-end; a real sim-to-real
  actuator gap remains untuned.** After the original wandb-only run was lost to a workstation
  wipe (reward weights recovered into `configs.py`), the policy was retrained on this checkout
  (`leg_lift_2026-07-22_21-35-05`, `eval/episode_reward ≈ 51.7`, `activation="elu"` — an
  earlier `"swish"` run trained fine but would have segfaulted on-robot since the vendored
  RTNeural only implements `tanh`/`relu`/`sigmoid`/`softmax`/`elu`), exported, and wired into
  `neural_controller_leg_lift` in the monorepo (O-button integration below is **done**, not
  pending). It compiled and ran against `pupperv3_mujoco_sim` without crashing and produces
  real closed-loop behavior, but not yet a clean lift: commanding a leg converges to a stable
  ~26° body tilt, most likely because training uses an idealized direct-position actuator model
  while the sim hardware interface drives a realistic torque-motor model — a genuine
  sim-to-real gap, not a code bug. `LIFT_DELTAS` in `configs.py` now hold real (non-zero)
  values but are still untuned against this gap. See `mujoco_playground/workspace/README.md`
  "Status / what still needs doing" for full detail, including two unrelated repo bugs fixed
  along the way (bad `libmujoco.so` symlinks in `pupperv3_mujoco_sim`).

## Training side — `mujoco_playground/workspace/` (our code)

The leg-lift training pipeline. See `workspace/README.md` for setup/run. Key files:

- `leg_lift_env.py` — `PupperLegLiftEnv` (brax `PipelineEnv`, MJX): reset/step/obs/reward and
  command sampling. Modeled on `pupperv3-mjx`'s `PupperV3Env` and go1 `getup.py`.
- `configs.py` — **single source of truth**: canonical 12-joint order, limits, home pose,
  per-leg lifted targets (`LIFT_DELTAS`, non-zero but untuned against the known sim-to-real
  actuator gap — see Status below), reward weights, PPO hyperparams, model path.
- `train.py` — brax PPO training entry; saves brax params to `output/<run>/mjx_params`.
- `export_policy.py` — converts brax params → `neural_controller` JSON (folds obs normalization
  into layer 0; same scheme as `pupperv3-mjx/export.py`). Emits `observation_layout` /
  `command_states` / `button_sequence` metadata for the deployment side.

**Training needs a CUDA GPU** (prior runs used an RTX 5090 / Blackwell sm_120) and a recent
`jax[cuda12]` — don't assume the current session's host lacks one; check `nvidia-smi` first
(see the Build & run section's note below). `Stanford/training/pupperv3-mjx` and its
`Pupper_RL_PUBLIC.ipynb` remain the reference for how a Pupper policy is trained and exported.

## Deployment side — `Stanford/pupperv3-monorepo/`

The leg-lift integration is **implemented and sim-validated**, following the pre-existing
locomotion template:

- **Policy = an exported JSON MLP** loaded by the `neural_controller` ros2_control plugin
  (C++, [neural_controller.cpp](Stanford/pupperv3-monorepo/ros2_ws/src/neural_controller/src/neural_controller.cpp)).
  A `behavior_ == "leg_lift"` branch there subscribes to `/leg_lift_command_index`, resets to
  `"stand"` on activation, and writes the one-hot command into the observation.
- [config.yaml](Stanford/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/config.yaml)
  now defines **three** policy instances — `neural_controller` (locomotion, `policy_latest.json`),
  `neural_controller_three_legged` (`policy_rich-donkey-233...json`), and
  `neural_controller_leg_lift` (`policy_leg_lift.json`).
- Each instance pins the canonical 12-joint order and `default_joint_pos` — the training env
  must match this joint ordering for the exported policy to be valid on-robot.
- **Runtime switching / button binding** lives in two places: the `joy_util_node` **parameters**
  are in `config.yaml` (`controller_names`, `switch_button_indices`, plus the leg-lift-specific
  `leg_lift_button_index`, `leg_lift_controller_name`, `leg_lift_command_states`,
  `leg_lift_cycle_states`); the **node itself** is
  `EStopController` in [joy_utils/src/estop_controller.cpp](Stanford/pupperv3-monorepo/ros2_ws/src/joy_utils/src/estop_controller.cpp)
  (the file predates the leg-lift feature — the class name is legacy, not descriptive of what
  it now does). O is handled separately from the plain `switch_button_indices` list: first
  press activates `neural_controller_leg_lift` and commands `front_l`; each subsequent press
  cycles `front_l → front_r → back_r → back_l → …`; X exits back to locomotion. It publishes
  the command index on `/leg_lift_command_index`, which `neural_controller.cpp` consumes.
  It's spawned in [launch.py](Stanford/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/launch.py)
  alongside the other controllers.

`pupperv3_mujoco_sim` (a MuJoCo-backed ros2_control hardware interface) can stand in for the
physical robot to test a deployed policy without hardware.

## Build & run (ROS2 side; Linux / ROS2 Humble)

The monorepo's commands target x86 Ubuntu 24 or the robot's Pi 5. This doc originally assumed
Claude Code always runs on a Windows authoring laptop with build/run happening elsewhere — that's
not guaranteed: **check `hostname`/`uname`/`nvidia-smi` at the start of a session** rather than
assuming. (One past session ran directly on Linux/Ubuntu 24.04 with Blackwell-class GPUs — i.e.
potentially capable of both training *and* an x86 ROS2 build itself — which doesn't match the
RTX 5090 workstation named in `workspace/README.md`, so don't assume it's the same machine either;
verify `ros2 --version` and GPU model before assuming either way.)

```sh
cd Stanford/pupperv3-monorepo/ros2_ws
source build.sh                  # colcon build + source install
ros2 launch neural_controller launch.py
```

Leg-lift training (`mujoco_playground/workspace/`) is a Python/JAX package — see
`workspace/README.md`; needs a CUDA GPU (check `nvidia-smi` — don't assume it's unavailable).

## Conventions (follow these)

- **No silent fallbacks.** Don't paper over failure with broad `try/except` or default
  values; surface it (warn/raise) so problems are visible.
- **Use `uv`** for Python package management. Never mutate `PATH` inside files.
