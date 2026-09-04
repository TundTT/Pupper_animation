# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this workspace is

A **working area for training a new RL policy for Pupper V3: wheeled locomotion.**
This branch (`wheel`) was forked 2026-08-29 from the `master` branch, which builds a
leg-lift behavior (raise one leg, hold it stably on the other three, lower it — see
[.notes/goal.md](.notes/goal.md) for that task's original research context). The
leg-lift work and its full decision history stay on `master`, untouched; this branch
reuses the same training pipeline, deployment plumbing, and vendored Pupper V3 model
as a hardware-verified starting point, customized for wheeled locomotion instead.
**The wheeled model + training pipeline exist and run end to end in sim, and a wheeled
policy (run 3, `wheel_2026-09-02_00-21-44`) is now hardware-validated as shippable** —
drives and turns correctly on the real robot with no issues in the latest test. See
`mujoco_playground/workspace/README.md`'s "The wheeled robot" and "Status" sections, and
`WHEEL_TESTING.md` on the `robot-code` branch for the full hardware test log. One
separate, still-open issue: a physical bump during the first hardware session caused a
momentary power disconnect that led to brief uncontrolled wheel spinning — a
connector/mounting robustness issue, not a policy or software bug.

> **Approach: this is an RL policy, NOT a scripted/keyframe animation** (inherited from
> the leg-lift branch this forked from). The chosen approach is to **train a
> reinforcement-learning policy** (the same kind of artifact as Pupper's locomotion
> policy) and deploy it to the robot. The CSV-keyframe `animation_controller_py`
> package in the monorepo is therefore **not** the relevant subsystem here — ignore it
> for this work.

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

- **This branch (`wheel`) is a fork of `master`'s leg-lift work (2026-08-29),
  training a wheeled-locomotion policy instead.**
- **The wheel replaces each leg's end effector, reusing the knee joint — 12 joints,
  not 16.** Each leg's `_3` joint (the old knee) is now the wheel's continuous spin
  joint, with the wheel's mass/geometry on that same body. Wheel mass/inertia are
  from **real scale measurements, not CAD estimates** (see
  `Stanford/training/pupper_v3_description/WHEEL_MASS_LOG.md`).
- **Actuation is MIXED and code must respect it**: `_1`/`_2` are position-controlled
  (ctrl = rad), `_3` is velocity-controlled (ctrl = rad/s). Both the providers'
  `PupperV3Env` and this repo's `leg_lift_env.py`/`randomize.domain_randomize`
  stamp one position-PD gain set over *every* actuator row, which silently corrupts
  the wheel actuators. Use `wheel_env.py` / `domain_randomize_wheeled`, which split
  by `configs.POSITION_ACTUATOR_ROWS` / `WHEEL_ACTUATOR_ROWS`.
- **The left/right leg frames are mirrored**, so the two sides' wheels spin about
  opposite world axes; `configs.WHEEL_FORWARD_SIGN` corrects this. Without it the
  sides fight and the robot does not translate at all. The real motors need the
  same convention.
- **Wheel speed is capped at 1 m/s** — the robot flipped in sim at the earlier
  1.44 m/s cap when all four wheels were driven to full scale.
- **Do not use `uv run` in `mujoco_playground/`** — it re-syncs jax to `uv.lock`'s
  0.6.2, mismatching the installed CUDA plugin 0.5.0, which disables the GPUs and
  segfaults on model load. Use `.venv/bin/python`. (`uv pip install` is fine.)
- **The leg-lift decision log and hardware-testing history live on `master`**, not
  here — `master`'s `CLAUDE.md` has the full "Project decisions made so far" for
  that task (reward-design iterations, CoM-correction hardware tests, the
  `front_l`/lowering-snap issues, etc.). Consult it for lessons that generalize
  (e.g. reward-shaping pitfalls, warm-starting, detached-training process notes),
  but treat its specific numbers/fixes as leg-lift-only, not applicable here.

## Training side — `mujoco_playground/workspace/` (our code)

The wheeled-locomotion training pipeline. See `workspace/README.md` for setup/run.
Key files:

- `wheel_env.py` — `PupperWheelEnv` (brax `PipelineEnv`, MJX): the wheeled task —
  reset/step/obs/reward and velocity-command sampling. Modeled on `pupperv3-mjx`'s
  `PupperV3Env`, but handles the mixed actuation noted above.
- `configs.py` — **single source of truth**: canonical 12-joint order, limits,
  default pose (the splayed ±1 rad abduction stance), per-joint `ACTION_SCALE`
  (mixed units: rad on position rows, rad/s on wheel rows — the policy head is
  tanh-squashed, so this IS the reachable range around the default), actuator row
  splits, wheel geometry, reward weights, PPO hyperparams, model path.
  `get_wheel_config()` is live; `get_config()` is legacy leg-lift.
- `train.py` — brax PPO training entry; saves brax params to `output/<run>/mjx_params`.
- `leg_lift_env.py` / `visualize.py` — **legacy leg-lift, unused here**, kept as a
  reference. The live leg-lift task is on `master`.
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
