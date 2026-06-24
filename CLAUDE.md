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

This directory is **not itself a git repo** — it bundles three independent checked-out
repositories plus the task notes:

| Path | What it is | Our use |
|---|---|---|
| `mujoco_playground/` | DeepMind's MJX RL suite — **now our editable training base.** | **Where we develop the leg-lift training pipeline.** Our code lives in `mujoco_playground/workspace/`; the rest is the upstream library we build on. Edit here. |
| `Stanford/pupperv3-monorepo/` | The code that **runs on the robot** (ROS2). | The **deployment target**. We read it to see what's deployed and write the code that runs/binds the new policy. Edit here. |
| `Stanford/training/pupperv3-mjx/` | The providers' **RL training pipeline for Pupper** (MJX/Brax env). | **Reference** for Pupper-specific details (env structure, joint order, `export.py` JSON format). Not edited. |
| `.notes/goal.md` | The task definition. | — |

The Pupper MJX model itself lives in `Stanford/training/pupper_v3_description/description/mujoco_xml/`
(`pupper_v3_complete.mjx.position.xml`); the workspace references it in place. Keep the repos
separate: distinct upstreams, histories, licenses. Don't move code between them.

## Project decisions made so far

- **One command-conditioned policy.** A single RL policy, separate from locomotion, observes
  a 5-way one-hot command = which leg is up (`stand`, `front_l`, `front_r`, `back_r`, `back_l`)
  and raises/holds/lowers that leg while balancing on the other three.
- **O button steps a clockwise sequence.** On the robot, each press of O advances
  `stand → front_l → front_r → back_r → back_l → …`, lowering the current leg and raising the
  next. This state machine lives **on the robot** (in/near `joy_util_node`), not in the policy —
  the policy is order-agnostic and only sees "which leg is up now."
- **Hold is operator-timed.** "Hold" = the command not changing, so duration is however long
  the operator waits between presses. No fixed duration baked into the policy; no retrain to
  change it. (Supersedes the earlier "fixed duration, retrain" decision.)
- **Status: scaffolded, untrained.** The training pipeline exists in `mujoco_playground/workspace/`
  but nothing has been trained or validated, and reward weights + `LIFT_DELTAS` are placeholders.

## Training side — `mujoco_playground/workspace/` (our code)

The leg-lift training pipeline. See `workspace/README.md` for setup/run. Key files:

- `leg_lift_env.py` — `PupperLegLiftEnv` (brax `PipelineEnv`, MJX): reset/step/obs/reward and
  command sampling. Modeled on `pupperv3-mjx`'s `PupperV3Env` and go1 `getup.py`.
- `configs.py` — **single source of truth**: canonical 12-joint order, limits, home pose,
  per-leg lifted targets (`LIFT_DELTAS`, placeholders to tune), reward weights, PPO hyperparams,
  model path.
- `train.py` — brax PPO training entry; saves brax params to `output/<run>/mjx_params`.
- `export_policy.py` — converts brax params → `neural_controller` JSON (folds obs normalization
  into layer 0; same scheme as `pupperv3-mjx/export.py`). Emits `observation_layout` /
  `command_states` / `button_sequence` metadata for the deployment side.

**Training runs on the CUDA workstation (RTX 5090 / Blackwell sm_120), not the Windows laptop** —
needs a recent `jax[cuda12]`. `Stanford/training/pupperv3-mjx` and its `Pupper_RL_PUBLIC.ipynb`
remain the reference for how a Pupper policy is trained and exported.

## Deployment side — `Stanford/pupperv3-monorepo/`

The on-robot integration template already exists and is proven — **mirror it**:

- **Policy = an exported JSON MLP** loaded by the `neural_controller` ros2_control plugin
  (C++, [neural_controller.cpp](Stanford/pupperv3-monorepo/ros2_ws/src/neural_controller/src/neural_controller.cpp)).
- [config.yaml](Stanford/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/config.yaml)
  already defines **two** policy instances — `neural_controller` (locomotion, `policy_latest.json`)
  and `neural_controller_three_legged` (`policy_rich-donkey-233...json`). A leg-lift policy is
  a **third instance of this same block** with its own `model_path`.
- Each instance pins the canonical 12-joint order and `default_joint_pos` — the training env
  must match this joint ordering for the exported policy to be valid on-robot.
- **Runtime switching / button binding** is in `joy_util_node` (same file): `controller_names`
  lists the switchable controllers and `switch_button_indices` maps buttons to them. Adding
  the leg-lift policy = add the controller block in config.yaml, spawn it in
  [launch.py](Stanford/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/launch.py),
  append its name to `controller_names`, and bind the O button.
- **New integration work (does not exist yet):** unlike the locomotion policy (driven by
  `cmd_vel`), the leg-lift policy needs its **command** (the 5-way one-hot "which leg is up")
  fed into its observation, advanced by an O-button state machine while the controller is active.
  `neural_controller`'s observation builder currently has no command of this shape. The exported
  JSON's `observation_layout` / `command_states` / `button_sequence` fields describe what to feed.

`pupperv3_mujoco_sim` (a MuJoCo-backed ros2_control hardware interface) can stand in for the
physical robot to test a deployed policy without hardware.

## Build & run (ROS2 side; Linux / ROS2 Humble — not the Windows host)

The monorepo's commands target x86 Ubuntu 24 or the robot's Pi 5. **Don't expect to build or
run ROS2 on this Windows host** — edit here, build/run on the target.

```sh
cd Stanford/pupperv3-monorepo/ros2_ws
source build.sh                  # colcon build + source install
ros2 launch neural_controller launch.py
```

Leg-lift training (`mujoco_playground/workspace/`) is a Python/JAX package — see
`workspace/README.md`; runs on the CUDA workstation, not this host.

## Conventions (follow these)

- **No silent fallbacks.** Don't paper over failure with broad `try/except` or default
  values; surface it (warn/raise) so problems are visible.
- **Use `uv`** for Python package management. Never mutate `PATH` inside files.
