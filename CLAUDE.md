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
| `Stanford/pupperv3-monorepo/` | The code that **runs on the robot** (ROS2). | The **deployment target**. We read it to see what's deployed and write the code that runs/binds the new policy. Edit here. |
| `Stanford/training/pupperv3-mjx/` | The providers' **RL training pipeline for Pupper** (MJX/Brax env). | **Reference + likely starting point** for defining the leg-lift env, reward, and policy export. |
| `mujoco_playground/` | DeepMind's general MJX RL environment suite. | **General RL reference** for patterns/infra. No Pupper code in it. |
| `.notes/goal.md` | The task definition. | — |

Keep the three repos separate: distinct upstreams, git histories, and licenses. Don't move
code between them or commit one's changes against another.

## Project decisions made so far

- **Separate policy.** The leg-lift is its own RL policy, distinct from the locomotion policy
  (and from the existing `neural_controller_three_legged` policy).
- **Activated by a controller button.** It runs as another runtime-switchable controller,
  bound to a (TBD) PS5 button — the same activation pattern as the locomotion policies.
- **Fixed hold duration.** Train for one hold duration; if a different duration is needed,
  retrain. Do **not** add command/observation inputs to make duration configurable.
- **Status: groundwork only.** As of this writing nothing is implemented — we are gathering
  context and laying out the project. Do not start training or writing integration code
  until asked.

## Training side — `Stanford/training/pupperv3-mjx/`

The Pupper-specific MJX training package. Key modules in `pupperv3_mjx/`:

- `environment.py` — the MJX/Brax environment (observations, actions, episode logic).
- `rewards.py` — reward terms.
- `domain_randomization.py` — sim-to-real randomization.
- `export.py` — exports a trained policy to the JSON format the robot's `neural_controller`
  loads. **This is the bridge between training and deployment.**
- `config.py`, `obstacles.py`, `utils.py`, `plotting.py`.
- `Pupper_RL_PUBLIC.ipynb` (in `Stanford/training/`) — the providers' end-to-end training
  notebook; the reference for how a Pupper policy is actually trained.

A leg-lift policy would most likely be a new/derived environment + reward here, exported the
same way. `mujoco_playground/` is secondary reference for RL infrastructure patterns.

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
  append its name to `controller_names`, and assign a button index.

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

Training (`pupperv3-mjx`) is a Python/JAX package — set up its own env with `uv` /
`requirements.txt`; needs a CUDA GPU for real training.

## Conventions (from the monorepo's [CLAUDE.md](Stanford/pupperv3-monorepo/CLAUDE.md) — follow these)

- **No silent fallbacks.** Don't paper over failure with broad `try/except` or default
  values; surface it (warn/raise) so problems are visible.
- **Use `uv`** for Python package management. Never mutate `PATH` inside files.
