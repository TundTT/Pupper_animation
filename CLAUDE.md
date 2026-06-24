# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this workspace is

This directory is a **working area for one task**: building a leg-raise / hold / lower
animation for the **Pupper V3** quadruped (see [.notes/goal.md](.notes/goal.md) for the
full goal and research context). It is not itself a git repo — it bundles two independent
checked-out repositories plus the task notes:

- **`Stanford/pupperv3-monorepo/`** — the Pupper V3 robot codebase (ROS2, neural locomotion
  policies, AI voice UI). This is where the animation feature lives and the primary place
  you will edit. Has its own [CLAUDE.md](Stanford/pupperv3-monorepo/CLAUDE.md) — read it.
- **`mujoco_playground/`** — an unmodified clone of DeepMind's MuJoCo Playground (MJX/JAX RL
  training library). Reference/dependency only; **no Pupper-specific code is here** and you
  almost certainly should not edit it for this task.

The goal: raise one leg at a time, hold it up for a *configurable* duration (so heat can be
applied to a smart-polymer link off the ground), then lower it; cycle through all four legs
while staying stable on the other three. This task owns **mechanical motion only** — not
heating or polymer sensing.

## The animation subsystem (the part that matters for this task)

The feature lives in `Stanford/pupperv3-monorepo/ros2_ws/src/animation_controller_py/`.

- **Animations are CSV keyframe files** in
  [animation_controller_py/launch/animations/](Stanford/pupperv3-monorepo/ros2_ws/src/animation_controller_py/launch/animations/).
  Columns: `timestamp_ns, timestamp_sec`, then the 12 joints. **The animation player ignores
  the timestamp columns** — playback speed comes entirely from the node's `frame_rate`
  parameter (default 30 Hz), so each row is one evenly-spaced keyframe. To encode a "hold,"
  repeat the same joint row for the desired number of frames (`duration_s * frame_rate` rows).
- **12 joints**, 3 per leg, named `leg_{front,back}_{r,l}_{1,2,3}`. The player reorders CSV
  columns into the canonical order declared in `animation_controller.py` / the launch file,
  so CSV column order does not have to match — but every joint name must be present.
- **Playback model** ([animation_controller.py](Stanford/pupperv3-monorepo/ros2_ws/src/animation_controller_py/animation_controller_py/animation_controller.py)):
  on `~/animation_select` (a `std_msgs/String` with the CSV stem), it (1) switches the
  controller_manager from the neural controllers to the three `forward_*_controller`s,
  (2) spends `init_duration` seconds interpolating from the current pose to frame 0, then
  (3) plays frames at `frame_rate` with linear interpolation. A 120 Hz timer drives output;
  it publishes position + per-joint `kp`/`kd` to `/forward_{position,kp,kd}_controller/commands`.
- **Gains** `kps`/`kds` (and `init_kps`/`init_kds`) are launch parameters, 12-long. Low kp
  (~5) means compliant joints — relevant for not stressing the smart-polymer link.

### How animations are normally authored

Per the monorepo README: teleop the real robot while recording an mcap bag, then
`scripts/mcap_to_csv.py <bag> -s <start> -e <end>`, drop the CSV into the `animations/`
folder, and rebuild. **For this task you will likely author the CSV programmatically
instead** (compute raise/hold/lower joint trajectories directly) rather than recording a
physical robot. `scripts/animation_editor/main.py <csv>` just plots a CSV's joint curves to
a PNG for inspection — useful for sanity-checking a generated trajectory.

After adding/renaming a CSV you must rebuild the ROS2 workspace and (if exposing it to the
voice UI) add the nickname in `ai/.../pupster.py`.

## Build & run (ROS2 side; Linux/ROS2 Humble)

These are the monorepo's commands — they target x86 Ubuntu 24 or the robot's Pi 5, **not the
Windows host this workspace sits on**. Do not expect to build/run ROS2 locally on Windows;
edit here, build/run on the target.

```sh
cd Stanford/pupperv3-monorepo/ros2_ws
source build.sh                  # colcon build + source install
ros2 launch animation_controller_py animation_controller_py.launch.py
# trigger an animation by publishing its CSV stem:
ros2 topic pub --once /animation_controller_py/animation_select std_msgs/String "{data: 'superman_recording_2025-10-22_17-47-41'}"
```

Run the animation node's tests (standard ament/pytest):

```sh
cd Stanford/pupperv3-monorepo/ros2_ws
colcon test --packages-select animation_controller_py
colcon test-result --verbose
# or a single test file directly:
pytest src/animation_controller_py/test/test_animation_controller.py
```

A MuJoCo build can stand in for hardware (`pupperv3_mujoco_sim` package, used as a
ros2_control hardware interface) so animations can be tried without the physical robot.

## Conventions (from the monorepo's CLAUDE.md — follow these)

- **No silent fallbacks.** Don't paper over failure with broad `try/except` or default
  values; surface the failure (warn/raise) so problems are visible. The existing animation
  controller leans on this — it raises on bad parameters and validates joint counts up front.
- **Use `uv`** for Python package management. Never mutate `PATH` inside files.
- Python packages here pin Python ≥ 3.10 (mujoco_playground wants 3.12) and manage deps via
  `pyproject.toml` + `uv.lock`.

## Keep the two repos separate

`Stanford/pupperv3-monorepo` and `mujoco_playground` are distinct upstreams with their own
git history, licenses, and CLAUDE/README files. Don't move code between them or commit one's
changes against the other.
