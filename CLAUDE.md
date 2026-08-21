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
- **Lift with the leg, not the body.** The policy must hold the standard standing pose —
  torso upright, at full standing height, roughly where it started, other three feet
  planted — and raise the commanded leg as high as it can from there. It must NOT reach
  height by leaning back, sitting, or resting another limb on the ground. This is enforced
  by the reward *and* by episode termination; see `workspace/README.md` "Reward design".
  (Supersedes the earlier fixed `LIFT_DELTAS` target-pose approach, which was removed.)
- **Status: `leg_lift_2026-08-11_01-35-11` was real-hardware tested (2026-08-17) with one
  hardware fault found, not a software/policy problem** — `front_l`, `front_r`, `back_l` all
  lifted cleanly and held; `back_r` did not lift because the leg physically snapped
  (pre-existing 3D-print crack lines), judged a print/hardware fault since command routing to
  `back_r` had already been verified working in an earlier session before that leg broke. Full
  test log and reporting template: `Stanford/pupperv3-monorepo/LEG_LIFT_TESTING.md`.
- **`leg_lift_2026-08-19_20-18-08` was real-hardware tested (2026-08-19): success, with one
  regression that led to the current policy.** This policy addressed three issues found in
  the 2026-08-17 hardware round — CoM-mismatch back-leg struggling, lift-snapping, and
  hold-phase oscillation (see `workspace/README.md` Status for the full rationale) — and on
  this test, **all four legs lifted and stabilized cleanly**, no e-stop, no fall. One
  regression: **`front_l` didn't lift as high as before**, suspected to be the CoM correction
  overshooting (that run trained ~4.5cm back / 2cm right of the model's original assumed CoM).
- **Both the 1cm/1cm and 1.5cm/1.5cm CoM corrections were hardware-tested (2026-08-20)
  and neither beat the original 2cm/2cm.** Verdict: 2cm/2cm is still the best CoM
  correction so far — 1cm was too central (undercorrected), 1.5cm didn't improve on
  2cm either. **Two more issues found, both to fix alongside reverting to 2cm/2cm in
  the next training pass:** (1) lowering a leg on O-button switch snaps down abruptly —
  the existing hip rate limit only covers raising, not lowering; (2) `front_l` lifts
  lower than the other three legs **across all three CoM variants tested**, so it's a
  separate issue from CoM tuning, not something more CoM sweeps will fix. Overall
  verdict is still a real success — command routing, balance, and 3-of-4 leg lift
  quality are solid; these are refinements on the 2cm/2cm base, not a redesign. Full
  hardware test log: `Stanford/pupperv3-monorepo/LEG_LIFT_TESTING.md`.
- **Tried and discarded a from-scratch (no warm-start) run** to test whether `front_l`'s
  shortfall was a policy artifact vs. CoM-related — it collapsed into an extreme
  version of the "lifts three, gives up on the fourth" local optimum, except mirrored:
  it lifted only `front_l` (near the 0.12m target) and gave up entirely on the other
  three. Informative (argues `front_l` isn't inherently harder to lift, so the
  warm-started lineage's shortfall is likely inherited/policy-level, not physical) but
  not deployable — discarded rather than kept. **Decision: keep warm-starting.**
- **`leg_lift_2026-08-21_18-17-50` is now deployed** — warm-started from
  `leg_lift_2026-08-19_20-18-08`, reverted to the 2cm/2cm CoM correction, and now
  includes a new **gradual-lowering rate limit** (`configs.LOWER_HIP_RATE_LIMIT_STEPS`)
  extending the existing raise-side hip rate limit to also cover a leg's hip while it's
  lowering after an O-button switch, fixing item (1) above. Item (2) (`front_l`) is
  still unaddressed. **READY FOR THE NEXT ROUND OF HARDWARE TESTING — awaiting
  results.** Watch specifically: whether lowering is now gradual (not a snap-down), and
  `front_l` lift height (not expected to have changed). Full rationale and this run's
  numbers: `workspace/README.md` Status section.

## Training side — `mujoco_playground/workspace/` (our code)

The leg-lift training pipeline. See `workspace/README.md` for setup/run. Key files:

- `leg_lift_env.py` — `PupperLegLiftEnv` (brax `PipelineEnv`, MJX): reset/step/obs/reward and
  command sampling. Modeled on `pupperv3-mjx`'s `PupperV3Env` and go1 `getup.py`.
- `configs.py` — **single source of truth**: canonical 12-joint order, limits, home pose,
  per-joint `ACTION_SCALE` (abduction 0.5 / hip 2.0 / knee 1.0 — the policy head is
  tanh-squashed, so this IS the reachable joint range around the home pose), reward
  weights, PPO hyperparams, model path.
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
