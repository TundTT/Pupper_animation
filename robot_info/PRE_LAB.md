# Prepare before the lab

The user's goal is to arrive, connect, calibrate, and test. Agents should complete
software integration and compatibility checks before that visit. This procedure
applies to future policies and runtime changes, not only alignment. It does not
authorize training, deployment, hardware startup, or motion.

## Lessons from the September 11, 2026 preparation

The laptop ROS checks passed, but the first Pi preparation still needed debugging
and rebuilding. The issues and evidence were:

| Finding | What the next agent must check beforehand | Evidence |
| --- | --- | --- |
| Both computers used Jazzy, but their spawner APIs differed. On the Pi, repeated `--controller-ros-args` overwrote earlier tokens, and `parse_native_args` was absent. | Test the actual target parser and resolved parameter values without executing hardware launch. A distribution name alone is not a version match. Calibration YAML overrides worked on both tested versions. | [Launch compatibility fix and parser tests](https://github.com/TundTT/Pupper_animation/commit/ecc6e75) |
| The Pi's deprecated generated parameter header contained duplicate definitions and failed compilation. | Test both a clean target build and an incremental upgrade of the prepared image. Use the canonical generated header path. Preserve failure logs; investigate stale/generated artifacts before changing controller behavior. | [Header fix](https://github.com/TundTT/Pupper_animation/commit/78f163d) |
| The Pi had local changes to homing-reference logging/values and optional launch nodes. | Compare the deployment checkout with the release revision, back up local edits, and review compatibility before updating. Do not assume the remote branch is the entire deployed configuration. | [Pi preparation record](https://github.com/TundTT/Pupper_animation/blob/e3e1d97/ALIGN_V5_LAB.md#pi-software-preparation-september-11) |
| The gamepad was initially disconnected; the legacy D-pad launcher could start hardware. | Prepare access and input instructions, inspect service behavior, and avoid duplicate or unintended launch paths. Identify the actual gamepad device rather than its motion-sensor device. | Same Pi preparation record |
| Startup reported failure to obtain FIFO scheduling permission. The process had a realtime-priority limit of zero. | Check scheduling permissions under the intended launch user/service in advance; verify actual scheduling and loop timing on the target. A successful build does not verify runtime deadlines. | Live September 11 startup inspection: `/home/pi/align-v5-setup-backup-c1ef4d2/startup.log`, `Could not enable FIFO RT scheduling policy ... Operation not permitted`, and `/proc/17289/limits` at that inspection. This is an observed limitation, not a completed fix. |

The six-package ARM64 build, 11 selected tests, and installed-overlay checks
eventually passed. The [saved Pi build log](https://github.com/TundTT/Pupper_animation/blob/e3e1d97/hardware_testing/align_v5_2026-09-11/pi-setup-build.txt)
is evidence for that revision only. It does not establish physical policy success
or prove that all future software will work.

## 1. Preserve a reproducible target environment

Use the most recently verified target inventory. Record the OS/image identifier,
architecture, kernel, compiler, CMake, Python, ROS package versions, native
dependencies, source-installed package revisions, build flags, and overlay order.
Also record the launch user, relevant groups/device permissions, service
configuration, and realtime/memory-lock limits. Package names or “ROS Jazzy” alone
are insufficient. Do not include credentials, full environment dumps, or private
keys in the record.

When access is available, save a sanitized inventory and a reproducible image or
build recipe for subsequent offline preparation. A matching spare Pi is useful;
an ARM64 container or emulated target image can cover compilation and parser
tests. An x86 build remains useful for fast checks, but does not replace ARM64
validation. Containers/emulation cannot validate the real Pi's devices, scheduler,
motor communication, or loaded control-loop timing.

Do not claim such an image or inventory already exists unless it has been located
and verified. If the target is unavailable and its exact environment cannot be
reproduced, complete the independent checks and state the specific gap before the
lab visit. Do not silently defer it under an unqualified “ready” report.

## 2. Complete the software checks before travel

- Build the intended release and all changed consumers in the matching ARM64
  environment. Use a clean build to detect undeclared dependencies and an
  incremental build when upgrading an existing prepared installation. Do not
  repeatedly rebuild unchanged software after these checks have passed.
- Run the behavior-specific numerical/parity, controller lifecycle, startup and
  reentry, command interruption/soft lowering, stop, and calibration-gate tests.
  Verify the exported model in the actual runtime backend, not just Python.
- Resolve launch substitutions and exercise the target spawner's real argument
  parser without launching hardware. Assert motion starts inactive and physical
  calibration remains required. Missing optional cameras/visualization must not
  break the core launch.
- Inspect the installed overlay, including package prefixes, generated headers,
  plugin libraries, model paths/hashes, and parameter overrides. Testing the source
  tree alone is insufficient. Exclude accidental use of another checkout's packages.
- Verify observation/action order, dimensions, joint modes, signs, offsets, gains,
  limits, initialization, policy rate, and watchdog/stop behavior against the
  selected training artifact and current hardware source. If a new behavior is
  outside the reference validator's schema, explicitly extend/review its contract
  and use dedicated checks; do not bypass a failing validator or force the model
  into a legacy interface.
- Test joystick activation failure, repeated presses, transitions and stop
  priority against a fake manager. Review the actual button/device mapping and
  minimal launch instructions for the intended controller.
- Review startup services and scheduling permissions under the intended user.
  Prepare any required changes before the visit, and explicitly retain actual
  scheduler/timing verification as a target check.

Reuse existing behavior preparation scripts where applicable. For example,
`scripts/prepare_align_v5.sh` on robot-code builds and tests without starting the
hardware. Its final installed preflight requires a connected joystick, so a
hardware-free run must report that device check as pending, not fabricate a pass.

## 3. Deliver a concrete handoff

Provide the tested source commit, environment/image identifier, model and relevant
artifact hashes, configuration, exact build/test commands and logs, launch command,
calibration procedure, controller buttons, first-test sequence, stop procedure,
and rollback procedure. Record known simulation failures and untested conditions.
Separate **software verified**, **target checks pending**, and **physical motion
unverified** in the readiness report.

Resolve LFS/submodule content and dependencies beforehand. If deployment is
authorized and the robot computer is reachable before the visit, stage or build
the release while the motor stack is stopped, preserve its local edits, and verify
the installed artifact. Never replace libraries under a running controller. If
offline, prepare the transfer/update commands and rollback copy for the agent;
do not report deployment as completed. Never use the other robot's checkout.

## 4. Keep the lab checks short and explicit

Recheck identity, selected revision, local changes, overlay, hashes, active
processes, device access, connected gamepad, and launch-service state. Reuse passed
builds/tests when the relevant software and environment are unchanged. If they
differ, identify the difference and run the affected checks before startup.

Read the deployment checkout's `STARTUP_CALIBRATION.md`. Obtain the user's
physical-position confirmation before fresh homing, start the selected stack with
motion inactive, capture and verify the live-session calibration, and report its
ID, wheel homes, and path. Then inspect fresh joint/IMU data, controller state,
scheduling and startup warnings. Only perform the authorized first motion test;
measure tracking and timing under load and retain the trial logs. Heating is manual.

If a new integration issue appears, fix and test it with preserved evidence, then
add the corresponding pre-lab regression check. Do not save time by weakening
calibration, ignoring a failed check, or describing a build as physical validation.
