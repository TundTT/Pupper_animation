# Lab-Ready Gate

The purpose of this gate is to make lab time about controlled hardware testing and feedback. Complete the offline sections before traveling to the robot.

Follow [PRE_LAB.md](PRE_LAB.md) for the preparation order and evidence requirements.
Do not present a laptop build or matching ROS distribution name as exact target
compatibility. The target package versions and installed artifacts must be checked.

## Policy Contract

- [ ] A hardware profile is explicitly selected.
- [ ] Joint names and order exactly match the canonical 12-joint list.
- [ ] Action types exactly match the selected profile.
- [ ] Observation layout exactly matches the deployed behavior.
- [ ] History order and reset seeding match the controller.
- [ ] Input width equals frame size times history length.
- [ ] Output width is 12 and final activation is `tanh`.
- [ ] Action scales, policy limits, gains, and defaults fit the hardware envelope.
- [ ] Leg-lift command states or locomotion command bounds are embedded where applicable.
- [ ] `python robot_info/validate_policy.py MODEL --profile PROFILE --strict` passes.

## Training And Numerical Checks

- [ ] Training configuration, seed, source commit, and dependency lock are recorded.
- [ ] Domain randomization covers measured latency, noise, gains, contact, and model uncertainty.
- [ ] Evaluation includes resets, fade-in, command transitions, saturation, and perturbations.
- [ ] Training-framework and RTNeural outputs agree on a shared observation fixture.
- [ ] The final JSON hash is recorded after export.
- [ ] The final JSON is a real model, not a Git LFS pointer.

## Software Integration

- [ ] The model loads through `neural_controller/NeuralController` without metadata warnings that indicate missing contract data.
- [ ] A clean ROS workspace build succeeds on ARM64 Debian 12 / ROS Jazzy or an equivalent target image.
- [ ] The base and workspace overlays are sourced in the correct order.
- [ ] Every required ROS package is installed; optional packages cannot block motor control.
- [ ] Launch files use the intended checkout and model path.
- [ ] Controller activation, deactivation, and emergency-stop paths are known.
- [ ] Exact target ROS/package versions and environment provenance are recorded; unavailable target checks are listed explicitly.
- [ ] The target spawner parses the resolved launch arguments correctly, including inactive startup and calibration overrides.
- [ ] Generated headers and incremental upgrades are checked in addition to a clean build when reusing a prepared installation.
- [ ] Installed plugin/configuration/model paths and hashes match the tested release.
- [ ] Joystick transitions, repeated presses, failed activation, and stop priority pass hardware-free integration tests.
- [ ] Launch-user device access, realtime scheduling permissions, service behavior, and optional dependencies have been reviewed.
- [ ] The update/rollback plan preserves target-local edits and excludes the other robot's checkout.

## Handoff Record

Record this with the artifact or test note:

```text
Behavior:
Hardware profile:
Source commit:
Training configuration:
Robot contract schema version:
Policy JSON path:
Policy SHA-256:
Input shape / history:
Action types / scale:
Offline validator result:
RTNeural parity result:
Target build command and result:
Exact target environment / package inventory:
Installed-artifact verification:
Launch-parser / lifecycle / joystick test results:
Update and rollback procedure:
Checks still requiring the physical target:
Launch command:
Safe initial pose:
First-test command limits:
Expected response:
Stop procedure:
Known limitations:
```

## Robot Preflight

These checks are read-only or build-only and should precede controller activation:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/pi/robot-code-leglift/ros2_ws
source install/local_setup.bash
ros2 pkg prefix neural_controller
sha256sum path/to/policy.json
```

Compare the robot's model hash with the handoff record. Confirm that the working tree and checkout are the intended test version. Do not overwrite another robot's checkout.

Inspect existing processes and services before launch. Only query controller-manager
services when a stack is already running; do not start hardware merely to make a
preflight query succeed. Never build or replace libraries beneath a running stack.

## Hardware Activation Boundary

Read `STARTUP_CALIBRATION.md` from the actual deployment checkout first (or inspect
`origin/robot-code:STARTUP_CALIBRATION.md` with `git show` from this reference branch).
Starting the hardware stack performs homing even when the policy is inactive.

Before starting a fresh hardware stack:

- Place the robot in the documented safe initial pose and support it appropriately.
- Obtain the user's explicit confirmation of the encoder-homing pose and marked-ring reference; a request to test is not physical confirmation.
- Confirm the correct physical profile, motor mapping, and controller are selected.
- Confirm emergency stop and controller deactivation are immediately available.
- Start with conservative command bounds and no unexpected joystick state.

After homing, capture shared calibration with `python3 scripts/calibrate_robot.py
capture` and verify `status` before activating motion. An agent may use
`--operator-confirmed` only after the actual confirmation for that startup. Report
the calibration ID, FR/FL/BR/BL homes and storage path. Reuse valid calibration in
the same encoder session; a fresh homing requires a new confirmed capture. Check
controller state, fresh joint/IMU feedback, scheduling and startup warnings before
the authorized trial. Observe policy initialization and fade-in before issuing a
task command. Heating remains manual.

Hardware activation remains a human-controlled safety decision. Passing offline checks reduces integration risk but does not certify physical safety.

## Feed Findings Back

After testing, distinguish among:

- Physical calibration changes, which update the hardware contract.
- Interface defects, which update controller code, training math, tests, and documentation together.
- Model-performance issues, which return to rewards, randomization, or training.
- Operational notes, which belong in the manually curated task-status documents.

Do not solve an interface defect with an undocumented lab-only patch.
