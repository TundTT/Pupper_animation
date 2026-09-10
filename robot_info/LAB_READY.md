# Lab-Ready Gate

The purpose of this gate is to make lab time about controlled hardware testing and feedback. Complete the offline sections before traveling to the robot.

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
ros2 control list_hardware_interfaces
ros2 control list_controllers
sha256sum path/to/policy.json
```

Compare the robot's model hash with the handoff record. Confirm that the working tree and checkout are the intended test version. Do not overwrite another robot's checkout.

## Hardware Activation Boundary

Before enabling actuators:

- Place the robot in the documented safe initial pose and support it appropriately.
- Confirm the correct physical profile, motor mapping, and controller are selected.
- Confirm emergency stop and controller deactivation are immediately available.
- Start with conservative command bounds and no unexpected joystick state.
- Observe initialization and fade-in before issuing a task command.

Hardware activation remains a human-controlled safety decision. Passing offline checks reduces integration risk but does not certify physical safety.

## Feed Findings Back

After testing, distinguish among:

- Physical calibration changes, which update the hardware contract.
- Interface defects, which update controller code, training math, tests, and documentation together.
- Model-performance issues, which return to rewards, randomization, or training.
- Operational notes, which belong in the manually curated task-status documents.

Do not solve an interface defect with an undocumented lab-only patch.
