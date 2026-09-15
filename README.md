# Selected robot controllers

Extracted from `robot-code` at `6b07745ab4c7ab59e6da7766a17896f118d5198d`.
This branch starts with an empty root commit and contains only the selected
controllers, their source dependencies, configuration, and focused tests.

## Selected behavior

| Behavior | Export / plan | Runtime |
| --- | --- | --- |
| Leg walking, heating backpack + 9 mm | `policy_walk_v2.json` | `NeuralController` |
| Wheels, heating backpack + 9 mm, 0.65 rad stance | `policy_wheel.json` | `NeuralController` |
| Leg to wheel, gentle lowering | `policy_leg_to_wheel.json` | `leg_to_wheel/policy.hpp`: actor, sequencer and lowering filter |
| X triangle roll to stand | `triangle_roll_plan.json` | `TriangleRollController`, measured roll and support feedback |
| Newest shipped lift and align | `policy_leg_lift_wheel.json` | `WheelLiftController`: continuous proximal actor, hub position PD, alignment gates and sequencing |

Exports and controller configuration are in `ros2_ws/src/neural_controller/launch`.
The zero-output `policy_wheel_to_walk_ready.json` is also retained: it supplies
the existing stance transition needed before entering walking or lift from wheels.
`combined_motion.yaml` retains the roll-to-walk frame and handoff settings.
`motion_buttons.py` retains the existing command dispatcher.

## Why some shared files remain

- Triangle roll inherits `KeyframeController` for encoder/IMU access and lifecycle
  handling. That implementation and its transitive headers are required; its
  independent controller is not registered in the plugin XML.
- Walking and wheels use the shared `NeuralController`, which also compiles
  historical alignment helpers. Those headers are retained without their old
  policy exports, configuration entries or separate plugins. Splitting this
  shared implementation is a later refactor.
- `robot_calibration` is a compile/runtime dependency of these controllers and
  the dispatcher. Its existing implementation is copied unchanged. No reference
  capture, calibration redesign or hardware activation was performed.
- Vendored RTNeural, Eigen and JSON headers are needed for inference. Training
  code, checkpoints, videos, historical policies and unrelated ROS packages are
  excluded. Upstream manifests retain provenance and may describe evidence files
  that remain only in the original repository.

## Current scope and remaining integration

This is a controller source bundle, not a complete robot startup distribution.
Hardware drivers, robot description, startup launch and calibration workflow will
be addressed separately. Existing calibration gates remain intact.

**Leg-to-wheel still needs a ROS hardware adapter**, including coordinate mapping
and validated clearance/contact feedback. Its C++ runtime and sequencer are
included, but it is not registered as a runnable ROS controller.

No software was deployed to the robot. No physical motion was tested.

## Build and test without hardware

Use Linux with ROS 2 Jazzy and the dependencies declared in the two package.xml
files. From this repository root:

```sh
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2_ws/src --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
colcon test --event-handlers console_direct+
colcon test-result --verbose
python3 scripts/verify_import.py
```

Tests use inference fixtures and fake controller interfaces. They do not start
the physical hardware stack. `IMPORT_MANIFEST.json` records imported file hashes
and the source commit. CMake, plugin registration and the walking/wheel config
were narrowed to the selected controllers; controller source and exports are
unchanged.

## Extraction validation (2026-09-15)

- ROS 2 Jazzy Release build passed for both packages in WSL Ubuntu.
- All 10 controller CTest groups passed, covering walking/wheel/leg-to-wheel
  inference, roll core/lifecycle, walking frame/handoff, lift core/lifecycle,
  and the button dispatcher.
- All 588 imported files matched their recorded hashes and their staged Git
  blobs. Selected walking/wheel parameter values match the source configuration.
- Calibration storage and all 20 Python storage tests passed. Its ROS capture
  suite passed 2 of 3 tests. `test_moving_encoders_prevent_capture` failed and
  reproduced identically against the original `robot-code` checkout: the fixture
  reports nonzero velocity with constant positions, while the existing CLI uses
  `StationarySample(check_velocity=False)` and checks position excursion instead.
  This pre-existing test/implementation mismatch remains unchanged for the later
  calibration discussion. The complete suite therefore does not pass.
