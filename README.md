# Selected robot runtime

Extracted from `robot-code` at `6b07745ab4c7ab59e6da7766a17896f118d5198d`.
This branch starts with an empty root commit and contains only the selected
controllers and their dependencies. It now adds Stanford-based robot infrastructure
with our gravity-pose calibration. No robot deployment or physical test has occurred.

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
  the dispatcher. Its existing position-stability capture and per-session validity
  checks are retained and connected to the new hardware startup.
- Vendored RTNeural, Eigen and JSON headers are needed for inference. Training
  code, checkpoints, videos, historical policies and unrelated ROS packages are
  excluded. Upstream manifests retain provenance and may describe evidence files
  that remain only in the original repository.

## Hardware infrastructure and calibration

Stanford's [pupperv3-monorepo](https://github.com/Nate711/pupperv3-monorepo/tree/6f96c5e79faa05492992c19918f8cd90b9243281)
provides the SPI/IMU driver, ROS hardware interfaces, command multiplexer and base
robot description. `STANFORD_IMPORT.json` pins the source and records adaptations.
We preserve our robot's motor mapping, IMU mounting, proximal limits and continuous
hubs. Stanford's display meshes are retained; they are not a verified model of the
heating backpack or 9 mm custom geometry. The selected policies' existing internal
geometry/model contracts remain unchanged and authoritative for their calculations.

The torque-threshold homing routine has been replaced with our confirmed hanging
pose and wheel-ring procedure. One editable file, `config/gravity_pose.yaml`, holds
the nominal joint references. See [STARTUP_CALIBRATION.md](STARTUP_CALIBRATION.md)
for the physical setup, startup/capture sequence, storage and software limitations.
All selected motion controllers start inactive. Nothing automatically heats,
reshapes, restores a physical pose, rolls or begins walking at startup.

**Leg-to-wheel's ROS hardware adapter remains the next task**, as explicitly agreed.
Its policy, C++ actor/sequencer and lowering filter are retained; it is not spawned.
Live clearance/contact feedback must be resolved before enabling that motion.

## Build and test without hardware

Use Linux with ROS 2 Jazzy and the dependencies declared in the five package.xml
files. From this repository root:

```sh
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths ros2_ws/src --ignore-src -r -y
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

## Validation

The original extraction passed the ROS Jazzy build and ten controller test groups.
This integration adds tests for the actual hardware lifecycle with fake SPI,
transport response validation, installed xacro/reference wiring, startup refusal,
and a ROS controller manager using only GenericSystem mock hardware.

The old calibration test fixture reported velocity while keeping positions fixed.
It now moves positions when testing movement rejection and separately verifies
reported-velocity noise at fixed positions. The runtime's existing
`StationarySample(check_velocity=False)` contract was not changed.

See `VALIDATION.md` for this integration's final results. Software validation does
not establish physical robot readiness. Hardware installation and supervised
reference/communication validation remain pending.
