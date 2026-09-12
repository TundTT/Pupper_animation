# Rigid inverted-triangle hardware trial

This test begins **after morphing**, with four rigid triangles in the reviewed
inverted starting pose and the additional **9 mm axial gap** on every limb.
It lifts, flips and plants one limb per explicit command, in this order:
rear-right, rear-left, front-right, front-left. It holds the planted endpoint
between commands. Heating remains manual. Walking does not start afterward.

The software uses the trajectory reviewed at
[`8904c2a`](https://github.com/TundTT/Pupper_animation/commit/8904c2a836951250317300b2722e36376a810a50).
The original 60 passing simulation replays apply to that continuous trajectory.
The added controller, arming procedure, pause behavior and hardware checks have
separate evidence in [the preparation record](hardware_testing/inverted_triangle/README.md).
Software verification does not establish physical clearance or loaded success.

## Before the first motion

Read [STARTUP_CALIBRATION.md](STARTUP_CALIBRATION.md). Its physical-confirmation
requirement still applies before a fresh hardware startup. Starting the stack
performs homing. Do not use the old saved calibration as a substitute for a live
encoder-session capture. The saved hanging reference remains unchanged.

There are two distinct physical references:

1. **Startup reference:** the documented supported homing pose and marked wheel
   rings. Confirm with the operator, start the stack, then capture/check the shared
   calibration as usual. Do not redefine this reference for the triangle motion.
2. **Triangle test start:** with motion inactive, support and position the rigid
   triangles to match the simulation start. Capture the separate triangle mapping
   only after the operator confirms the actual assembly, gap and pose. This
   records the hub-to-model offset in the current encoder winding; it does not
   zero encoders or alter the shared startup calibration.

Proximal start angles, in the existing hardware coordinate frame:

The top row of [these three-view simulation images](hardware_testing/inverted_triangle/first-leg-reference.png)
shows the exact starting pose. Subsequent rows show rear-right lift, rotation and
landing. The image is copied unchanged from the parent CAD review at `8904c2a`.

| Limb | Joint 1 (rad) | Joint 2 (rad) | Model hub angle (rad) |
| --- | ---: | ---: | ---: |
| Front right | 1.0 | -0.29 | 2.14159265359 |
| Front left | -1.0 | 0.29 | 4.14159265359 |
| Rear right | 1.0 | -0.29 | 2.14159265359 |
| Rear left | -1.0 | 0.29 | 4.14159265359 |

The hub numbers are **model coordinates**, not numbers to command directly to
hardware. The operator must match the physical triangle orientation, including
the same face/base orientation as the model. Pointing roughly upward alone is not
a calibration measurement. Proximal readings must be within 0.03 rad of the table.
The software rejects other poses instead of silently inventing proximal offsets
or moving from the hanging pose. An automated approach from another pose is
outside this prepared trial.

Keep the robot externally supported during activation and the two-second gain
ramp. Once status is READY, verify that releasing the external load onto the
triangle bases leaves it stable and inside the tracking guards. Use a support
that can catch a collapse without obstructing the limb sweep for the first loaded
trial. Faults release torque; they do not hold the robot up or execute a recovery.

Inspect the printed assembly and manually check the limb sweeps while unpowered.
The limiting nominal simulated gap is about 2.72 mm between a shin and a nearby
motor; front/rear triangle gaps are larger. Printed tolerances, wire routing,
deformation and compliance are not certified by these values. Ground contact and
clearance are audited in simulation, but there are no equivalent measured contact
or collision signals in this controller. Do not infer floor support from reported
motor effort: that hardware field is a PD estimate.

## Software preparation (no motors)

The September 12 native Pi preparation is complete in the separate
`ros2_ws/install-triangle` overlay. Use the exact launch/source commands in
[the native Pi record](hardware_testing/inverted_triangle/native_pi/README.md)
for that installation; its previous default `install` was preserved for rollback.

The target checkout is `/home/pi/robot-code-leglift`. Do not use the separate
`/home/pi/pupperv3-monorepo` checkout. Inspect running processes, local changes and
the installed overlay before updating or building; never replace the library
under a running controller. Preserve target-local edits and retain the prior
working revision/install for rollback.

```bash
cd /home/pi/robot-code-leglift
bash scripts/prepare_inverted_triangle.sh
```

The script builds the affected packages and checks the installed artifact without
launching hardware. `TRIANGLE_BUILD_BASE` and `TRIANGLE_INSTALL_BASE` can select an
isolated preparation build. No MuJoCo, FCL, Python training environment or GPU is
needed on the Pi. Runtime uses the existing C++ ROS controller library.

Before a lab launch, check package prefixes, source/artifact hashes, joystick
device selection and the actual stop button. This launch uses button index 10
(PS on the Pi's DualSense `/dev/input/js0`) and requires fresh `/joy` messages.
Verify the device mapping for this session. Every activation/motion face-button
binding and stop-release binding is disabled in this trial. D-pad heater/legacy
startup services are not part of the minimal launch.

Native Pi package/parser checks, scheduling permissions, device access and actual
520 Hz timing still need checking on the intended Pi. An ARM64 emulated build
checks architecture compatibility, not device behavior or an exact Debian image.

## Launch and calibration

Only after actual physical startup confirmation:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/pi/robot-code-leglift
source ros2_ws/install/local_setup.bash
ros2 launch neural_controller inverted_triangle_trial.launch.py
```

The joint/IMU broadcasters start; the triangle controller is **inactive**. No
alignment, walking or wheel policy is spawned. The common controller-manager YAML
still defines other plugins, but this launch does not activate them.

In a second sourced terminal, after homing completes, capture and report the live
calibration ID, wheel homes and path using the standing startup procedure:

```bash
python3 scripts/calibrate_robot.py capture
python3 scripts/calibrate_robot.py status
```

Reuse an already-valid calibration in the same live encoder session. After the
operator positions and confirms the separate triangle start with all motion
controllers inactive:

```bash
python3 scripts/inverted_triangle_reference.py capture
python3 scripts/inverted_triangle_reference.py status
```

For agents, the reference script accepts `--operator-confirmed-inverted-start`
**only after the user's actual confirmation**. No such confirmation has been
obtained during software preparation. A replacement mapping requires explicit
`--replace`; prior mappings remain in the calibration history directory.

## Execute the first test

Record joint states, IMU, joystick, commands, controller status and stop events:

```bash
ros2 bag record /joint_states /imu_sensor_broadcaster/imu /joy /emergency_stop \
  /inverted_triangle/command /neural_controller_inverted_triangle/status \
  /neural_controller_inverted_triangle/motor_commands
```

Confirm the IMU topic name from `ros2 topic list` on the installed target. Keep
the robot supported while activating. Activation is a motor action:

```bash
ros2 control switch_controllers --activate neural_controller_inverted_triangle --strict
ros2 topic echo /neural_controller_inverted_triangle/status
```

Activation rejects missing/stale calibration or triangle mapping, an incorrect
start pose, moving joints, stale IMU, excessive tilt or missing gamepad. It ramps
gains over two seconds and then holds READY. A pre-activation command is discarded.

After support/clearance checks, command only the rear-right trial:

```bash
ros2 topic pub --once /inverted_triangle/command std_msgs/msg/Int32 '{data: 1}'
```

It shifts support, lifts, rotates **-pi**, lowers and holds PAUSED with completed
count 1. Inspect the tracking log and the physical landing before proceeding.
Only after each preceding trial is accepted, commands 2, 3 and 4 perform rear-left
**+pi**, front-right **-pi**, then front-left **-pi**. A command sent while moving
is discarded; it never queues the next leg or interrupts a lift with another leg.
Duplicate completed commands do not repeat a flip. There is no “run all” command.
Five-second and one-minute planted pauses passed the separate nominal simulation
checks; longer holds and physical stability still require observation.

Stop immediately through the verified gamepad stop, or either topic:

```bash
ros2 topic pub --once /emergency_stop std_msgs/msg/Empty '{}'
# Equivalent controller-specific latched abort:
ros2 topic pub --once /inverted_triangle/command std_msgs/msg/Int32 '{data: -1}'
```

Stops zero position/velocity/feed-forward/gain command fields. The continuous
hardware profile has no hard-limit gain override. Torque release can let the robot
drop, so the support must catch it. Do not use stop as a soft-lowering request.
No autonomous recovery, interrupted-trajectory resume or walking handoff is
implemented. Reactivation requires lifecycle reset and the original triangle
starting pose; after a partial flip, reposition under external support before
attempting that.

## Runtime contract and telemetry

All twelve commands are position PD with zero desired velocity and zero
feed-forward effort. Proximal gains are `kp=5`, `kd=0.25`; hub gains are `kp=4`,
`kd=0.15`, exactly matching the simulation after arming. Quintic interpolation
uses the original phase durations at a 520 Hz nominal update rate. Commands do
not wrap hub angles, including the rear-left final extra revolution.

Runtime guards: update gap/period at most 10 ms, IMU measurement age at most
100 ms, gamepad receipt age below 500 ms, tilt below 8 degrees, measured joint
speed at most 2 rad/s, proximal tracking error at most 0.25 rad, hub error at most
0.15 rad, predicted PD demand at most 1.5 Nm. These are new conservative trial
guards, not a claim that all 60 original scenarios were rerun through them.
They are distinct from the original simulation audit gates.

Before rotation, before lowering and at planted endpoints, the active hub must
settle within 0.035 rad, all speeds within 0.1 rad/s and proximal errors within
0.22 rad. Supporting hub tracking remains subject to the 0.15 rad guard because
the original loaded simulation includes up to about 0.103 rad of hub deflection.
A failed settling check holds that endpoint for at most three seconds, then faults.
No measured floor-clearance or support-force guarantee follows from these checks.

`status` is a 20 Hz Float64MultiArray:

| Index | Meaning |
| ---: | --- |
| 0 | State: 0 ramp, 1 ready, 2 running, 3 planted pause, 4 complete, 5 fault |
| 1 | Completed leg count (0..4) |
| 2 | Current exported segment index |
| 3 | Elapsed seconds in segment |
| 4 | Fault code |
| 5 | Maximum joint tracking error (rad) |
| 6 | Maximum predicted PD demand (Nm; not measured torque) |
| 7 | Last update period (s) |
| 8 | IMU measurement age (s) |
| 9 | Additional settling wait (s) |

Fault codes: 1 timing, 2 invalid/stale sensors, 3 tilt, 4 tracking, 5 speed,
6 predicted torque, 7 operator stop/gamepad stale, 8 settle timeout,
9 activation failure, 10 invalid command. Faults latch until lifecycle reset.
`motor_commands` contains five fields per joint in FR/FL/BR/BL order:
position, velocity, feed-forward effort, kp, kd. Fault release is zero in all 60.

## Hardware evidence and assumptions

- **User-confirmed design:** rigid post-morph triangles, manual heating, extra
  9 mm axial gap; lower motor housing approximately 5 mm off the floor. The last
  value is a physical estimate, not an encoder-derived measurement.
- **Documented hardware/software contract:** robot-info at
  [`e9b0417`](https://github.com/TundTT/Pupper_animation/tree/e9b04173b034d595a85e6147d73a45cdfe9393e3/robot_info),
  especially HARDWARE.md, SOFTWARE.md and PRE_LAB.md. Pi 5 ARM64, ROS Jazzy,
  SPI/CAN joint routing, position/velocity/gain interfaces and IMU convention.
- **Current implementation overrides the old limited leg profile:**
  [components.xacro](ros2_ws/src/pupper_v3_description/description/components.xacro)
  has continuous hub limits and no hub hard stops. The existing user-confirmed
  continuous-hub change is recorded in the September 9 hardware log and joystick
  source. We did not reintroduce robot-info's older third-joint stops.
- **PD and effort distinction:**
  [hardware writer](ros2_ws/src/control_board_hardware_interface/src/control_board_hardware_interface.cpp)
  clamps feed-forward effort independently of kp/kd, and reports a calculated PD
  effort estimate. The new 1.5 Nm check uses host feedback; it is not a certified
  board-level total-torque clamp or a foot-force sensor.
- **Simulation geometry and limits:** immutable model hash
  `768cbeb0bfbaab0c898e6078d0b778ee2968718d98bdb2857e13538daeb607a8`,
  original tip, 9 mm gap. Plan hash
  `204c0a54172874749b6702e2802872b265489bffc18ce9b20bb08b2bde41e369`.
  The compatibility script checks current joint axes/origins and all command
  limits; physical assembly and ring identity remain operator-confirmed facts.
