# Walking and wheel hardware trial: 9 mm gap

This trial selects the walking export from `walk_2026-09-11_23-02-17` at
12,779,520 steps and the wheel export from `wheel_2026-09-11_21-58-55` at
201,850,880 steps. Their recorded training geometry uses a 0.009 m outward
assembly translation. That translation is not an encoder offset.

## Controls

Use `ros2 launch neural_controller locomotion_trial.launch.py` after the startup
procedure below. This focused launch loads only the two locomotion policies,
joint/IMU broadcasters, gamepad, teleop, and velocity mux.

| Control | Focused trial action |
|---|---|
| Triangle, `/joy` index 2 | Activate `neural_controller_walk_v2` |
| Circle, index 1 | Activate `neural_controller_wheel` |
| PS, index 10 | Publish emergency stop and request both policies deactivated |
| Options, index 9 | Reactivate last selected policy; defaults to walking before first selection |
| X, Square, L2 | Unbound |
| Left stick | Forward/back and walking sideways |
| Right stick horizontal | Yaw |

Both policies start inactive and require valid shared calibration. Changing
policy requests deactivation of the other policy. Circle's old leg-lift cycle
is disabled. Release sticks before selecting or reactivating a policy; activation
starts the policy's existing two-second home ramp and two-second action fade-in.
Verify the physical gamepad indices from `/joy` before relying on this table.

Trial stick limits are ±0.20 m/s forward, ±0.10 m/s sideways and ±0.40 rad/s yaw.
Wheel behavior ignores lateral commands. The mux accepts only `/teleop_cmd_vel`
and publishes zero after 500 ms without active input, checked every 50 ms.
These limits apply to this joystick path; a separate `/cmd_vel` publisher can
bypass them. Keep only the selected trial stack running.

The full `launch.py` also uses triangle/circle, but retains X alignment and the
legacy L2 action. Use the focused launch for this test. Do not use D-pad Up or the
legacy robot service to launch another checkout. Heating remains manual.

## Preparation while the Pi is off

```bash
bash scripts/prepare_locomotion.sh
```

This builds the shared calibration/hardware/controller/joystick consumers and
command mux, checks the two export hashes and runtime metadata, exercises
inference against independent reference actions, and tests launch and joystick
dispatch using isolated software fixtures. It does not start physical hardware.

## September 13 follow-up

The operator reported both policies were "not bad" on September 12, with wheels
feeling slow at the original 0.20 m/s trial cap. This is qualitative feedback;
tracking speed was not measured. The robot was powered off before the requested
increase, so **0.40 m/s is pending native rebuild and physical testing**. Pulling
files alone is insufficient: the mux and neural-controller command-topic change
requires `bash scripts/prepare_locomotion.sh` before launch. The selected weights,
gains, action scales, yaw limit, PS stop binding and saved start pose are unchanged.

The Pi currently has a detached older commit plus the reviewed lab overlay. Before
updating, preserve that overlay and the saved pose/calibration; reconcile it against
`origin/robot-code` without discarding robot-local changes. Follow the existing
startup/calibration procedure after its next boot.

## Pi preparation and startup

1. Power on the Pi for software preparation, leaving the robot supported. Inspect
   the host, processes, services, branch, local changes and installed package
   prefixes first. The recorded checkout is `/home/pi/robot-code-leglift`; verify
   its remote is this project. Preserve robot-local changes and calibration files.
2. Transfer the reviewed changes and actual LFS weights, then build there with
   `bash scripts/prepare_locomotion.sh`. PC libraries are x86-64 and cannot be
   copied to the ARM64 Pi. Verify `joy_linux`, `teleop_twist_joy`,
   `controller_manager`, `robot_state_publisher`, `xacro`, joint/IMU broadcaster
   packages, and the newly built package prefixes. Check native library linkage
   and current scheduling limits. No hardware launch is part of preparation.
3. Follow [STARTUP_CALIBRATION.md](STARTUP_CALIBRATION.md). The operator must
   explicitly confirm this startup's supported encoder-homing pose and marked
   ring reference **before** launching. The stored permanent desired pose does
   not yet implement automatic physical homing across power loss.
4. From the selected checkout and overlay, launch the focused trial. Once homing
   and fresh joint states are ready, run `python3 scripts/calibrate_robot.py capture`
   and `python3 scripts/calibrate_robot.py status`. Report calibration ID, FR/FL/BR/BL
   wheel homes and storage path before motion. An agent may use
   `--operator-confirmed` only after the actual confirmation for this startup.
5. With torso supported and sticks neutral, verify the real button mapping and
   stop response, then let the operator perform brief supported policy tests
   before floor testing. Reuse calibration when switching within this live
   encoder session; restarting hardware requires a new confirmed capture.

Use a separate recorder when collecting the trial; this minimal launch does not
include the L1/R1 recorder node:

```bash
ros2 bag record -o locomotion-gap9-trial /joy /cmd_vel /wheel_cmd_vel /joint_states \
  /imu_sensor_broadcaster/imu /emergency_stop /rosout \
  /neural_controller_walk_v2/observation /neural_controller_walk_v2/policy_output \
  /neural_controller_walk_v2/position_command \
  /neural_controller_wheel/observation /neural_controller_wheel/policy_output \
  /neural_controller_wheel/position_command
```

## Evidence and limits

Training and export provenance are in `hardware_testing/walk_gap9_2026-09-11`
and `hardware_testing/wheel_gap9_2026-09-11`. Hardware joint/interface facts come
from `ros2_ws/src/pupper_v3_description/description/components.xacro` and
`ros2_ws/src/control_board_hardware_interface/src/control_board_hardware_interface.cpp`.
Button dispatch is implemented in `ros2_ws/src/joy_utils/src/estop_controller.cpp`;
velocity timeout in `ros2_ws/src/cmd_vel_mux/src/cmd_vel_mux_node.cpp`.

Software fixtures do not establish physical stability, geometry fit, wheel
direction, battery condition, motor temperatures, or stop response. Both policies
received qualitative operator feedback at 0.20 m/s on September 12; the 0.40 m/s
wheel change still requires a native Pi rebuild and physical trial.
