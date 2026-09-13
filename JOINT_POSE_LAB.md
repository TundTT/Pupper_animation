# Supported joint-position controller

`JointPoseController` accepts named joint targets through `scripts/set_joint_pose.py`.
It is for setup on the stand with all limbs clear, not a ground-contact motion
planner. It does not recalibrate joints or establish offsets across power loss.
Read STARTUP_CALIBRATION.md before startup; reuse this session's valid calibration.

For software preparation with the hardware stack stopped, the normal
`neural_controller` package build installs this separate plugin along with the
roll controllers. On this checkout, `bash scripts/prepare_measured_roll.sh` uses
the selected dependency/controller overlays and does not start hardware.
In an already calibrated session, load the plugin inactive using the sourced
overlay (only if it is not already loaded):

```bash
ros2 param set /controller_manager neural_controller_joint_pose.type neural_controller/JointPoseController
ros2 run controller_manager spawner neural_controller_joint_pose --inactive
```

Do not reload the whole manager or restart hardware just to change joint targets.
This command-line wrapper currently requires other motion controllers to be inactive
before a new request; it is not an online joystick or active-target streaming API.

The controller ramps stiffness for 2 seconds, restores upper joints first (at most
0.3 rad/s and 0.8 rad/s² in the target trajectory), then moves hubs (at most
0.5 rad/s and 1.2 rad/s²). Estimated commanded PD torque is capped at 0.6 Nm.
It retains fresh-gamepad/PS stop, finite feedback, IMU age, 8-degree torso tilt,
velocity, tracking and timeout guards. Any fault releases torque and stays latched.
Measured speed and torque limiting are not a guarantee against collisions.

Targets must be within existing policy command bounds. Starting upper joints may
be at most 0.35 rad outside those bounds for bounded inward recovery. This is a
software recovery allowance, not a verified mechanical range. References are
clipped inside the normal bounds, with zero initial position gain and a gain ramp.
The measured pose may not move farther outward by more than 0.015 rad. Unknown
or larger displacement is rejected. Upper moves are at most 1 rad; hub moves at
most 3.2 rad. Tips-up chooses the nearest equivalent winding of the current-session
operator-confirmed tips-up reference; it never silently recalibrates a hub.

After loading/configuring `neural_controller_joint_pose` **inactive**, preview:

```bash
python3 scripts/set_joint_pose.py --preset tips-up
python3 scripts/set_joint_pose.py --joint leg_front_r_1=1.0 --joint leg_front_r_2=0.0
```

Only after actual operator confirmation that the torso is supported and limbs clear:

```bash
python3 scripts/set_joint_pose.py --preset tips-up --execute --operator-confirmed-supported
```

Other motion controllers must be inactive. The command holds at completion; do not
carry an ordinary hold to the floor. For an operator-supported transfer, explicitly
use `--transfer-hold`: tilt is ignored only after the controller reaches HOLD.
Activation and all moving phases still require a level torso. IMU freshness,
PS/disconnect, tracking, speed and timing stops remain active during transfer.
Support the torso's full weight. The separate floor-roll controller retains its
tilt guard; it does not inherit this request setting. Preserve the physical encoder session when
switching controllers. Initial validation uses the actual displaced encoder snapshot
in a simplified damped plant, and checks staged motion, recovery bounds, target speed
and acceleration, PD torque and fault guards. That test is not hardware validation.

## September 13 supported hardware reset

Request `fd313581a9734c0b9253dcb1168477e9` completed on the stand in
18.819 seconds, restoring the approved upper positions and current-session tips-up
hub reference. Final maximum measured tracking error was 0.02204 rad (1.26 degrees),
with no controller fault. Evidence is in
`hardware_testing/joint_pose/runs/fd313581a9734c0b9253dcb1168477e9/`.
The controller remained active holding afterward. This is a supported reset,
not a floor roll or balance result. Calibration and hardware process were preserved.

The first activation was stopped by the command script because it consumed queued
inactive telemetry. The script now subscribes to volatile status after activation,
avoiding that queue. The failed attempt remains on the Pi. Both the core trajectory
test and fake-interface ROS activation/PS/IMU/deactivation test passed before loading.
