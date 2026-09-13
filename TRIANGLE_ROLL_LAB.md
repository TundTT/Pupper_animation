# Simultaneous inverted-triangle roll: hardware feedback trial

This is the **all-four roll onto the tips**, followed by bounded settling and a
hold. It is separate from the old sequential `inverted_triangle_trial` launch.
No walking controller is spawned or activated. Heating remains manual.

The motion is ported from simulation commit
[`221e166`](https://github.com/TundTT/Pupper_animation/commit/221e16681952bb9350bd2c1019c2b092c347432a):
2 s initial hold, 12 s simultaneous forward roll, 32 s settling. Roll damping is
doubled; settling restores the base gains. The final target then freezes.
Activation adds a separate 2 s gain ramp and waits for an explicit start command.
The 46-second sequence is a feedback trial, not a claim of formation robustness.
The PC's 36 original dynamics passes and 5/7 formation passes describe simulation;
some unequal rigid shapes still fail support. The runtime does not measure feet
touching the floor, load support or collision clearance. DONE means the command
sequence finished, not that the robot successfully stood.

## Physical setup and mapping

Read [STARTUP_CALIBRATION.md](STARTUP_CALIBRATION.md) before any hardware startup.
An online robot or permission to build is **not** confirmation for homing/motors.
Reuse valid calibration in an already-live encoder session. Never restart merely
to switch controllers, and never overwrite a loaded controller library.

Before motion, confirm four cooled **rigid triangles**, 9 mm additional outward
spacing, and the assembly/payload corresponding to the backpack model. Round
wheels do not reproduce this ground-contact test. Confirm that wires and the
physical sweep are clear. Keep a support/catch available that does not obstruct
rolling; stop/fault releases torque and can let the robot fall.

There are two separate references: ordinary supported startup homing, and the
triangle's point-up test start. With controllers inactive, match the triangle
start to the model and capture its mapping. The proximal angles are FR/FL/BR/BL
`[1,-0.29], [-1,0.29], [1,-0.29], [-1,0.29]` radians. Model hub angles are
`[2.14159265359,-2.14159265359,2.14159265359,-2.14159265359]`.
The left model winding differs from the old sequential plan by one whole turn;
the physical starting orientation is the same. These are **not raw hardware
targets**. The mapping preserves actual encoder turns and is tied to this plan
and the live startup calibration. No automatic approach or encoder zero change.

## Build and launch

Correct Pi checkout: `/home/pi/robot-code-leglift`; the other robot's
`/home/pi/pupperv3-monorepo` must remain untouched. Source preparation is:

```bash
cd /home/pi/robot-code-leglift
bash scripts/prepare_triangle_roll.sh
```

This builds/checks a separate `ros2_ws/install-roll` overlay without launching
hardware. It refuses a running hardware stack. Preserve the existing install
for rollback; a stack shutdown/restart requires the supported physical procedure.

Only after physical startup confirmation, with the old stack safely stopped:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/pi/robot-code-leglift
source ros2_ws/install-roll/local_setup.bash
python3 scripts/check_triangle_roll.py --install-base ros2_ws/install-roll
ros2 launch neural_controller triangle_roll_trial.launch.py
```

Launch homes the hardware. The new controller is spawned **inactive**. Complete
the shared startup capture/status procedure in a second sourced terminal. Report
calibration ID, four wheel homes and storage path. Then separately position and
confirm the rigid triangle start with all motion inactive:

```bash
python3 scripts/inverted_triangle_reference.py capture --roll
python3 scripts/inverted_triangle_reference.py status --roll
```

An agent may add `--operator-confirmed-inverted-start` only after actual physical
confirmation. The separate `triangle-roll-map.json` preserves startup calibration
and cannot reuse the old sequential mapping. Replace only after an explicit new
reference confirmation. The reported map identifies its calibration and plan.

## Run one trial

Verify fresh gamepad input and actual PS stop (index 10), with motors inactive.
Record `/joint_states`, `/imu_sensor_broadcaster/imu`, `/joy`, `/emergency_stop`,
`/triangle_roll/command`, `/neural_controller_triangle_roll/status` and
`/neural_controller_triangle_roll/motor_commands` with `ros2 bag record`.
Confirm topic names against the installed stack.

Keep the robot supported during activation and its gain ramp:

```bash
ros2 control switch_controllers --activate neural_controller_triangle_roll --strict
ros2 topic echo /neural_controller_triangle_roll/status
```

Wait for READY (`data[0] == 1`), verify the base-supported start is stable, and
only when the operator is ready start one continuous roll:

```bash
ros2 topic pub --once /triangle_roll/command std_msgs/msg/Int32 '{data: 1}'
```

PS or command `-1` latches a stop and releases torque. Releasing PS does not
resume. Commands 2–4 from the old sequential trial are rejected. There is no
mid-roll pause or automatic repeat. Do not activate walking afterward: the new
triangle mapping has not been integrated into the walking observation frame.
After the trial, support the robot before deactivation; preserve the recording.

Status fields: `[state, completed, phase, motion_seconds, fault, max_tracking_error,
estimated_PD_torque, period, IMU_age, joint_tracking_settled, estimated_load_FR,
estimated_load_FL, estimated_load_BR, estimated_load_BL, settling_seconds]`.
States: 0 gain ramp, 1 READY, 2 RUNNING, 4 DONE/hold, 5 FAULT. Phases: 0 initial
hold, 1 roll, 2 settling, 3 final hold. Neither field 9 nor the load estimates
proves foot support. Fault numbers retain the old controller's enum: timing 1,
sensors 2, tilt 3, tracking 4, speed 5, estimated torque 6, stop/gamepad 7,
activation 9, invalid command 10.

## Compatibility and deliberate hardware differences

The existing C++/ROS 520 Hz stack is reused, with no MuJoCo, GPU or Python
inference on the Pi. Fixed nominal chain transforms, hinge references, COMs and
tip vertices are exported into C++; a 520-row sequential fixture compares its
support correction with the pinned Python/MuJoCo implementation. The runtime
uses measured q/qd, command history and body-frame projected gravity only.

Source-backed hardware claims: `robot-info` commit
`e9b04173b034d595a85e6147d73a45cdfe9393e3`, `robot_info/HARDWARE.md`, describes joint
order and the BNO055/xyzw interface. Current
[components.xacro](ros2_ws/src/pupper_v3_description/description/components.xacro)
and [hardware interface](ros2_ws/src/control_board_hardware_interface/src/control_board_hardware_interface.cpp)
are authoritative for this branch's continuous hubs, gain limits and PD-estimated
effort; the older robot-info leg hard stops do not apply to this configuration.
[read_sensors](ros2_ws/src/neural_controller/src/keyframe_controller.cpp) supplies
normalized body-frame negative-Z gravity from the xyzw quaternion.

The simulator clips total torque at 3 Nm; the hardware interface's effort clamp
does not clamp the full PD demand. This trial therefore faults above **1.5 Nm
estimated PD demand** rather than pretending the simulator's torque clamp exists
on hardware. Other runtime guards: 8 degree tilt, 2 rad/s measured joint speed,
0.15 rad hub / 0.25 rad proximal tracking error, 100 ms IMU age, 500 ms gamepad
age, and invalid/nonpositive/>10 ms controller periods. At nominal 520 Hz the
motion and correction command-rate limits match the simulation. Timing/jitter
on the target Pi remains a hardware measurement, not inferred from laptop tests.

Feedback references are bounded to +/-0.075 rad around walking defaults; commands
are bounded to +/-0.2 rad and software position limits. The final target is held
after 32 s; support adaptation does not run indefinitely. Gain ramp, start gate,
stop behavior, final frozen hold and the 1.5 Nm guard are hardware additions and
do not inherit simulation acceptance. Record first-test behavior before tuning.
