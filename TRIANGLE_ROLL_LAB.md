# Simultaneous inverted-triangle roll: hardware feedback trial

This is the **all-four roll onto the tips**, followed by bounded settling and a
hold. It is separate from the old sequential `inverted_triangle_trial` launch.
No walking controller is spawned or activated. Heating remains manual.

The measured-shape candidate comes from W&B run
[360ffcd09344487d](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/360ffcd09344487d).
The C++ executor ports its measured-state entry, 12 s simultaneous roll, and
32 s bounded support settling. Entry lasts at least 2 s and waits for measured
position/speed to settle, with a timeout. Every phase shares one speed and
acceleration filter. Roll damping is doubled; settling restores base damping.
The final target freezes without activating walking.

Activation ramps gains for 2 s while holding the **current measured position**.
It waits in READY for a separate START command before approaching or rolling.
The suspended diagnostic uses the same entry and roll, then filters toward the
nominal standing target for 32 s with ground-load correction disabled. It is a
motion inspection on a stand, not a test of ground support.

The saved baseline uses provisional rigid profiles with 57 mm above and 35 mm
below the hub, retaining the 9 mm outward spacing. Operator measurements were
(55,37), (57–58,34), (60,32), (56,37) mm. These are cold-shape dimensions, not a
reconstructed full contour. The load estimator retains the nominal geometry
used in the passing run. It estimates loads from joints and IMU, without contact
sensors. DONE means the sequence finished, not proof that the robot stood.

## Physical setup and mapping

The operator-approved upper home is recorded in
`hardware_testing/start_pose/approved_upper_pose.json`. Alignment neutral and the
roll's nominal tip-up/tip-down poses share `policy_home.hpp`: motors 1/2 are
`[1,0],[-1,0],[1,0],[-1,0]`. Hub mapping remains separate. This source consolidation
does not change the validated trajectory: START still includes the small hip
preparation below. Approval of the static stance is not validation of the roll.

Read [STARTUP_CALIBRATION.md](STARTUP_CALIBRATION.md) before any hardware startup.
An online robot or permission to build is **not** confirmation for homing/motors.
Reuse valid calibration in an already-live encoder session. Never restart merely
to switch controllers, and never overwrite a loaded controller library.

Before motion, confirm four cooled **rigid triangles**, 9 mm additional outward
spacing, and the assembly/payload corresponding to the backpack model. Round
wheels do not reproduce this ground-contact test. Confirm that wires and the
physical sweep are clear. Keep a support/catch available that does not obstruct
rolling; stop/fault releases torque and can let the robot fall.

Normal startup establishes the proximal encoder frame; the point-up reference
establishes hub orientation. The measurement-only stiffness session is **not**
a model calibration: its arbitrary startup pose must not be reused as if its
reported proximal angles were physical model angles.

After valid normal calibration, the roll accepts shoulders within 0.15 rad and
hips within 0.20 rad of `[1,0], [-1,0], [1,0], [-1,0]` (FR/FL/BR/BL), subject to
hardware limits. This includes the documented hanging hips at about +/-0.18 rad.
These are entry eligibility limits, not evidence every pose succeeds on the floor.
With motion inactive, the operator confirms tips up and captures hub offsets.
The old exact +/-0.29 rad hip pose is no longer required. START gradually moves
the hips toward `[-0.1,+0.1,-0.1,+0.1]` and holds the measured hub positions
until entry settles. Proximal offsets are never guessed from an unknown pose.

The point-up model hub reference is `[pi-1,1-pi,pi-1,1-pi]`. Mapping preserves
actual encoder turns and is tied to this plan and the current calibration.
Forward roll decreases right hub angles and increases left hub angles by about
pi radians. A fixed target winding is selected once; live angles are never
wrapped. Old mappings are rejected by the new schema and plan hash.

## Build and launch

Correct Pi checkout: `/home/pi/robot-code-leglift`; the other robot's
`/home/pi/pupperv3-monorepo` must remain untouched. Source preparation is:

```bash
cd /home/pi/robot-code-leglift
bash scripts/prepare_measured_roll.sh
```

This reuses the already-prepared `install-roll` dependencies and builds the new
controller in `ros2_ws/install-measured-roll`, without launching hardware.
It refuses a running hardware stack. Preserve both prior installs for rollback.
The native preparation script expects the saved `measured-parity.csv` generated
by `hardware_testing/triangle_roll/prepare_measured_fixture.py` from the passing
run's audit artifact. The prepared Pi contains that fixture.

Only after physical startup confirmation, with the old stack safely stopped:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/pi/robot-code-leglift
source ros2_ws/install-roll/local_setup.bash
source ros2_ws/install-measured-roll/local_setup.bash
python3 scripts/check_triangle_roll.py --install-base ros2_ws/install-roll --controller-install-base ros2_ws/install-measured-roll
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
States: 0 gain ramp, 1 READY, 2 RUNNING, 4 DONE/hold, 5 FAULT. Phases: 0 measured
entry, 1 roll, 2 settling, 3 final hold. Neither field 9 nor the load estimates
proves foot support. Fault numbers retain the old controller's enum: timing 1,
sensors 2, tilt 3, tracking 4, speed 5, estimated torque 6, stop/gamepad 7,
entry timeout 8, activation 9, invalid command 10.

## Supported stand diagnostic

Use `ros2 launch neural_controller triangle_roll_stand.launch.py` from the
`install-roll` dependencies plus `install-measured-roll` controller overlay,
only after the same confirmed startup/homing procedure.
This launch still homes hardware; it is not a motor-free preview. The torso must
be level and securely supported, with all limbs clear of the stand. Verify the
gamepad stop and capture both the live startup reference and point-up mapping
before activation, as for the ground trial. START performs the bounded approach.

This explicit, read-only `stand_only: true` profile retains the two-second
activation gain ramp and waits for command 1. It approaches the measured entry,
rolls all four limbs together for at least 12 seconds, then filters toward the
nominal standing targets for 32 seconds and holds. Ground-load estimation never
runs. The normal `triangle_roll_trial.launch.py` uses bounded load adaptation
during that settling phase. Both profiles share the same plan and physical
mapping. The suspended profile is a diagnostic, not proof of ground balance.

Status now appends field 15: 1 means stand diagnostic, 0 means ground sequence.
Field 14 (support elapsed seconds) must remain zero throughout the stand test.
No walking controller is launched. A supported test can check rotation direction,
clearance and tracking, but cannot establish balance or ground support.

## Compatibility and deliberate hardware differences

The existing C++/ROS 520 Hz stack is reused, with no MuJoCo, GPU or Python
inference on the Pi. Fixed nominal chain transforms, hinge references, COMs and
tip vertices are exported into C++; a 520-row sequential fixture compares its
support correction with the pinned Python/MuJoCo implementation. A separate
24,051-step replay compares entry, roll, settling commands and damping against
the actual passing measured-shape recording. The runtime
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
