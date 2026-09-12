# Deterministic alignment: preparation and first trial

Latest status: see [September 12 lab baseline](hardware_testing/keyframe_align/LAB_BASELINE_20260912.md). The operator reported successful supported and repeated floor trials with position PID. Earlier velocity-control and pending-test descriptions below are historical. New gap-trained walking/wheel exports are preserved alongside alignment; alignment geometry has not yet been adapted to that gap.

**The earlier e6192ab physical release is withdrawn.** The corrected candidate
is for a supported startup/stop verification first, not immediate full alignment.
Read [the correction and staged test](hardware_testing/keyframe_align/STARTUP_FAULT_FIX.md)
before following the launch procedure below. The [incident](hardware_testing/keyframe_align/INCIDENT_20260911.md)
and its missing physical evidence remain documented.

This candidate is on **codex/keyframe-robot-integration**, based on robot-code
e3e1d97. The controller is a separate plugin; existing walking, wheeled and v5
policies remain available. Do not merge the older alignment development hardware
stack into this checkout. Runtime revision b9e2214 has passed all 13 selected
software checks in both the isolated Pi checkout and `/home/pi/robot-code-leglift`.
It is installed in the latter with the three local hardware/launch edits preserved.
The correction has not started hardware or performed motion.

The motion core/configuration comes from alignment source **a110b09**.
[source_manifest.json](hardware_testing/keyframe_align/source_manifest.json)
checks the exact copied files. The new plugin loads no network and performs no
training. Its legacy `model_path` parameter names the keyframe configuration JSON.

## Behavior and hardware contract

The eight proximal joints receive absolute position references; the four hubs
receive velocity commands from encoder-angle feedback. Proximal gains are 5/0.25;
hub gains are 0/0.35; fault/stop requests zero position, velocity, effort, kp and kd. This removes holding torque; it is not an electrical power disconnect.
Canonical joint order is FR, FL, BR, BL, each with joints 1, 2, 3. Commands are
stand=0, FL=1, FR=2, BR=3, BL=4. The deterministic update runs at 520 Hz, without
policy decimation. It uses the same coordinates as the audited C++ simulator.

Sources: [robot-info wheel hardware profile](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/robot_info/HARDWARE.md),
current [components.xacro](ros2_ws/src/pupper_v3_description/description/components.xacro),
and [hardware read/write implementation](ros2_ws/src/control_board_hardware_interface/src/control_board_hardware_interface.cpp).
The older robot-info leg-profile third-joint stops do not apply to these continuous
hubs. The source preflight checks CAN channels, axes, origins, limits and gains
against the existing reviewed geometry. These are implementation/model checks,
not new measurements of the physical robot.

The shared [startup calibration](STARTUP_CALIBRATION.md) supplies live-session
homes. Target is `wrap(home + pi)`, per the user's marked-ring convention. X never
captures home. Activation refreshes only holding angles from current encoders.
Reactivation clears pending commands and completion flags while reusing valid
calibration from the same encoder session. A fresh hardware activation requires a
new confirmed capture. Heating remains manual.

Rotation requires estimated floor clearance >10 mm, conservative wheel spacing
>10 mm, body spacing >5 mm, tilt <0.12 rad, angular speed <0.3 rad/s, and stable
qualification for 0.2 seconds. The floor bound now uses the lowest support-wheel
bottom with modeled radius bounds, rather than the highest unloaded support. It
assumes level rigid ground and the modeled wheel geometry. Do not use it as a
terrain/contact sensor.

Shift/lift/recenter references last 1.5 seconds each; landing lasts 3 seconds.
Filters may extend them. Nominal shift plus lift is about 3.7 seconds, and a full
180-degree cycle including lowering/verifying is about 20.5 seconds. Smaller
starting-angle errors take less time. Large touchdown drift invalidates success;
it does not trigger a grounded rotation retry. Normal command changes first lower
the old wheel. The 48-second attempt watchdog also lowers and latches a retry block.

Invalid encoders/quaternions, missing hardware IMU age, IMU age over 0.1 seconds,
invalid commands, excessive tilt, or update gaps over 0.04 seconds latch the
zero-torque request until lifecycle reactivation. Fault reasons are logged and appended to status. The hardware interface does
not expose a separate per-joint sample timestamp; finite encoder checks and manager
cadence are not proof of fresh motor feedback after a lower-level transport fault.

## Software evidence and pending target checks

The final simulator passed all 22 cases: nominal, eight IMU bias directions, four
friction/random-angle cases, four combined cases, and cancellation in each motion
phase. Final all-wheel error was at most 0.47 degrees; actual modeled rotation
clearance was at least 19.7 mm; descent speed near ground was below 24.4 mm/s.
Final tolerance is currently 0.035 rad (~2 degrees), an engineering test threshold
whose adequacy for physical reshaping still needs verification.

[W&B Media and full audits](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/e276b6aaa0a74a5b)
include failed development cases. Nine core tests and the standalone x86/ARM64
full-sequence tests pass. See [software validation](hardware_testing/keyframe_align/SOFTWARE_VALIDATION.md)
for the ROS/installed-overlay checks and exact environment evidence.
Native x86 ROS checks and all 12 selected native Pi CTests pass, including the
calibration and joystick integration tests that failed under ARM64 emulation.
The Pi build used commit 1494801 in `/home/pi/keyframe-software-test`; clean and
incremental installation checks passed without activating the hardware.

The Pi at `10.140.55.163` was subsequently reached and tested with its actual
Debian 12 / ROS Jazzy packages. Gamepad/device access, scheduling permissions,
selection of the physical-trial checkout, physical pose and motion remain pending.
The inventory found no gamepad and a realtime-priority limit of 0. Passing software
tests is not a Pi realtime timing measurement. Read the
[robot-info pre-lab checklist](https://github.com/TundTT/Pupper_animation/blob/e9b0417/robot_info/PRE_LAB.md).

## Before deployment, while the motor stack is stopped

1. Inspect target identity and local changes in `/home/pi/robot-code-leglift`.
   Preserve the prior Pi homing/launch edits; do not use the other robot's checkout.
   Record current commit, installed overlay and local patch before updating.
   `bash scripts/keyframe_target_inventory.sh` performs read-only inspection of
   versions, prefixes, gamepad names, running managers, services and scheduling.
   Save its output and compare with the recorded build environment before travel.
2. Review the candidate diff against e3e1d97. Bring over this integration branch,
   not the alignment development branch. Do not replace a running library. A
   deployment still requires the user's authorization.
3. Run `bash scripts/prepare_keyframes.sh`. It builds and tests without starting
   hardware or modifying calibration; failures must be resolved before launch.
   Retain a clean ARM64 target build and an incremental installation check. Verify
   the selected overlay and actual spawner parser, not just the ROS distro name.
4. Check gamepad identity, launch services and realtime-priority permissions under
   the intended user. The previous Pi session reported no FIFO permission; this
   work has not changed the Pi's scheduling configuration. Keep the legacy D-pad
   launcher from starting a duplicate stack.
   A candidate PAM limits file is provided at
   `hardware_testing/keyframe_align/90-quadmorph.conf` for the recorded user `pi`.
   If inventory confirms the same zero-priority limit, review existing limits and
   install it under `/etc/security/limits.d/` during authorized target preparation,
   then use a fresh login and verify `ulimit -r`. A systemd-launched stack instead
   needs equivalent service limits. Confirm actual FIFO scheduling with `chrt -p`
   after the separately authorized startup; a configuration file is not proof.
5. Record release commit, configuration and installed library hashes and logs.
   Rollback is the recorded previous revision plus its preserved local patch and
   matching built overlay. Restore only with motion stopped; do not mix libraries
   and configuration from different revisions.

## First physical session

### Quick tuning between trials

Motion settings are in `ros2_ws/src/neural_controller/launch/keyframe_config.json`.
Wheel alignment now uses `wheel_control_mode: "position_pid"`. The controller sends
absolute encoder-angle targets and zero desired wheel velocity; the motor's PD
loop performs correction. `wheel_position_kp: 4.0` and `wheel_position_kd: 0.15`
are the next lab gains: P was raised from 2 after the rear wheels settled slowly. P=4 has not yet been physically validated. These JSON values supply the
actual wheel gains; the generic YAML wheel gain slots remain activation defaults.
No outer wheel-speed PI loop remains. A near-target integral now supplies bounded feed-forward torque: `wheel_integral_ki: 0.5` Nm/(rad�s), `wheel_integral_limit_nm: 0.10`, and `wheel_integral_window_rad: 0.10`. It accumulates only after the angle ramp finishes, outside the alignment tolerance, inside that window, and below the alignment speed threshold. It resets on target crossing, closed rotation gates, cancellation/timeout, large disturbances, stop and reactivation. Accepted bias is frozen through lowering/holding to avoid dropping the correction abruptly; it does not accumulate there. All three settings are in the JSON, and Ki=0 disables further integral accumulation after reload.

The angle target follows a quintic ramp bounded by `wheel_speed_limit` (0.5 rad/s)
and `wheel_acceleration_limit` (1.2 rad/s²). These limit the angle trajectory;
they are not velocity commands. Targets use the nearest equivalent calibrated
angle in the current encoder revolution. Closed rotation gates discard pending
angle demand; reopening starts a fresh ramp from the measured angle.

Completion tolerances are configurable too: `alignment_angle_tolerance_rad`
(0.025), `landing_angle_tolerance_rad` (0.035),
`alignment_speed_tolerance_rad_s` (0.08), and `alignment_settle_seconds` (0.5).
`hold_error_limit_rad` (0.10) releases an aligned target after a large disturbance.
The first 28 status fields retain their indices; wheel velocity fields 8–11 stay zero.
Field 22 now reports the active wheel integral torque in Nm. Read `motor_commands` for actual wheel
angle targets and gains. Previous velocity-controller simulation results do not
validate this new control law. The focused position test exercises ramp limits,
angle wrapping, gate pause/resume, reverse correction after 0.051 rad overshoot,
and automatic lowering on all four wheels using idealized feedback.

`rotation_floor_clearance_m` is **0.005** (5 mm), changed at the operator's request
after the supported FL trial reported about 9 mm and was blocked by the old 10 mm
threshold. This is a hardware trial adjustment; the earlier simulation audit used
10 mm. Lift poses, wheel/body margins and all other settings are unchanged.

For later numeric-only edits, with the controller unloaded/stopped, copy this JSON
to `ros2_ws/install/neural_controller/share/neural_controller/launch/keyframe_config.json`.
No C++ rebuild is needed. Reloading the controller reads the file. A fresh whole-stack
startup still requires confirmed homing pose and fresh calibration. Keep the edited
source JSON with the trial record; its changed hash is an intentional tuning change
and must be recorded in the source manifest before running the strict preparation
checker. Do not rerun the whole build/test suite for a numeric-only iteration.

Read STARTUP_CALIBRATION.md first. Obtain the operator's explicit confirmation of
the supported encoder-homing pose and marked rings **before** fresh startup. A
request to test is not that confirmation. Then, from the reviewed target checkout:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/local_setup.bash
ros2 launch neural_controller keyframe_trial.launch.py
```

This launch performs hardware homing and spawns the keyframe controller inactive.
After homing, capture and inspect shared calibration using the documented workflow:

```bash
python3 scripts/calibrate_robot.py capture
python3 scripts/calibrate_robot.py status
```

An agent may append `--operator-confirmed` only after the operator's actual
confirmation for this startup. Report calibration ID, homes and storage path.
Check fresh sensor data, stop availability and loop timing before motion.

Record a bag before the first authorized motion:

```bash
ros2 bag record -o keyframe-first-trial \
  /joint_states /imu_sensor_broadcaster/imu /joy /rosout \
  /keyframe_align_command_index /neural_controller_keyframe_align/alignment_status \
  /neural_controller_keyframe_align/motor_commands
```

X first activates the controller into its entry/stand sequence. Allow all four
wheels to settle. The next X requests FL; later presses request FR, BR, BL. Start
with one wheel and inspect clearance/rotation/lowering before continuing. Repeated
presses during activation cannot skip wheels; stop takes precedence over X.
The controller lowers before acting on a changed wheel request.

A normal soft return to stand is:

```bash
ros2 topic pub --once /keyframe_align_command_index std_msgs/msg/Int32 '{data: 0}'
```

PS requests zero torque and controller deactivation; Options reactivates after the
cause is resolved and the operator is ready. Do not substitute a controller switch
or shutdown for normal lowering. Reentry can retry a completed sequence using the
same valid calibration; restarting hardware is unnecessary.

`alignment_status` is a 28-value array at 20 Hz. Its first 25 values preserve C ABI v2: eight proximal
commands; four wheel velocities; phase; active command; completed mask; gate mask;
timeout/failure retry command (-1 otherwise); floor/wheel/body margins; active target
error; attempt seconds; integral contribution; authority; failed-wheel mask.
Phases are ENTRY=0, SHIFT=1, LIFT=2, ROTATE=3, LOWER=4, RECENTER=5, HOLD=6, STOPPED=7.
Wheel masks use FR/FL/BR/BL bits 0/1/2/3. Gate bits are trajectory=1, floor=2,
wheel=4, body=8, tilt=16, angular speed=32. Stop status remains published, so a
stale previous motion status cannot be mistaken for continuing authority.

Indices 25–27 add fault code, supplied update period and IMU age. The separate
`~/motor_commands` topic reports position/velocity/effort/kp/kd for all twelve
joints in canonical order. See STARTUP_FAULT_FIX.md for codes and interpretation.
