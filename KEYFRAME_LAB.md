# Deterministic alignment: preparation and first trial

This candidate is on **codex/keyframe-robot-integration**, based on robot-code
e3e1d97. The controller is a separate plugin; existing walking, wheeled and v5
policies remain available. Do not merge the older alignment development hardware
stack into this checkout. Nothing has been deployed or started on the Pi.

The motion core/configuration comes from alignment source **a110b09**.
[source_manifest.json](hardware_testing/keyframe_align/source_manifest.json)
checks the exact copied files. The new plugin loads no network and performs no
training. Its legacy `model_path` parameter names the keyframe configuration JSON.

## Behavior and hardware contract

The eight proximal joints receive absolute position references; the four hubs
receive velocity commands from encoder-angle feedback. Proximal gains are 5/0.25;
hub gains are 0/0.35; stop sets command/gain values to zero except damping 1.
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
reviewed damping stop until lifecycle reactivation. The hardware interface does
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
The native x86 ROS checks pass and the ARM64 controller build/tests pass, but
calibration and joystick integration checks remain unresolved under ARM64
emulation. Run them successfully on the actual target with the motor stack
stopped before declaring this candidate ready for a physical trial.

The Pi's last recorded address, `10.140.55.163`, was unreachable during this work.
Package-version parity with that Pi, its gamepad/device access, scheduling
permissions, physical pose and motion remain pending. An emulated architecture
check is not a Pi realtime timing measurement. Read the
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
  /keyframe_align_command_index /neural_controller_keyframe_align/alignment_status
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

PS requests the established damping emergency stop; Options reactivates after the
cause is resolved and the operator is ready. Do not substitute a controller switch
or shutdown for normal lowering. Reentry can retry a completed sequence using the
same valid calibration; restarting hardware is unnecessary.

`alignment_status` is a 25-value array at 20 Hz, matching C ABI v2: eight proximal
commands; four wheel velocities; phase; active command; completed mask; gate mask;
timeout/failure retry command (-1 otherwise); floor/wheel/body margins; active target
error; attempt seconds; integral contribution; authority; failed-wheel mask.
Phases are ENTRY=0, SHIFT=1, LIFT=2, ROTATE=3, LOWER=4, RECENTER=5, HOLD=6, STOPPED=7.
Wheel masks use FR/FL/BR/BL bits 0/1/2/3. Gate bits are trajectory=1, floor=2,
wheel=4, body=8, tilt=16, angular speed=32. Stop status remains published, so a
stale previous motion status cannot be mistaken for continuing authority.
