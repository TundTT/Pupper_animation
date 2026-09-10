> Current startup procedure: [STARTUP_CALIBRATION.md](STARTUP_CALIBRATION.md). Every fresh stack startup needs physical confirmation and saved session calibration before policy activation. Historical X-time capture/recalibration instructions below are superseded.

# Selected long-stride walking policy: hardware test preparation

`policy_walk_v2.json` now contains the selected long-stride checkpoint from
`walk_2026-09-07_16-48-35`. Activate it with the existing
**`neural_controller_walk_v2` / Square (button 3)** binding. This checkpoint is
prepared for its first hardware test; no robot was connected or activated here.

## Exact policy

| Item | Value |
|---|---|
| Training run | `walk_2026-09-07_16-48-35`, W&B `zw331stu` |
| Selected step | **21,626,880**, `params_000021626880` / `selected_params` |
| Checkpoint SHA256 | `f9b30170c1d927c039204a92b4015fff59ebbdc9ede2137aca4878812290ea5e` |
| Export SHA256 | `814d095421ebfbf7d2b8a1cdb4f1fa9d2ccd7c8fb85406e4162ce60bcfd42b2a` |
| Inputs / outputs | 144 / 12, four newest-first 36-value history frames |
| Joint order | FR, FL, BR, BL; hip, abduction, knee within each leg |
| Home angles | `[1,0,-1, -1,0,1, 1,0,-1, -1,0,1]` rad |
| Action scales | `[0.5,0.25,1.1]` repeated four times |
| Position gains | kp **5.0**, kd **0.25**, including initial move to home |
| Startup | 2 seconds to home, then 2 seconds action fade-in |
| Commands | vx ±0.35 m/s, vy ±0.15 m/s, yaw ±0.8 rad/s; upright only |
| Tilt cutoff | 0.65 rad |

Use **`eval/video_selected`** in the
[W&B run](https://wandb.ai/QuadMorph/pupper-leg/runs/zw331stu) to identify this gait.
`best_params` is a later, shorter-stride checkpoint and is not installed here.
The obstacle clip is `eval/video_obstacles_10mm`.

## Interface and simulation changes

This update replaces the weights, inference fixtures, selection manifest and
validation notes. The policy interface matches the previous hardware export:
joint order, home angles, gains, action scales, command limits and startup settings.
The export includes normalization and the tanh output. Runtime targets are absolute
positions `home + scale * action`; no additional normalization, tanh or leg sign
changes are needed.

Training now uses stiff capsules extended to 62.650 mm overall length so their
distal ends are flush with the outer ring. The ring compression reward has zero
weight; ring-side avoidance remains. The ring is still a reward-only geometric
proxy and does not simulate TPU deformation or load sharing. Simulation geometry
is not a hardware calibration offset or an encoder adjustment.

Gravity-droop zeroing, knee raw-reference checks, mechanical end-stop protection,
emergency-stop handling and reset-history behavior remain configured as before.
The entire target envelope fits the hardware soft limits, with at least 0.4216 rad
margin to knee hard limits. This checks targets, not measured-state overshoot.

## Checks and measured performance

- Read-only preflight passed: exact export hash, joint order, scales, gains,
  Square binding, command envelope and joint target limits.
- The controller's vendored RTNeural Eigen float32 backend matched original Brax
  deterministic inference on **128 observations**, including home/droop startup,
  reverse and turns. Maximum action difference was **1.43051e-6** (less than
  1.6e-6 rad after scaling). Command-contract and reset-history checks passed.
- Full ROS/Jazzy build and hardware validation remain pending on the robot; this
  preparation machine has no ROS installation.

At 0.20 m/s forward on the rigid model with friction 2, this checkpoint has
2.17 Hz cadence, approximately 98 mm front strides, 224 ms median major swing
duration, and 9.2 / 11.3 mm front swing peaks. It preserves much of the preferred
older policy's longer stride while improving front clearance. Later checkpoints
lifted higher but shortened strides and increased command roughness.

All **57/57 flat trials** survived: 13 commands × 3 seeds at friction 2 and
6 commands × 3 seeds at friction 1, each 8 seconds. On separate native MuJoCo
courses it completed **3/3 five-mm and 3/3 ten-mm obstacle trials**, with no falls.
The preferred older policy completed 1/3 ten-mm trials under the same physics.
These are small simulation samples, not hardware success rates.

The obstacle course has three 30 mm-wide bars at a 0.20 m/s command and friction 2.
Ten-mm trials still accumulated 860 capsule side-contact records and 4.74% mean
ring-side overlap. Contact records repeat at 250 Hz and are not distinct impacts;
ring overlap is a geometric proxy, not force. Completion means the torso cleared
the course, not that every foot cleanly cleared every bar.

Current evidence: [hardware_testing/walk_2026-09-07_long_stride](hardware_testing/walk_2026-09-07_long_stride).
The older `hardware_testing/walk_2026-09-07` directory documents the previous policy.
Repeat local checks with:

```sh
python3 scripts/check_walk_policy.py
bash scripts/test_walk_policy.sh
```

## Bring-up on the robot

Use this branch update in the existing project checkout; confirm its remote and
branch first, as described in [SSH.md](SSH.md). Policy JSON uses Git LFS: the file
must contain weights, not a small LFS pointer. Preserve any robot-local calibration
changes when updating. Do not start another control process if the D-pad launch
service already started one; avoid D-pad Up during a manual launch.

Before boot/calibration, support the torso with **all legs hanging freely and
untouched**. Booting with feet supported assigns the wrong gravity-droop zeros.
Check all four `Homing reference check OK` messages. A `HOMING MISMATCH` is currently
an error log, not an automatic homing abort: stop and resolve the hanging pose or
calibration before activating a policy. Do not change reference numbers just to
silence the message.

Build the updated controller on the robot, then verify the installed policy:

```sh
# From this repository root, after obtaining this branch update:
git branch --show-current
git lfs pull
python3 scripts/check_walk_policy.py
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPython3_EXECUTABLE=/usr/bin/python3 -DCMAKE_CXX_FLAGS=-g0
# Continue only after colcon reports success.
source install/local_setup.bash
cd ..
python3 scripts/check_walk_policy.py --package-share "$(ros2 pkg prefix neural_controller)/share/neural_controller"
ros2 launch neural_controller launch.py
```

Launch starts walk mode **inactive**. With zero stick commands and the torso
supported, press **Square** to activate. Confirm the two-second move to home is
smooth. Use the existing emergency-stop button (configured `/joy` index 12),
verifying the actual device mapping before relying on it. A terminal stop is:

```sh
ros2 topic pub --once /emergency_stop std_msgs/msg/Empty '{}'
```

**X switches to the older locomotion policy; it is not an emergency stop.**
Verify the stop during a supported startup as well as after startup before floor
walking. Reactivation clears the controller's stop latch and starts the init ramp
again, so keep the robot supported for this check.

After a successful supported check, place the robot on a clear, level floor with a
spotter and activate at zero command. Begin with 5–10 seconds of standing, then
short straight commands around ±0.10 m/s. Progress to reverse -0.15/-0.20 m/s and
separate left/right turns around ±0.2 rad/s, returning to zero between trials.
Only progress toward ±0.35 m/s or ±0.5 rad/s turns after the slower tests are clean.
Watch for toe/tip support, TPU side loading, asymmetric lift, knee end-stop activity,
and torso oscillation. End the trial if calibration, contact, or motion looks wrong.

For command-controlled trials, avoid competing joystick publishers. Always send
zero at the end: a direct `/cmd_vel` publisher is not a timed motion primitive.
This controller does not add a new timeout for direct commands in this port.

## Record and assess

Use the existing bag recorder (L1 start / R1 stop), or run
`bash scripts/record_all_except_raw_camera.sh` in a separate terminal. Capture
`/cmd_vel`, `/joint_states`, `/neural_controller_walk_v2/observation`,
`/neural_controller_walk_v2/policy_output`,
`/neural_controller_walk_v2/position_command`, IMU data, and a side-view video.
Check the actual policy rate:

```sh
ros2 topic hz /neural_controller_walk_v2/observation
ros2 topic echo /neural_controller_walk_v2/observation --once
```

The existing manager setting is 520 Hz / repeat_action 10, **52 Hz nominal**;
its established scheduling aims for approximately 50 Hz in practice. Training was
50 Hz. Record the actual rate and jitter; do not treat the nominal setting as a
measurement or globally change timing while testing this policy.

Log the branch commit, export hash, robot calibration changes, stop response,
standing behavior, command sequence, forward/reverse/left/right quality, visible
ring loading, motor temperatures, and bag/video locations. Hardware results remain
**pending**; append them here after the first test.
