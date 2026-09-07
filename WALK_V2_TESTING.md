# Selected walking policy: hardware test preparation

The September 7 reverse/spin improvement is installed as `policy_walk_v2.json`,
using the existing **`neural_controller_walk_v2` / Square (button 3)** binding.
This checkpoint is simulation-validated and prepared for its first hardware test.
No robot was connected or activated during this preparation.

## Exact policy

| Item | Value |
|---|---|
| Training run | `walk_2026-09-07_04-04-36`, W&B `61am09cq` |
| Selected step | **129,761,280**, selected by direction-specific evaluation |
| Checkpoint SHA256 | `d240a2cd345a3e00eec49fdd2ac8b9697aabb191b6383e68271099edd25acd1f` |
| Export SHA256 | `d296c9d8b08a1b46f756c89362505a248c5921b633a4d55778632d7fbaa65b04` |
| Inputs / outputs | 144 / 12, four newest-first 36-value history frames |
| Joint order | FR, FL, BR, BL; hip, abduction, knee within each leg |
| Home angles | `[1,0,-1, -1,0,1, 1,0,-1, -1,0,1]` rad |
| Action scales | **`[0.5,0.25,1.1]` repeated four times** |
| Position gains | kp **5.0**, kd **0.25**, including the initial move to home |
| Startup | 2 seconds to home, then 2 seconds action fade-in |
| Commands | vx ±0.35 m/s, vy ±0.15 m/s, yaw ±0.8 rad/s; upright only |
| Tilt cutoff | 0.65 rad (about 37 degrees), matching training |

The complete export is copied byte-for-byte from the selected training artifact;
normalization and the tanh output are already included. Do not apply normalization,
tanh, mirrored-leg sign changes, or a second action scaling at runtime. The robot
controller sends **absolute joint targets** `home + scale * action`. The training
XML's actuator inputs were offsets from home; do not send those offsets directly
as hardware positions.

[Video on W&B](https://wandb.ai/QuadMorph/pupper-leg/runs/61am09cq): choose
**`eval/video_selected`**, not the training-reward-best panel. The selected file is
129M even though training finished at 173M.

## What this port changes

- Replaces the previous walking weights and uses their matching knee scale.
- Honors exported command bounds before constructing observations. Global joystick
  gains stay as configured for the other controllers; commands outside this walk
  policy's range saturate at the training limits. Exported upright orientation is
  held fixed instead of accepting `/cmd_pose` tilt commands.
- Checks the configured joint order/action types against the export on load.
- Clears observation/action history on reactivation and fills history from the
  first measured frame, matching the training reset convention.
- Checks emergency stop before both the move-to-home phase and policy decimation.
  Previously those early returns could bypass the stop handler. This shared
  controller correction also applies to the other neural instances.
- Sets only walk mode's init kp to 5.0 and tilt cutoff to 0.65 rad.

The current `robot-code` calibration is retained: **gravity-droop zeroing, not
hard-stop homing**. Homing velocities/kp/thresholds remain zero. The four knees
retain their raw-reference checks and mechanical end-stop protection. Do not
restore the old September 5 guide's hard-stop instructions. Its old L2 activation
and old gait/standing measurements are superseded by this document.

The entire tanh target envelope fits the current hardware soft limits. Knee
commands span right **[-2.1, 0.1]** and left **[-0.1, 2.1]** radians, with at least
**0.4216 rad** to the configured knee hard limits. This checks commanded positions;
measured-state overshoot and physical assembly/calibration still need testing.

## Checks completed here

- Read-only preflight passed for policy hash, joint order, scales, gains, button
  binding, command envelope, and every joint target versus hardware limits.
- Compiled the same vendored **RTNeural Eigen float32 backend** used by the ROS
  controller. Across 64 reference observations generated from the original JAX
  checkpoint, maximum absolute action difference was **1.81e-6** (under 2e-6 rad
  after the largest scale). Includes home/droop startup inputs and both turn and
  reverse commands. Contract/reset-history tests passed too.
- The training workspace passed all 30 tests before this port.
- **Full ROS/Jazzy plugin build and hardware tests remain to be run on the robot**:
  this preparation machine has no ROS installation. The standalone inference test
  does not certify controller lifecycle, CAN communication, or timing.

Evidence is in [hardware_testing/walk_2026-09-07](hardware_testing/walk_2026-09-07).
To repeat the local checks (Python needs PyYAML; C++ needs a C++17 compiler):

```sh
python3 scripts/check_walk_policy.py
bash scripts/test_walk_policy.sh
```

Simulation comparison: 13 commands × 3 seeds × 10 seconds, first second discarded,
nominal model without pushes/noise/latency. Both policies survived 39/39. Reverse
cadence fell from 5–6 Hz to 2–3 Hz with comparable or better reverse tracking.
Spin speed error fell 39% left / 63% right. Modeled ring-side overlap fell from
1.85% to 0.0023%; randomized training evaluation had about 1.4%, so the clean nominal
number does not cover every geometry. Forward speed error rose 5–8 mm/s. The TPU
ring is a reward-only geometric proxy; load sharing/deformation is not simulated.

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
