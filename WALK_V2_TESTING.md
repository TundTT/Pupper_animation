> Current startup procedure: [STARTUP_CALIBRATION.md](STARTUP_CALIBRATION.md). Every fresh hardware-stack startup requires operator-confirmed physical positioning and saved live-session calibration before policy activation.

# Selected walking policy for the 9 mm assembly gap

The active `policy_walk_v2.json` now contains the selected **12,779,520-step** checkpoint from `walk_2026-09-11_23-02-17` (W&B `j3xez9z5`). This replaces the old walking weights in the same file. **Square (button 3)** still activates `neural_controller_walk_v2`.

The active `policy_wheel.json` is the September 11 wheel gap retrain. **Triangle (button 2)** still activates `neural_controller_wheel`. X retains the existing alignment binding. Controller launch/configuration, calibration, gains, timing, and emergency-stop behavior are unchanged.

## Exact walking policy

| Item | Value |
|---|---|
| Training run | `walk_2026-09-11_23-02-17`, W&B `j3xez9z5` |
| Selected step | **12,779,520**, `params_000012779520` / `selected_params` |
| Checkpoint SHA256 | `878376a33dfcfaf4f312410c38482f9afe6b21621f5b442051eba4c1c72dafa3` |
| Export SHA256 | `854ac8ba4ffc305079b7f6f7b52187a211413c3cdb18f0de016dd819ff2450a8` |
| Inputs / outputs | 144 / 12; four newest-first 36-value history frames |
| Joint order | FR, FL, BR, BL; unchanged within each leg |
| Home angles | `[1,0,-1, -1,0,1, 1,0,-1, -1,0,1]` rad |
| Action scales | `[0.5,0.25,1.1]` repeated four times |
| Position gains | kp 5.0, kd 0.25, including initial move to home |
| Startup | 2 seconds to home, then 2 seconds action fade-in |
| Commands | vx ±0.35 m/s, vy ±0.15 m/s, yaw ±0.8 rad/s; upright only |
| Tilt cutoff | 0.65 rad |

Use **`eval/video_selected`** in the [W&B run](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/j3xez9z5). This is the policy the user reviewed and approved. The final and best-training-reward checkpoint is different and is not the deployed branch artifact.

## Validation and measured performance

- Read-only preflight passed: export hash, active bindings, gains, action scales, and joint target envelopes.
- The controller's RTNeural Eigen float32 backend matched 128 independent JAX reference actions with maximum error **1.73599e-6**. Command-contract and reset-history checks passed. The active wheel policy also passed its 128-case check.
- **39/39** eight-second selection trials and **39/39** twelve-second trials with fresh seeds and different floor friction completed. In the latter comparison against the old policy on the modified XML, yaw-rate error is **26.4% lower**, XY velocity error **4.9% higher**, and normalized action changes **7.1% higher**.
- At forward 0.2 m/s and friction 2: cadence **2.40 Hz**, front strides **90.6/91.2 mm**, major swing time **200 ms**, front swing peaks **12.4/14.1 mm**. Both old and selected policies completed 3/3 forward gait trials. The new gait is slightly quicker and shorter than the old one.
- The selected video covers the full 32-second command showcase. Hardware and obstacle-course validation have **not** been performed for this checkpoint. Historical obstacle successes for the September 7 policy do not establish this checkpoint's performance.

Runtime targets remain `home + scale * action`; the export folds observation normalization and includes the tanh output. No extra normalization or joint-sign conversion is required. The 9 mm assembly shift is training geometry, not an encoder calibration offset. The rigid flush foot model and gait settings are retained from the warm-start baseline.

Current evidence: [hardware_testing/walk_gap9_2026-09-11](hardware_testing/walk_gap9_2026-09-11). The earlier `walk_2026-09-07_long_stride` directory preserves the previous release's measurements.

```bash
python3 scripts/check_walk_policy.py
bash scripts/test_walk_policy.sh
bash scripts/test_wheel_policy.sh
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

**X retains the existing alignment binding; it is not an emergency stop.**
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
