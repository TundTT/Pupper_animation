# Native Pi triangle preparation

Software deployed and verified on the intended Pi, September 12, 2026. No hardware
stack was started, calibration captured, or motor motion requested during this
preparation. The gamepad was connected and the operator pressed/released PS.

Base release: `ac3d824072e801dbb0a115c20b04c2a84ebbca8f`, plus the PS-stop correction
recorded by source hashes in [the native manifest](triangle-native-manifest.json).
The core trajectory, original triangle tip and additional 9 mm gap are unchanged.

## Installed location and rollback

- Selected checkout: `/home/pi/robot-code-leglift`, branch `robot-code`.
- Prepared overlay: `/home/pi/robot-code-leglift/ros2_ws/install-triangle`.
- Clean build: `/home/pi/robot-code-leglift/ros2_ws/build-triangle`.
- Previous checkout, local edits, build and install copied in full to
  `/home/pi/robot-code-leglift-backup-before-triangle-ac3d824` before updating.
  Local edits also retained in the named Git stash `before-inverted-triangle-ac3d824`.
- The old `ros2_ws/install` and `ros2_ws/build` were not rebuilt. They must not be
  sourced for this triangle trial. Some old install entries are absolute symlinks;
  rollback must restore the saved checkout/build at its original path while the
  hardware stack is stopped, rather than launching directly inside the backup.
- The saved repaired hanging preset matches the previous Pi copy byte-for-byte.
  No files under `/home/pi/.local/state/quadmorph` were replaced.

For this prepared trial, after the operator's actual startup-position confirmation:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/pi/robot-code-leglift
source ros2_ws/install-triangle/local_setup.bash
ros2 launch neural_controller inverted_triangle_trial.launch.py
```

Starting this command homes the hardware. Follow
[STARTUP_CALIBRATION.md](../../../STARTUP_CALIBRATION.md) and then the separate
[triangle start mapping](../../../INVERTED_TRIANGLE_LAB.md). All motion starts
inactive. There was no valid live encoder session at the end of software setup.

## Checks completed

- Native Debian 12 / ARM64 / GCC 12 / Python 3.11 / installed ROS Jazzy clean
  six-package build: passed in 3 min 50 s. Incremental preparation also passed.
- All 3 calibration test groups, all 8 selected controller/launch groups,
  the joystick group and 4 triangle-reference Python cases passed.
- The installed plugin lifecycle test verifies held PS prevents activation,
  PS during operation releases all command fields, release cannot resume motion,
  and malformed gamepad input invalidates freshness. It also verifies calibration,
  map, sensor and first-leg behavior using fake hardware interfaces.
- Actual Pi spawner parsing and substitution resolution passed. The stop node
  uses index 10, with activation/stop-release button bindings disabled.
- Installed package prefixes, source/plan hashes, 9 mm geometry, joint/CAN order,
  modes, gains, limits and linked-library dependencies passed.
- Actual device `/dev/input/js0`: DualSense Wireless Controller, 8 axes and
  13 buttons. Kernel mapping identifies PS/BTN_MODE as index 10; the operator's
  press and release were observed there. `/dev/input/js1` is the motion sensor.
- A motor-free ROS input check on separate DDS domain 96 received 1,582 messages
  in approximately 12 s, maximum receipt gap 50.63 ms. Autorepeat was configured
  at 20 Hz; additional device events explain the higher average rate.
- The `pi` SSH session permits realtime priority 60; `chrt -f 50 true` succeeded.
  SPI/I2C device permissions and user groups were inspected. Actual controller
  scheduling, sensor freshness and loaded 520 Hz timing remain startup/test checks.
- No controller manager or legacy D-pad launch process was running; legacy
  launch services were inactive. The other robot's checkout was not modified.

The initial launch test failed because ROS returns a tuple for the button-index
sequence, while the new assertion expected a list. The assertion now compares
sequence contents. The failure is retained in
[the clean-build log](triangle-native-clean-build.log); the complete successful
incremental preparation is in [the final log](triangle-native-preparation.log).
No production safety thresholds were loosened. Native calibration and joystick
tests also resolve the outstanding target-test gap left by ARM emulation.

## Compatibility evidence and remaining physical work

Hardware/software comparison uses `robot-info` at
`e9b04173b034d595a85e6147d73a45cdfe9393e3`, specifically
[`robot_info/HARDWARE.md`](https://github.com/TundTT/Pupper_animation/blob/e9b04173b034d595a85e6147d73a45cdfe9393e3/robot_info/HARDWARE.md),
[`SOFTWARE.md`](https://github.com/TundTT/Pupper_animation/blob/e9b04173b034d595a85e6147d73a45cdfe9393e3/robot_info/SOFTWARE.md), and
[`PRE_LAB.md`](https://github.com/TundTT/Pupper_animation/blob/e9b04173b034d595a85e6147d73a45cdfe9393e3/robot_info/PRE_LAB.md).
Its older third-joint hard-stop table is not the current deployed continuous-hub
profile; the actual `robot-code` description and installed checker are the source
for this trial's limits. The PS binding is verified from today's physical input,
not inferred from an older joystick comment.

Before motion: obtain the startup/homing-position confirmation, capture the live
calibration, then confirm the rigid inverted assembly, installed 9 mm gaps and
exact triangle starting pose before capturing its separate mapping. Verify fresh
joint/IMU feedback and actual scheduling. Activation requires external support
during its gain ramp. First loaded test is rear-right only, followed by inspection.
Stops release torque; the support must catch the robot. Physical clearance, loaded
motion and a walking-policy handoff have not been validated by this preparation.
