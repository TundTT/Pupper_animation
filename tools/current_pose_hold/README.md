# Current-position hold (no calibration)

User-requested measurement/positioning mode. It reads current SPI-frame encoder
positions and holds those exact values. No homing, model offsets, pose restoration,
policy, or calibration-file writes. It cannot establish a policy calibration.

Build on the robot: `bash tools/current_pose_hold/build.sh /tmp/quadmorph-current-hold`

Run: `/tmp/quadmorph-current-hold/current_pose_hold --hold`

All 12 targets are captured once after a stationary valid-feedback window. Gains
ramp over 3 seconds to kp=7.5, kd=0.35. PS on the connected DualSense releases;
SIGINT/SIGTERM/SIGHUP also release. The normal ROS stack must be stopped: this
mode holds the same exclusive `/tmp/control_board_hardware.lock` throughout.
Joystick loss, invalid transport/feedback, excessive drift/speed/estimated torque,
or loop delays stop the hold and send zero gains/enable flags. Feedback freshness
has the existing SPI driver's limitations (no per-motor firmware sample counter).
No claim of rigid mechanical locking or crash-proof firmware shutdown is made.

The core and its original test were reused from `robot-code`'s
`ros2_ws/src/measurement_hold/` at 6b07745. The standalone adapter uses this branch's
unchanged `rt_spi.cpp`. Tests cover fixed targets, ramp, stop latch, bad feedback,
timing, channel/winding preservation and disabled shutdown packets.

Initial hardware run reached HOLDING for all 12 channels. Calibration JSON SHA256
was unchanged before/after enabling. This mode was explicitly authorized without
the normal gravity calibration procedure; normal policy startup gates remain intact.
