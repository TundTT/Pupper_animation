# Transfer hold software check — September 13, 2026

After the operator reported battery loss, the Pi booted with no motor stack.
The existing isolated joint-pose build compiled the updated source on the Pi.
The core and fake-interface ROS wrapper tests passed; the latter uses ROS domain
193 and a temporary fake calibration directory, without physical motor access.

The explicit `--transfer-hold` request ignores torso tilt only in the final HOLD
state. Tests check tilted activation and movement rejection, continued stiffness
through tilt in HOLD, stale-IMU release, PS stop/latch, disconnect release and
deactivation. The first fault is now preserved rather than overwritten by STOP.
The floor roll's tilt protection is unchanged.

The verified library was installed while the hardware manager was absent. Build
and installed SHA-256: `291bf415a52c5cdf96ddf02b7fe962fb80dded0b14720d6588931be1352475d2`.
The previous binary remains on the Pi as
`hardware_testing/joint_pose/libjoint_pose_before_transfer.so`.

No hardware startup or physical transfer was performed by this preparation.
Fresh startup calibration is still required after the battery power cycle.
