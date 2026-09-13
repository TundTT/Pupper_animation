# Native Pi preparation

Built on the Pi at source `3115c63`, in `/home/pi/robot-code-leglift`, using the
separate `ros2_ws/install-roll` overlay. Six packages built; all 11 selected
CTest tests and two mapping tests passed. This includes the installed roll
plugin with mock joint/IMU interfaces, start/stop and calibration rejection.
The previous `install-triangle` overlay was preserved.

`preparation.json` pins the installed ARM64 library, plan and build log.
`build-and-tests.log` retains warnings and test results. The motor stack was
confirmed absent after preparation. No hardware launch, homing, calibration
capture, controller activation or motor command was performed.

Use `TRIANGLE_ROLL_LAB.md` in the repository root for physical confirmation,
startup, mapping and the explicit single-roll command. Physical assembly and
supported-pose confirmation are still required. DONE is a final command hold,
not proof of four-tip support. Walking remains separate and inactive.
