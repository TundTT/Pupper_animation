# Logged half-turn and second reboot check

User confirmed the supported tips-down startup reference and controller readiness.
Fresh calibration `b987f986ff154f70b0b0d3ee553ddd9a` was captured on boot
`4c724137-6a6f-4084-98c9-62503bcad82f`. No old calibration was restored.

Both commanded hub half-turns completed: up in 18.803 seconds and down in 18.824
seconds. Upper joints retained their calibrated hanging positions. The controller
reported fault 0 and held afterward. Pi health and motion telemetry were streamed
to this laptop in addition to Pi-side files. Sampled supply was 4.930–4.974 V,
temperature 62.0–68.6 °C, and all sampled throttling flags were zero. The blackout
was not reproduced. These one-Hz observations cannot exclude short supply transients.
The two failing GUI/voice services were stopped during this test; the controller
library was loaded at startup rather than reloaded in the live process as previously.
These differences prevent treating this as an exact repetition of the prior failure.

## Operator's subsequent power cycle

The user reported the appendages remained tips-down. Post-boot raw SPI was read
twice on boot `4e6b3aa4-00bd-4e9d-b956-7b52127dbb2e`, with all transmitted bytes
zero, no motor enable, no ROS hardware stack, and no homing/calibration. Both
observations gave identical settled joint positions.

Compared with `settled_tips_down.json`, the eight upper-joint raw readings changed
by at most **0.0454 degrees**. Hub differences were:

| Hub | Raw-coordinate change |
| --- | ---: |
| Front right | −180.43° |
| Front left | −144.06° |
| Back right | +180.12° |
| Back left | +179.95° |

The discrepancy is present in raw feedback before the controller runs; no new
post-boot motion command produced it. It is not a uniform correction of minus
180 degrees. The differences happen to be near integer multiples of 36 degrees,
but gearbox/firmware behavior has not been established from a primary hardware
source; that numerical pattern is a hypothesis for investigation, not a calibration
rule. No automatic persistent reference or hub correction has been enabled.

Upper-reference repeatability is promising across both observed power cycles.
This still does not establish behavior after moving the joints while unpowered.
The before sample was taken during settled HOLD before the user switched power;
small differences can include physical settling. Disabled SPI replies have valid
checksums and become nonzero after startup, but carry no per-motor freshness counters.

Sources: `reboot_comparison.json`, `actual_model.urdf`, `startup.log`,
`settled_tips_down.json`, and the two `disabled_4e6b3aa4*.json` captures in
`hardware_testing/start_pose/encoder_checks/`. Coordinate decoding follows
`ros2_ws/src/control_board_hardware_interface/src/rt/rt_spi.cpp` and
`ControlBoardHardwareInterface::copy_actuator_states`; user statements are the
source for physical configuration and lack of repositioning.
