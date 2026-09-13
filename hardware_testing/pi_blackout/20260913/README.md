# Pi blackout after supported reset, September 13

The operator confirmed the reset reached the correct physical pose, then reported
a black Pi screen and lost SSH. The operator power-cycled afterward and confirmed
the joints were not moved. Investigation did not start ROS hardware, home joints,
enable motors, or replace calibration.

## Blackout investigation

- Prior boot: `3cedd58c-4293-4826-85ae-72a90093b497`; current boot:
  `4c724137-6a6f-4084-98c9-62503bcad82f`.
- Supported reset telemetry reached HOLD at 18.819 seconds with fault 0.
- Saved previous kernel/journal/controller logs contain no identified kernel panic,
  OOM kill, controller exception, undervoltage warning, or orderly shutdown at the
  blackout. Pstore and the queried systemd core-dump journal were empty.
- The new boot reports an unclean journal. This cannot distinguish the original
  incident from the operator's subsequent power cycle.
- Current-boot readings: `get_throttled=0x0`, roughly 70–71 °C, about 7.4 GB available
  RAM, 90 GB free storage, EXT5V snapshot 4.9513 V. These readings do not establish
  voltage, temperature, memory, or kernel state at the previous blackout.
- `pupper-gui.service` and `llm-agent.service` in the OTHER checkout were repeatedly
  failing (missing GUI binary and voice API configuration). Previous-boot counters
  reached at least 564 and 282. Both were stopped for this boot only after inspecting
  their service definitions; neither was disabled or removed. Restore with
  `sudo systemctl start pupper-gui.service llm-agent.service` if desired, after fixing
  their configuration. They are a confirmed background-load issue, not a proven cause.

**Cause remains undetermined.** Do not claim the controller caused or could not
have caused the system-level failure. A future supervised motor run should record
Pi supply/temperature/process state alongside motion telemetry. No crash-reproduction
motion was performed during this investigation.

## Encoder observation without homing

`scripts/read_disabled_spi.py` exchanged 100 all-zero packets over about two seconds,
using the existing driver's wire layout: enable flags, gains, velocity and torque
were all zero. Both SPI devices were unclaimed beforehand. Initial replies were
zero; subsequent replies contained stable finite positions and valid checksums.
The bridge exposes no per-motor freshness timestamp in this packet.

`analyze_encoder_observation.py` reconstructs the pre-shutdown raw frame using
the saved startup log and URDF, and compares final HOLD feedback to median disabled
SPI readings after reboot. It applies the same negative input signs and CAN mapping
as `rt_spi.cpp` and `ControlBoardHardwareInterface::copy_actuator_states`.

- All eight upper joints agree within **0.503 degrees**; seven agree within 0.086.
- All four hub coordinates change by about **minus pi radians** (179.90–180.06 degrees).
- User reports no physical motion; the old snapshot was taken at HOLD entry rather
  than immediately at power loss, so small differences include possible settling.
- This is promising evidence for retaining upper-joint references, and evidence
  that hub coordinates must not be blindly restored across reboot. It does not
  establish the firmware's winding behavior or authorize persistent homing.

Results: `encoder_comparison.json`. Raw after data:
`hardware_testing/start_pose/encoder_checks/disabled_4c724137.json`.
The previous-session calibration remains historical and invalid for a fresh startup.
