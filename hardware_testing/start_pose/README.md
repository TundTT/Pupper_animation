# Saved start reference

The user selected the calibration from the successful repeat floor trial on
September 12, 2026 as the desired permanent startup reference. This means the
captured pre-test home, not the robot's later aligned pose or an arbitrary live
encoder sample. Original evidence is `successful_floor_calibration.json`, from
Pi trial `/home/pi/keyframe-floor-repeat-IzzTlX/calibration.json`.

`start_pose.json` is the editable desired reference, with all 12 named joint
angles in radians. It is not a live calibration or a hardware encoder offset.

From the repository root:

```bash
python3 scripts/start_pose.py status
# Explicitly select a new, verified saved calibration as the desired home:
python3 scripts/start_pose.py save --from-calibration /path/to/calibration.json --replace
```

Replacing the preset preserves its previous contents under `history/`. Direct
JSON edits are also possible; retain the joint names and radians. Neither command
moves the robot, changes running controller gains, or modifies live calibration.

## Automatic startup remains unfinished

The user's new request supersedes the preference for manually recapturing home
each startup. It does not establish a persistent physical encoder coordinate
system. Current `control_board_hardware_interface.cpp`, `do_homing()`, assigns
`zero_position = measured_position - homed_position` at each startup; the deployed
description has zero homing torque thresholds and gains. Therefore copying the
same saved numerical home into a new session cannot recover a moved wheel's
physical orientation. The Pi's local homing log prints its pre-offset readings;
the old comment claiming repeatable absolute feedback is not sufficient evidence
of retained wheel output angle after power loss.

Next implementation prerequisite: identify motor/firmware feedback semantics and
verify a repeatable physical angle mapping across a battery power cycle, including
wheel revolutions and any gearbox ambiguity. Then use persistent offsets and a
bounded return trajectory to this preset instead of redefining zero at launch.
Do not claim the saved preset already implements automatic homing. The working
alignment controller and its P/I/D configuration remain unchanged.
