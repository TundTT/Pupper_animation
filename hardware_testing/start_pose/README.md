# Saved start reference

## Approved upper policy home (September 13)

`approved_upper_pose.json` records the user's approved motor-1/motor-2 pose:
`[1,0],[-1,0],[1,0],[-1,0]` in FR/FL/BR/BL order. The user visually approved
the hold on the stand and reported that the robot stood when placed on the floor.
This is separate from the hanging encoder-initialization reference below.
Hub/spoke calibration remains independent. The full roll has since been inspected
on the stand; dynamic floor roll/support and walking handoff remain unvalidated.
See `LAB_HANDOFF.md` and the preserved hardware trial records.

The alignment neutral and measured-roll tip-up/tip-down defaults now share
`policy_home.hpp`; the numerical trajectories are unchanged. Roll activation
holds live feedback and START approaches its entry stance smoothly. The existing
small hip preparation (`-0.1,+0.1,-0.1,+0.1` rad) remains before the roll.
This approved target does not establish encoder offsets across battery changes.

During the user's floor-to-stand handling, the live keyframe controller logged
`KEYFRAME FAULT 6: excessive tilt` at Unix time 1789294256.012840795 and latched
zero torque. The implemented threshold is 0.6 rad (about 34.4 degrees).
The exact peak tilt was not recorded. The fault remained latched after returning
level; do not automatically restart it. The roll controller's 8-degree tilt limit
is stricter. Move between floor and stand with motor support intentionally stopped
and torso physically supported, rather than carrying an active test controller.

The current reference is the operator-confirmed hanging pose after minor repairs,
captured September 12, 2026 at 08:01:11 UTC. Calibration ID:
`535afed263c24d7fa65a3c44dc7083d0`. The robot was supported on its stand; after
homing, the operator reconfirmed it was hanging and still. Source evidence is
`repaired_stand_calibration_20260912.json`. The unchanged stationary-data checks
passed after earlier attempts rejected reported-velocity outliers. Both walking
and wheel policies remained inactive throughout capture and replacement.

The earlier successful repeat-floor-trial reference is preserved in
`successful_floor_calibration.json` and the previous preset under `history/`.
The replacement was saved on the Pi and mirrored to this workspace. Live-session
calibration remains at `/home/pi/.local/state/quadmorph/calibration.json` on the Pi;
this repository copy is evidence, not a substitute for a valid live session.

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

September 13 decision: retain per-boot hub/spoke calibration and pause deeper
motor-firmware diagnosis while preparing supervised motion tests. Two unchanged-pose
power-cycle comparisons found consistent upper readings but large hub-coordinate
jumps; see `hardware_testing/pi_blackout/`. Those observations do not establish
upper-reference recovery after arbitrary unpowered movement. Automatic upper startup
still needs a validated coordinate mapping and integration with the calibration gate.
Do not claim this saved preset implements automatic homing or require a broad encoder
investigation merely to repeat an already-calibrated supervised motion trial.
