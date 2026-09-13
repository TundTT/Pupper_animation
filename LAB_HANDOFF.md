# QuadMorph motion handoff — September 13, 2026

Start with `robot-info:ROBOT_INFO.md` and `robot_info/QUADMORPH.md` for durable
physical facts. This file records implementation status, not permission to move
the robot. Read `STARTUP_CALIBRATION.md` at every actual hardware startup.

## What to use

| Task | Entry point | Evidence / boundary |
| --- | --- | --- |
| Move named joints or restore a supported pose | `scripts/set_joint_pose.py`, `JOINT_POSE_LAB.md` | Native Pi controller/stop tests and supported hardware reset passed. Stand setup only; not a floor contact planner. |
| Simultaneous cold triangle roll and settle | `TRIANGLE_ROLL_LAB.md`, `measured_roll.hpp` | Full suspended sequence inspected by user: no rubbing. Physical floor roll/support remains unvalidated. |
| Raw readings before startup | `scripts/read_disabled_spi.py` | All transmitted SPI bytes zero; no homing or motor enable. Replies do not contain per-motor freshness counters. |
| Pi blackout diagnostics | `scripts/pi_health_monitor.py` | One-Hz persistent JSONL and stdout for a laptop SSH copy. Does not command motors. |
| Legacy hub-only alternative | `HUB_ROLL_LAB.md`, `hub_roll_controller.cpp` | Separate diagnostic implementation, not the controller used for the latest successful resets. |
| Measurement-only stiffness | `ros2_ws/src/measurement_hold/` | Arbitrary-pose measurement utility; its pose is not a policy calibration. |

The tested joint-position controller exposes `neural_controller/JointPoseController`
in `libjoint_pose_controller.so`. It restores upper joints before turning hubs,
ramps gains, bounds the move, and holds afterward. A fault releases torque and
latches. Release of the PS button does not restart motion. Never carry an active
controller without accounting for its tilt stop and resulting loss of support.

## Calibration decision

The user accepts calibrating the selected hub/spoke reference at each boot and
does not want a deep motor-firmware investigation to delay motion testing.
The intended workflow is saved upper-joint targets plus per-boot hub calibration.
**Automatic restoration of the upper physical reference is not implemented.**
The present hardware startup still redefines session encoder offsets and requires
the confirmed usual hanging/tips-down setup and shared capture. A saved target is
not a saved, validated cross-boot physical coordinate transform.

Two unchanged-pose reboot observations found upper-joint differences below
0.51° and 0.05° respectively. Hubs jumped by roughly 180° in the first, and by
roughly ±180° / −144° in the second. No fixed reboot correction is justified.
See `hardware_testing/pi_blackout/20260913/` and `repro_4c724137/` for the raw
evidence, comparison scripts, and limits of those observations. Never restore
those historical session IDs or encoder offsets as live calibration.

The approved upper target is `[1,0],[-1,0],[1,0],[-1,0]` rad in FR/FL/BR/BL order,
stored separately in `hardware_testing/start_pose/approved_upper_pose.json` and
shared by `policy_home.hpp`. It was visually approved and statically stood on
the floor; that is not proof of dynamic roll balance or walking readiness.

## Motion evidence and next step

- Measured-shape simulation source: commit `2f34c9ce2ef696d31c804ebf0bcfb7a71bd05349`,
  [W&B rollout](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/360ffcd09344487d).
  Source metadata, native Pi checks and preserved failed build audit are under
  `hardware_testing/triangle_roll/measured_360ffcd09344487d/`.
- Full supported stand sequence: `hardware_testing/triangle_roll/lab/stand_1789296036/`.
- General reset: `hardware_testing/joint_pose/runs/fd313581a9734c0b9253dcb1168477e9/`.
- Logged up/down hub trial: `hardware_testing/pi_blackout/repro_4c724137/`.
  Both half-turns completed; the blackout did not recur. The original blackout's
  cause remains undetermined, not fixed or attributed to a service by this test.

The next useful physical test is the full simultaneous roll onto the tips and
stable floor hold, recorded on video with feedback logging. No new policy training
or broad robustness sweep is needed just to prepare that supervised trial.

Walking handoff is a separate unfinished interface task. The roll maps current
hub encoders into model coordinates; ordinary walking still consumes joint
positions directly, subtracts walking defaults, and sends defaults plus actions.
Before enabling walking, apply one fixed, session-bound hub mapping consistently
to observations and commands, preserve winding, initialize action/history state,
and check continuity of targets/gains during controller transfer. Matching visible
tip orientation alone does not establish matching policy coordinates. First check
zero-command walking hold before any forward walking command. The existing roll
launch deliberately does not start walking.

Detailed encoder diagnosis can stay paused. This coordinate handoff and a floor
support observation are the immediate work needed for an honest roll-to-walk test.
