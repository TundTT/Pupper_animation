# QuadMorph motion handoff — September 13, 2026

Latest source integration: `COMBINED_MOTION_LAB.md` describes X roll, Triangle
walking and Circle wheels in one exclusive controller stack, using the new
backpack policies. The walking frame now preserves per-session offsets and
full-turn winding, with a final-roll command/gain snapshot for initialization.
The Pi now has this source built and checked in `ros2_ws/install-combined`;
see `hardware_testing/combined_motion_2026-09-13/pi-deployment.json`.
The combined hardware stack has now been tested: the operator verified the
floor flip and reported successful walking tests, with mapped walking activation
observed in the Pi log. See the combined-motion hardware session summary for
evidence boundaries and startup/button behavior. Fresh calibration is still
required after each hardware restart.

Floor evidence update: `hardware_testing/triangle_roll/lab/floor_1789304250787618321/`
completed the 30-degree-tolerance roll and settling without a fault. The operator
reported that it worked well and stood on the tips, and supplied a floor photo.
The earlier trial `floor_1789303481142662027` was assisted during transition;
do not label that earlier attempt an unassisted balance result.

Start with `robot-info:ROBOT_INFO.md` and `robot_info/QUADMORPH.md` for durable
physical facts. This file records implementation status, not permission to move
the robot. Read `STARTUP_CALIBRATION.md` at every actual hardware startup.

## What to use

| Task | Entry point | Evidence / boundary |
| --- | --- | --- |
| Move named joints or restore a supported pose | `scripts/set_joint_pose.py`, `JOINT_POSE_LAB.md` | Native Pi controller/stop tests and supported hardware reset passed. Stand setup only; not a floor contact planner. |
| Simultaneous cold triangle roll and settle | `TRIANGLE_ROLL_LAB.md`, `measured_roll.hpp` | Operator verified suspended and floor sequences, then walking handoff; broad hardware robustness remains unvalidated. |
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

The floor roll and walking handoff have now been exercised. Further repeat tests
can use X followed by Triangle without chat confirmation between those button
presses, after the initial supported setup and reference are confirmed. No new
policy training is required just to repeat this supervised test.

Later September 13 update: the first floor attempt did run, stopping at 7.713 s
on the original 0.15 rad hub tracking guard. See the explicit tracking-retry
section in TRIANGLE_ROLL_LAB.md and the preserved floor evidence. The user wants
30 degrees of hub tracking tolerance with unchanged gains and trajectory; walking
is deferred. No successful physical roll-to-stand is established by that failure.
The supported joint-position controller now offers `--transfer-hold`, ignoring
torso tilt only after reaching HOLD while preserving its other stop checks.

Historical pre-integration issue (resolved by the combined launch): the roll maps current
hub encoders into model coordinates; ordinary walking still consumes joint
positions directly, subtracts walking defaults, and sends defaults plus actions.
Before enabling walking, apply one fixed, session-bound hub mapping consistently
to observations and commands, preserve winding, initialize action/history state,
and check continuity of targets/gains during controller transfer. Matching visible
tip orientation alone does not establish matching policy coordinates. First check
zero-command walking hold before any forward walking command. The existing roll
launch deliberately does not start walking.

Detailed encoder diagnosis can stay paused. Use the combined launch and its
fixed walking frame, rather than the legacy ordinary walking entry, for the
tested roll-to-walk workflow.
