# Alignment v5: bounded corrections, clearance recovery and checkpoint selection

Current branch: `codex/align-motion-v2`. Follow [AGENT_TRAINING_HANDOFF.md](AGENT_TRAINING_HANDOFF.md) on the PC. No policy training ran on the laptop. The motion contract is now **v5, 83 observations and 8 actions**; older policy contracts are rejected. The extra observation is the controller's remaining residual gain, not an additional physical sensor. The current `robot-code` calibration work remains separate and untouched.

## Evidence motivating the change

The [v4 foundation run](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/5c2f3adfb8704f65), source `0b55cf1`, finished at 5,734,400 steps but completed only 60/64 requested wheels. All four failures were front-left. The source-matched replay reproduced all four failures; halving actor output or setting it to zero completed 16/16 front-left cases. In the full-sequence video, FR stayed lifted but could not rotate because the FR–BR conservative spacing reached 6.53 mm, below the 10 mm rotation gate. One foundation failure also froze a marginal posture in VERIFY and could not recover.

That evidence supports limiting corrections around the feasible reference. It does not establish that zero residual is sufficient under physical disturbances or all randomized models. Earlier checkpoints passed the smaller internal evaluation while the final one regressed, motivating selection by task audits instead of always transferring final weights.

## Control and reward changes

- Active residual limits are halved to 0.02/0.02 rad; support limits are halved to 0.05/0.075 rad. Active hip residuals still cannot cancel the lift. Reference poses, the 6.5-second lift, 10-second supported lowering, joint limits, velocity and acceleration limits remain as in v4.
- At the lift apex, floor/wheel/body margins below 12/12/7 mm, or loss of the existing stability gate, trigger a smooth two-second fade of the currently requested correction toward the reference. Recovery remains latched for that wheel, including VERIFY, and resets at the next lift. This prevents a bad correction from repeatedly returning when clearance recovers. The same command filter continues to limit applied velocity and acceleration.
- Rotation still requires the original strict 10 mm floor, 10 mm wheel and 5 mm body gates, tilt <0.12 rad and angular speed <0.3 rad/s. Triggering recovery is not permission to rotate, and the reference is not guaranteed safe for every hardware disturbance. If reference recovery cannot satisfy the gate, the simulation task still times out as a failure.
- Observation index 82 exposes `residual_gain` in [contract.py](training/wheel_align/contract.py), [export.py](training/wheel_align/export.py) and the C++ runtime. Python/C++ parity includes this gain. Diagnostics log gain and time spent recovering.
- The reward now penalizes sustained residual magnitude (0.5 times the sum of squared active-policy outputs per second) in addition to action changes. The penalty applies while corrections have authority, not during scripted descent/recovery. A blocked apex also costs 6 reward units per second, in addition to existing margin deficits, angle delay and timeout costs. Weights are an initial training choice; PC audits decide whether learning improved.

## Audits, media and selection

Audits and videos use the scheduler's actual terminal rest condition: all requested slots handled, the motion in IDLE/HOLD, and 104 control ticks of settling. Training autoreset follows this condition too. A timeout remains a failure while its old wheel finishes lowering. The single-wheel budget is increased to 64 seconds to include the 48-second timeout, full descent and settling. The full-sequence maximum remains 320 seconds.

Headline final angle/speed errors now concern requested wheels only; all-wheel angle error is separately labeled. Audit passes also explicitly require zero timeouts and completed lowering/settling. Training's internal evaluation uses 64 environments. Intermediate training videos show their actual stage plus a separately labeled full-angle sequence diagnostic.

[select_checkpoint.py](training/wheel_align/select_checkpoint.py) tests positive-step checkpoints newest-first. Foundation and single require balanced 64-case audits on two seeds; sequence requires nominal, randomized and interrupted 64-case audits. Failed candidates and final `mjx_params` remain intact. The first passing candidate is copied to `selected/mjx_params` with matching configuration, SHA-256, step and audit provenance. If all candidates fail, the selector returns an error and the curriculum stops. It never falls back to params_0 or silently approves an internal reward score.

Selected checkpoint videos use `policy/selected_<scope>` in the same W&B run. Selection status, stage and step have separate summary fields; prior final-checkpoint failures are not overwritten. Selected weights, audit reports and video traces are artifact files. The PC handoff transfers/exports the selected checkpoint, not the last checkpoint. Each new stage still starts a fresh optimizer, critic and step count with transferred actor and normalization.

## Local validation

- All 27 Python tests passed (the existing suite plus five new regressions, with affected media/selection tests rerun after final diagnostic changes). They cover geometry, native MuJoCo full/interrupted motions, Python/C++ parity, export, real video generation, logging, recovery continuity/latching, timeout descent and checkpoint selection/holdout rejection.
- ROS Jazzy built successfully and all five targeted runtime/lifecycle/legacy tests passed after regenerating the motion fixture for v5. The old v4 fixture was rejected as incompatible. RTNeural matched an untrained 83-input export across 128 observations with maximum action error 4.03e-7; verification of future trained weights remains pending.
- CPU preflight traced randomized batches and all curriculum stages with 83 observations, and ran three physical steps with finite outputs and zero optimizer updates.
- In 16 nominal front-left cases, the previously failing v4 actor used only as a diagnostic input to the v5 controller completed 16/16, with minimum spacing 12.51 mm and peak approach 0.03734 m/s. This deliberately uses its first 82 inputs and is not a compatible v5 training checkpoint or deployment export.
- Full-sequence MJX diagnostic: the old actor as a stress input, all-positive saturated outputs, and zero outputs each completed all four wheels without timeout or physical termination. Minimum gaps were 11.77, 11.98 and 16.60 mm; peak approach speeds were 0.03470, 0.04590 and 0.04214 m/s. Recovery activated in both nonzero cases and stayed inactive for the zero actor. These are nominal scenarios, not the final randomized trained-policy audit.
- The actual audit function passed a four-environment, balanced zero-actor foundation check in 24 simulated seconds. An injected permanently closed rotation gate correctly failed with one timeout, ran the full 10-second descent and settling, and ended at 60 seconds. This verifies failure-path audit coverage without lowering the rotation threshold.

The required PC curriculum and its trained-policy audits remain outstanding. There is no hardware approval or deployment in this change.

## Hardware and geometry sources

The actuator order, wheel mode, gains and encoder/IMU interface are grounded in [robot-info HARDWARE.md at b311700](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/robot_info/HARDWARE.md), cross-checked against [robot-code components.xacro at 582fd88](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/pupper_v3_description/description/components.xacro). Wheel-mode continuous hubs are distinct from the third-joint stops listed for leg mode. The imported model, collision bounds and reference-fitting limits are documented in [ALIGN_MOTION_V4.md](ALIGN_MOTION_V4.md).

Startup home plus pi still follows the user's reshaping notes. Heating remains manual per the user's instruction. The user confirmed the current displayed leg shape while anticipating future edits; that does not validate all simulated wheel margins. Any geometry change requires updated model/bounds/references and new validation. Integrate the separate current robot-code calibration implementation before any future physical startup; do not overwrite it with this training branch's older startup code.
