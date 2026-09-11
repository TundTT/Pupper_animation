# Alignment v4: coordinated support and staged training

Implementation is ready for PC training on `codex/align-motion-v2`; the branch name is retained, but the motion contract is **v4**. No policy training was performed on the laptop. Follow [AGENT_TRAINING_HANDOFF.md](AGENT_TRAINING_HANDOFF.md) for the staged run and required audits. v2/v3 checkpoints are incompatible with this controller and training environment.

## What failed in v3

The reviewed [v3 run, 17d412eac8004407](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/17d412eac8004407) trained at commit `8be7fd378c05b0dba95aabe1dea6ea981de721e7` for 50,790,400 environment steps. Nominal evaluation completed FL 0/64, FR 2/64, BR 64/64 and BL 64/64; no environment completed all four. Nominal peak ground approach was about 0.219 m/s, above the 0.10 m/s audit limit. Randomized evaluation also had a small conservative wheel-envelope overlap.

The checkpoint video and replay in its original source environment showed distinct problems:

- FL stayed in LIFT until the fixed 32-second command slot cancelled it. In a nominal replay it remained on the ground while BR unloaded; the support reward still counted a wheel hovering 4.2 mm above the floor as contact because the old test accepted clearance below 6 mm.
- FR reached rotation but lost clearance at VERIFY. A separate phase observation let the actor change its posture just when the wheel needed to remain lifted and settle. The next timed command then interrupted it.
- The reference lifted mainly one hip without establishing a coordinated front support posture. The residual action could partly cancel that lift. Passing an isolated static pose check did not establish that the complete physical motion worked.
- Brief impacts were penalized as time-integrated costs, making their contribution small relative to sustained posture reward.

These are simulation findings, not claims about unobserved hardware behavior. Historical artifacts and videos must remain tied to the original source checkout.

## Motion and learning changes

The front references now move all eight proximal joints to a coordinated stance. Their nominal targets were fitted using bounded native MuJoCo pose searches and then checked through the full motion; this was reference fitting, not policy training. Left and right front motions were both simulated. The rear reference uses a 1.05 rad active hip lift. Targets are in [configs.py](training/wheel_align/configs.py); [generate_reference.py](training/wheel_align/generate_reference.py) generates the matching C++ constants, and tests reject drift between them.

The lift follows a smooth 3.5-second support-shift waypoint and a 3-second rise. Lowering starts from the current applied eight-joint target, retains the supporting stance during a 6-second descent, then recenters over 4 seconds. An early interruption uses the current height rather than first raising to a landing waypoint. Position, velocity and acceleration remain continuous through phase changes under the shared command filter.

The actor retains eight residual outputs. Active-joint corrections are limited to 0.04 rad; the active hip correction can only add lift. Support corrections are limited to 0.10/0.15 rad. Corrections fade in during the support shift. VERIFY freezes the last desired posture and presents the same actor phase as ROTATE; LOWER follows the supported descent without new actor corrections. Runtime and Python use the same equations and limits.

Rotation still requires the existing gates: estimated floor clearance >10 mm, wheel spacing >10 mm, body spacing >5 mm, tilt <0.12 rad and angular speed <0.3 rad/s. These thresholds were not relaxed. Training support credit now uses actual simulated floor contact normal force, with a separate penalty for leaving the active wheel loaded at the apex. Contact forces are reward-only information; they are not new policy observations or hardware sensor requirements. A peak-approach cost accumulates increases in impact severity across physics substeps, without shrinking the event by control timestep or refunding it when a wheel lifts again.

The automatic training/evaluation/video scheduler waits for completion, pauses in stand, then requests the next wheel. A 48-second timeout records a failure. Deliberate interruption cases get 20 additional seconds for descent and retry. Real joystick requests still interrupt immediately and lower the old wheel before starting the requested one. The 320-second full-sequence budget is a maximum, with early exit after completion and settling.

Training has three separately logged stages: foundation (5M steps, nominal dynamics and small angle errors), single (10M, full-angle individual wheels with moderate randomization), sequence (35M, wider randomization, shuffled orders and interruptions). Each intermediate stage must pass its own 64-environment audit before transferring the actor and observation normalizer. The next optimizer, critic and step count start fresh. This budget is an initial experiment, not a convergence guarantee.

## Laptop validation

The following checks passed with zero policy optimizer updates:

| Check | Result |
| --- | --- |
| Python suite | 22 tests passed, including the actual video worker, export, reward/contact regressions, interruption recovery and Python/C++ motion parity |
| CPU MJX preflight | 82 observations / 8 actions; batched randomized shapes and all three stage shapes; three finite physical control steps |
| Native MuJoCo, zero residual, full half-turns | All four completed; minimum wheel spacing 16.59 mm; worst approach approximately 0.0421 m/s |
| Native MuJoCo, zero residual, interrupted/retried | All four completed; worst approach approximately 0.0426 m/s; about 50.1 seconds per interrupted wheel |
| MJX automatic full sequence, zero actor, one nominal environment | All four completed and settled in 114 seconds; no timeout, fall or unsafe rotation; simulation gate passed |
| Same MJX sequence measurements | Minimum wheel/body spacing 16.59/18.99 mm; peak approach 0.04215 m/s; final angle error <=0.01472 rad; final wheel speed <=0.00461 rad/s |
| ROS Jazzy build and targeted CTest | Both packages built; all five motion/lifecycle/legacy-policy tests passed |
| RTNeural export fixture | 128 observations matched Brax outputs; maximum action error 4.77e-7 |
| Historical video compatibility | Actual v3 checkpoint rendered four control steps through the v4 uploader worker while importing the unchanged v3 checkout; MP4, CSV and metadata written |

The RTNeural input was an **untrained test fixture**, not the next trained policy. The complete randomized and interruption audits of trained weights remain PC work. Static nominal geometry, simulator contact fidelity, fitted poses and the eventual physical joint calibration remain limits on sim-to-real confidence. The build also reports existing RTNeural convolution-template warnings; the tested alignment network is an MLP. No robot controller selection or deployment was performed.

## Hardware evidence and calibration scope

- Joint order, wheel mode, position/velocity actuator split, gains and encoder/IMU interfaces: [robot-info hardware contract at b311700](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/robot_info/HARDWARE.md). Its leg-mode third-joint stops are not applied to continuous wheel mode. The runtime baseline is cross-referenced against [robot-code components.xacro at 582fd88](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/pupper_v3_description/description/components.xacro).
- Alignment model/meshes originate from [align-hybrid at bfd74cc](https://github.com/TundTT/Pupper_animation/tree/bfd74cc0a8a57f64f316db7b0e26718acc290f41), with the collision/limit corrections described in [ALIGN_MOTION_V2.md](ALIGN_MOTION_V2.md). This is still a wheel-alignment model, not a simulation of polymer deformation or the transformed leg.
- The startup point-ring/opposite-base convention (`base_target = wrap(home + pi)`) comes from the user-provided `quadmorph_reshaping_process.md`. Heating remains manual per the user's instruction. The user confirmed the displayed leg outline/mounting for now and may change it later; this does not independently validate every simulated wheel clearance or material property.
- The current `robot-code` checkout has separate, ongoing physical calibration work. It was left untouched in this task. v4 retains this training branch's startup-home target convention and does not recalibrate on X. Before any future hardware integration, reconcile with the newer calibration implementation rather than overwriting it with this branch's older startup code. Calibration-dependent physical startup is not authorized by the training handoff.

Changes to hardware geometry require updating the model, regenerated bounds/transforms/reference, fresh motion checks and retraining. No extra motors, force sensors or heater automation are assumed by this change.
