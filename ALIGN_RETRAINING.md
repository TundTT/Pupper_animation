> Current work is motion contract v4. Read [ALIGN_MOTION_V4.md](ALIGN_MOTION_V4.md) and [AGENT_TRAINING_HANDOFF.md](AGENT_TRAINING_HANDOFF.md). The material below records earlier revisions.

# Alignment retraining: motion contract v3

Branch: `codex/align-motion-v2`. New runs use `quadmorph-align-motion-v3` metadata. Follow [AGENT_TRAINING_HANDOFF.md](AGENT_TRAINING_HANDOFF.md) for PC setup, one fresh 50-million-step run, three audits, and export. No training was performed on the laptop.

## Diagnosis and evidence

The PC report for `align-motion-v2-seed0-20260910T191937Z` reported 0/64 complete sequences in each audit, approximately pi maximum final angle error, and randomized/interrupted contact approach speeds above 0.10 m/s. The actual trained checkpoint/trace was not available locally during this fix. Consequently, a trained-policy LIFT stall is a hypothesis, not an observed trace. A maximum error near pi does not establish that every wheel rotated to an inverted target.

Source inspection of the previous `training/wheel_align/env.py` at `cf1012a` found posture, hip tracking and safety costs but **no reward for wheel-angle progress, verification, or task completion**. A stationary policy could collect most of the roughly 11 reward units per second without completing alignment. The old nominal-hip tracking reward also opposed active-leg residual corrections. PPO's discount of 0.99 at 52 Hz had an effective horizon of approximately 1.9 seconds, much shorter than a worst-case half-turn at 0.25 rad/s (12.6 seconds before settling).

A native-MuJoCo zero-action probe stayed in LIFT for 25 seconds. Its settled encoder/IMU floor margin was approximately -17.3 mm and tilt 0.173 rad, failing the 10 mm / 0.12 rad gate. Nominal kinematic clearance alone had not established dynamic feasibility. Fixed-pose searches with the old active residual limits found balance/spacing tradeoffs; the best support-corner pose with adequate floor clearance had only approximately 5 mm wheel spacing. These bounded searches do not prove that every v2 policy must fail.

Allowing active residuals of [0.20, 0.12] rad produced a feasible pose. Replaying that fixed diagnostic action through the actual v3 Python supervisor, reference limiter and native-MuJoCo velocity actuators completed a nominal front-left sequence: LIFT 3.19 s, ROTATE 13.63 s, VERIFY 0.46 s, LOWER 4.13 s. At enabled-rotation control samples, minimum estimated floor/wheel/body margins were 10.72 / 10.75 / 23.71 mm. This is a narrow nominal margin and one diagnostic sequence, not evidence of randomized robustness or soft touchdown. The full physics-substep audits remain required.

Reproduce these two probes without learning:

```bash
"$PY" -m training.wheel_align.diagnose
```

The fixed action in that module is a feasibility fixture, not a deployment policy or a physical-robot command.

## Changes

- `rewards.py` adds dense penalties for missing clearance/stability, signed actual wheel-angle progress during safely enabled rotation, and one-shot supervisor-verified and completed events. Apex rewards cannot be farmed by holding forever. Angle alone cannot credit an interrupted descent as completed.
- `env.py` reduces posture reward dominance, tracks the applied active-hip command, doubles clearance/contact penalties, and measures worst near-ground downward speed across **all four wheels** at every physics substep. Termination carries a penalty. Contact speeds are simulated training/audit quantities, not a new onboard sensor requirement.
- `configs.py` and `wheel_align_motion.hpp` enlarge active residual authority from [0.06, 0.04] to [0.20, 0.12] rad. Support speed limits change from [2, 3] to [1, 1.5] rad/s and acceleration from [12, 16] to [6, 8] rad/s² to reduce abrupt support redistribution. These are command bounds, not guarantees on physical impact speed. Active speed/acceleration, three-second lift, four-second lower, joint limits and all rotation safety gates are retained.
- `train.py` increases discounting to 0.999 (approximately 19.2-second horizon). Architecture, 82 encoder/IMU/supervisor observations, eight residual outputs and locked dependencies stay compatible in shape, but **not in action meaning** with v2. Fresh training is required.
- Runtime and export metadata use contract v3. The updated controller accepts legacy v1 and current v3, and rejects v2 rather than reinterpreting its weights. Existing deployment YAML remains on its legacy checkpoint.
- Training metrics and audits report phase time, floor/wheel/body/stability gate blockage, enabled rotation, angle progress, verified/completed events and per-wheel final errors. Nested audit diagnostics are uploaded to W&B summary fields. Online logging and real checkpoint videos in Media remain enabled by default.

## What to inspect on the PC

Preserve `/home/theerawit/Pupper_animation-align-motion-v2` and its failed run unchanged, including its local exporter fix. Use a new v3 worktree from the latest remote branch. Do not bypass checkpoint source hashes to load the failed v2 policy into this environment; historical video upload still uses its original checkout via `log_run --source-root`.

During training, look beyond total reward: `eval/episode_rotation_enabled`, phase and blocked-gate metrics, and completion events should show learning of the whole sequence. Brax episode metrics are accumulated over an episode; they are not automatically seconds. The audit converts phase/blockage/enabled-rotation indicators to mean seconds per environment. Blockage reasons may overlap. A near-zero enabled-rotation duration points to gate failure; substantial rotation with large final errors calls for inspecting the actual target/angle CSV. The trained video is diagnostic, not acceptance.

Use all three existing 64-environment audits after the run. Thresholds remain all four wheels completed and aligned/settled, no unsafe rotations, >5 mm conservative wheel gap, positive body gap, and maximum contact approach speed <0.10 m/s. Failed environments are retained. An export is not permission to deploy. If this experiment fails, return the audits and video instead of silently launching another run or weakening the gate.

## Hardware assumptions and calibration

The current XML and unchanged conservative wheel/body geometry remain the basis of feasibility. The user confirmed this geometry for now; future shin edits require updated geometry checks and retraining. Existing 12 motors, joint order, encoder/IMU interfaces and continuous wheel hubs are grounded in the [robot-info hardware contract](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/robot_info/HARDWARE.md) and [robot-code components.xacro](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/pupper_v3_description/description/components.xacro). The imported model is from [align-hybrid bfd74cc](https://github.com/TundTT/Pupper_animation/tree/bfd74cc0a8a57f64f316db7b0e26718acc290f41); see [the geometry evidence](ALIGN_MOTION_V2.md#geometry-and-source-evidence) for collision-proxy limitations and older conflicting comments.

Startup calibration remains separate from button-triggered hold capture. The `wrap(home + pi)` convention comes from the user-provided `quadmorph_reshaping_process.md`; physical marked-ring orientation still needs user verification. There is no evidence here justifying a pi sign/offset change. Heating remains manual as explicitly requested.

## Validation

Laptop results for this revision: all 14 Python tests passed, including 3,000 Python/C++ control-step comparisons; all five targeted ROS CTest cases passed after building `neural_controller` and `joy_utils` in WSL/ROS Jazzy. RTNeural matched 128 exported untrained fixtures with maximum error 4.77e-7. The runtime also rejected an intentionally obsolete v2 metadata fixture. Generated geometry matched the XML. MJX preflight passed randomized batch shape `[2,82]` and three compiled control steps with zero policy optimizer updates.

Tests cover reward ordering and signed progress, one-shot completion, a four-wheel supervisor/velocity-servo sequence without instantaneous hub-angle substitution, native front-left dynamic feasibility, geometry/FK agreement, export and logging/video regressions (including an actual MP4 rollout and offline W&B media). ROS lifecycle and inference checks use test-only untrained fixtures. None of these replace training or the trained-policy audits.
