# Audit follow-up training

Run: `workspace/output/walk_2026-09-07_00-08-09`

Log: `mujoco_playground/walk-train-7.log`

W&B: https://wandb.ai/QuadMorph/pupper-leg/runs/rlhxeko3

Detached process PID: 2104013. Exact command and environment are in `launch.json`.
Budget: 200 million requested steps, 8,192 environments, seed 0, GPU 0.
Warm start: `walk_2026-09-06_03-24-30/best_params`; fresh optimizer state.

## Changes

- Action-rate weight -0.05 → -0.30; other reward weights retained.
- Common length scale U(0.90,0.98), per-leg U(0.96,1.04).
- Height reference derived at reset from neutral-pose capsule projections in the
  actual randomized model, accounting for both length and mount tilt.
- Torso-height reward and low-height termination measured along the floor normal.
- Parent-frame mount-tilt description corrected; mechanism and ±0.07 bounds retained.
- Push-free evaluation and 50 fps showcase videos; training still includes pushes.
- Raw action-difference RMS and terrain-relative/reference height metrics added.
- Best checkpoint saved immediately, including the warm start in selection;
  run metadata includes evaluation settings, warm-start hash, and source snapshots.

Action limits, swing reward/cap, air-time and torque weights, ring geometry, and
9 mm bottom allowance were preserved. No phase-shaped trajectory was introduced.

This combines reward and physical-model improvements; it is not an isolated
action-rate ablation. Gait improvement remains to be established from the trained
policy. Compare tracking, survival, action roughness, cadence, swing duration and
touchdown speed under matched commands and push settings. Video is still too
coarse to establish a 60 ms swing waveform; retain 250 Hz instrumentation for that.

## Verification

- Environment, randomization and geometry suite: 15 passed (`tests.log`).
- New length-reference and slope-translation tests: 2 passed (`height_tests.log`).
- Four-second direct MJX standing checks at scales 0.864, 0.94 and 1.0192 pass.
- Neutral references agree with the audit's independently settled heights within
  0.08 mm; two-metre slope travel leaves height reward and survival unchanged.
- 64-step PPO smoke run completes and saves parameters (`smoke.log`); this tiny
  randomly initialized fixture verifies plumbing, not policy quality or PPO tuning.
- Checkpoint callback ordering, regression retention, improvement selection and
  parameter save/load were exercised separately.
- Live full-run configuration, checkpoint hash and source snapshots verified.
- Additional per-slot height-reference and sampled length-bound assertions pass
  (`slot_checks.log`, two tests).
- Full-run initial evaluation completes: reward 32.667, mean episode length 600,
  velocity error 0.068 m/s; the step-zero checkpoint is retained as best. This
  evaluates the starting policy with the new protocol, not training improvement.
- Initial showcase verified as 800 frames at 50 fps (16 seconds).
- First trained checkpoint saved at 14,417,920 steps; parameter leaves are finite
  and the saved parameters differ from the warm start. Training remains running.
