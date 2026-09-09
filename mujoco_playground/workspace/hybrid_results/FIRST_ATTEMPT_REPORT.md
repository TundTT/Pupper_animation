# QuadMorph Phase 1 — first hybrid attempt

The eight-position-joint policy learns lift/lower and completes full align sequences, but this first attempt is below the 45/64 reference. The main observed bottleneck is a mismatch between the learned lifted pose and the conservative encoder readiness guard introduced in this environment. No further training or guard tuning was performed after this evaluation.

| Evaluation | Full sequence | Survival | Falls | Drift p95 | Calibrated held error p95 |
|---|---:|---:|---:|---:|---:|
| Named historical favorite | 45/64 | 64/64 | 0 | 4.80 cm | 0.0248 rad |
| Hybrid, randomized physics, 0.25 rad/s slew | **21/64 (32.8%)** | **62/64** | **0** | **4.60 cm** | **0.0410 rad** |
| Same checkpoint, nominal physics | 20/64 | 64/64 | 0 | 3.47 cm | 0.0362 rad |
| Same checkpoint, randomized physics, 0.5 rad/s slew | 23/64 | 62/64 | 0 | 5.08 cm | 0.0411 rad |

The default remains 0.25 rad/s. Doubling speed only recovered two sequences and increased drift in this paired sample. This is an evaluation sensitivity check of one frozen policy, not another training round. No rotation below 5 mm actual wheel clearance was recorded in any of these three audits. Every counted success also passed actual touchdown checks and the stricter 0.06 rad final-error check. The two randomized-physics terminations were excessive drift beyond 15 cm, not falls.

## What was trained and checked

One corrected PPO run from random parameters: 50,790,400 actual environment steps, 4,096 parallel environments, ELU policy [128,128,128], nonprivileged critic [256,256,256], seed 0. The configured budget was 50 million; PPO batch rounding explains the excess. The corrected run took 255 seconds including compilation and training evaluations on one RTX PRO 6000 GPU. It used no imitation, teacher, or existing checkpoint. The reward contains no wheel-reaching, angle-error, rotation-progress or completion term.

An earlier invalid launch was stopped after its 45,711,360-step checkpoint. Brax's default autoreset restored the pose/observation but retained task timer, command, phase and wheel snapshots; later episodes therefore trained stand. A corrected wrapper restores the complete task state while preserving PPO truncation and episode statistics. The smoke test crosses repeated episode boundaries and verifies command/phase/target/snapshot restoration and timeout flags. Both launches remain on W&B, with the invalid one clearly labeled. Two setup/compilation interruptions did not constitute additional training experiments.

Smoke checks also verified 8 actions / 51 observations, continuous wheel joints, separate position/velocity actuator gains, mixed-actuator randomization, persistent calibration under interruption/retry, periodic-angle PD invariance, observation independence from world translation, and JIT physics. The final training sources/model match their recorded SHA-256 snapshot. Python compilation and whitespace checks pass.

## Evaluation protocol and limits

The audits use 64 robots with independent calibration and accumulated wheel phases. Both front-first orders are represented (32 each), followed by both rear legs. Commands are external. Each selected leg receives 20 seconds, with 2 seconds stand before/between/after: 90 simulated seconds total. No autoreset occurs in audits; terminated robots are frozen and subsequent samples excluded.

The historical final acceptance thresholds were checked directly: all four complete, final calibrated error below 0.1 rad, speed below 0.25 rad/s, and no termination. Held error is measured against persistent calibrated targets on completed wheels only, matching the historical metric. Nonactive snapshot-hold error is also reported separately: p95 0.0464 rad for the default randomized audit. The 0.035 rad / 0.08 rad/s / 0.5 s internal verification is stricter than final acceptance.

The historical favorite's saved unseen-64 results were read from W&B. These comparisons do not claim identical timing, random draws, or termination thresholds: the new slow-slew sequence lasts 90 seconds and the new thresholds are documented explicitly. This is one training seed and one 64-robot audit seed (20260908), not a multi-seed robustness claim.

## Failure evidence

At the final checkpoint, completed legs were FR 59/64, FL 48/64, BR 55/64 and BL 29/64. Thus both fronts work frequently; back-left dominates the full-sequence shortfall.

Back-left reached rotation on 50 robots but verification on only 29. Among the 33 unfinished back-left windows that remained alive, 13 ended in lift and 20 in rotate. Their wheels were physically at least 5 mm clear for about **98.8%** of samples, yet the encoder guard rejected abduction deviations above 0.25 rad for **62.7%** of samples on average. Hip and tilt rejection were only about 1.8% and 1.5%. These conditions overlap; the angular-speed diagnostic uses an estimate from recorded quaternions, not an exact controller IMU replay.

This identifies a specific interface mismatch to investigate next: the learned, physically clear pose often lies outside the guard I introduced. My recommended next step is to validate a less restrictive encoder/IMU readiness criterion against actual clearance across the randomized model distribution, keeping the eight-position-joint plus wheel-PD split. That change has **not** been made or tested. Further tuning awaits the user's requested first-attempt check-in.

## Actuation and deployment contract

Verified model: eight bounded position joints, four unlimited velocity-actuated wheel joints. Wheel actuator gain/bias remains kv=0.35 / position bias=0 / velocity bias=-0.35; position rows use kp=5, damping=0.25. The user's existing model-comment correction is preserved: the motors support position and velocity modes; mode is behavior configuration.

Calibration is persistent `wrap(home_i + pi)`, independent of accumulated driving phase and unchanged on reselection or interruption. All wheels use `2*wrap(reference-angle) - 0.35*velocity`; nonactive wheels actively hold locked encoder snapshots, and the active reference slews only while the lift guard permits. An isolated MuJoCo reaction test measured peak torque 0.00185 Nm at 0.25 rad/s versus 0.00926 Nm at 1 rad/s, with a half-turn settling in 13.64 seconds at the chosen rate. The paired frozen-policy speed audit above adds a free-robot balance check.

The deployment convention is one-hot commands and **eight direct joint-position outputs** (standard action scale/offset), with IMU, encoder positions/velocities, previous action and target-error sin/cos observations. All four wheel channels remain classical. A small self-contained wheel helper owns PD, target/snapshot storage, slew and phase/settling bookkeeping; the leg positions remain direct policy outputs. The helper's encoder/IMU gates must match deployment. No simulation contact, body position, or privileged critic input enters the actor/controller. The selected hip's training rate clamp must be mirrored or its raw-output parity validated. C++ integration, policy export and hardware testing were not performed in this first attempt.

## Artifacts

- [Corrected W&B run](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/gawg2max)
- [Default randomized audit](attempt1_corrected/audit64.json)
- [Nominal audit](attempt1_corrected/nominal64.json)
- [Frozen-policy speed check](attempt1_corrected/slew05_audit64.json)
- [Failure detail](attempt1_corrected/back_left_failure_detail.json)
- [Training curve](attempt1_corrected/training_progress.png)
- [Successful 90-second sequence video](attempt1_corrected/success_robot0.mp4) — a selected success, not a representative success-rate estimate; recorded randomized motion, nominal visual geometry.
- Checkpoint: `attempt1_corrected/mjx_params` (saved locally, excluded from Git).
- [Implementation/deployment layout and reproduction](../HYBRID_ALIGN.md)

## W&B cleanup completed

All **38 approved original runs were deleted**, and their absence was verified against a fresh project listing. Preserved: favorite `olqtuca3`, parent `1wpnxywj`, milestone `9xcraanb`, invalid hybrid diagnostic `486wowtn`, and corrected hybrid `gawg2max`. No additional runs were deleted. The original inventory, exact deletion log, and post-cleanup listing are retained in this directory.
