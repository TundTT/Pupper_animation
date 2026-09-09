# QuadMorph Phase 1 — wider clearance guard and retraining

The approved change raised the abduction readiness bound from **0.25 to 0.35 rad**. The retrained policy achieved **50/64 full-sequence successes** on the original audit and **49/64** on a fresh audit. The success rate exceeds the 45/64 comparison target, but drift and survival still need work: the original audit's p95 drift increased from 4.60 to 6.51 cm, and neither new audit had 100% survival. All terminations were excessive drift; neither audit recorded a fall or unsafe rotation.

| Configuration | Success | Survival | Falls | Drift p95 | Calibrated held-error p95 |
|---|---:|---:|---:|---:|---:|
| Historical named favorite | 45/64 | 64/64 | 0 | 4.80 cm | 0.0248 rad |
| First hybrid, 0.25 rad guard | 21/64 | 62/64 | 0 | 4.60 cm | 0.0410 rad |
| First checkpoint, only guard widened to 0.35 | 48/64 | 61/64 | 0 | 5.05 cm | 0.0432 rad |
| **Retrained, original audit seed 20260908** | **50/64 (78.1%)** | **60/64** | **0** | **6.51 cm** | **0.0396 rad** |
| **Retrained, fresh audit seed 20260909** | **49/64 (76.6%)** | **61/64** | **0** | **3.87 cm** | **0.0369 rad** |

The frozen-checkpoint check isolates the guard's effect: most of the success improvement was available without changing the learned weights. Retraining added two successes on the original set, with worse drift on that set. Both checkpoints and the guard-only comparison are retained; the retrained policy is not claimed to dominate every metric. No further training, threshold sweep, reward changes, or slew changes followed these audits.

## Guard selection and physical margin

The runtime guard still uses only encoders and IMU. The sole behavioral change is its abduction-deviation threshold. Hip >1.1 rad, tilt <0.12 rad, angular speed <0.3 rad/s, initial readiness dwell, settling criteria, interruption handling, and the **0.25 rad/s wheel-target slew** are unchanged. The physical clearance acceptance margin remains **5 mm**.

Recorded poses from the first attempt were examined at abduction bounds 0.25, 0.30, 0.35, 0.40, 0.45 and 0.50 rad. A 0.35 bound admits **76,840** previously blocked samples; their lowest recorded clearance was **24.4 mm**. Increasing to 0.40 admits only four more samples. This supports 0.35 as a modest relaxation rather than removing the constraint.

Every newly admitted sample was then checked offline with plain MuJoCo geometry, covering all four legs, with:

- torso lowered by a full 10 mm;
- wheel radii increased by the maximum modeled 2.5 mm;
- joint-angle perturbations up to ±5 mrad;
- roll/pitch perturbations up to ±10 mrad per axis.

There were **zero violations of the 5 mm margin**; minimum stressed clearance was **13.7 mm**. This is empirical coverage of recorded poses with sampled perturbations, not an exhaustive proof for every possible robot configuration. Geometry remains offline validation/audit data and is not added to policy or controller observations.

The subsequent live check of the old policy with the new guard recorded no unsafe rotation and **31.0 mm** minimum rotating clearance. After retraining:

| Audit | Rotating samples | Minimum rotating clearance | Unsafe rotations | Successful sequences passing physical touchdown and stricter 0.06 rad final-error checks |
|---|---:|---:|---:|---:|
| Original seed | 98,828 | **22.2 mm** | **0** | **50/50** |
| Fresh seed | 109,585 | **26.3 mm** | **0** | **49/49** |

Clearance is the lower surface of the actual tilted collision cylinder, accounting for randomized wheel radius and half-width, sampled at the 50 Hz control rate. These measurements preserve the same physical audit requirement as the first attempt; they are simulation validation, not hardware certification.

## Training and unchanged contract

One new PPO run was completed from random parameters: **50,790,400 actual steps**, 4,096 parallel environments, seed 0. All PPO settings match the first corrected run exactly. The configured 50M budget rounds upward to complete batches. Training plus compilation/evaluations took **233 seconds** on one GPU. No old checkpoint was loaded for training, and no additional training run was launched.

The policy still has **eight direct joint-position outputs** for abduction and hip. All four wheel channels remain classical PD: `2*wrap(reference-angle) - 0.35*velocity`, with persistent `wrap(home_i + pi)` targets, nonactive locked snapshots, and slow reference slew. The one-hot command, 51 hardware-realistic observations, action/latency handling, soft stance behavior, 35 mm lift/lower and 20 mm idle drift allowances, reward, model, physics randomization, and actuator gains are unchanged. There is no new wheel-reaching reward or simulation-only input.

The parameterized guard is recorded in each new run's config. Evaluation reads that value automatically; the first attempt's older config retains its original 0.25 rad guard unless explicitly overridden. The deployment helper must use the same guard value. Robot C++ integration, policy export, and hardware deployment were not performed in this attempt.

## Audit details and remaining failures

Both audits use 64 randomized robots and starting wheel phases, both front-first orders, and all four legs. Each external leg command gets 20 seconds, with 2-second stand windows before, between and after: **90 seconds** total. There is no autoreset in the audit. Success, survival, drift, and held-angle definitions match the first hybrid audit. Historical-reference timing and termination settings are not asserted to be identical.

Original-seed per-leg completions:

| Leg | First attempt | Retrained |
|---|---:|---:|
| Front right | 59/64 | **61/64** |
| Front left | 48/64 | **58/64** |
| Back right | 55/64 | **59/64** |
| Back left | 29/64 | **57/64** |

The previous back-left bottleneck is substantially reduced. The four original-seed terminations and three fresh-seed terminations all crossed the unchanged **15 cm body-drift limit** without triggering the fall condition. Their exact robot indices, times, and drift values are in the audit detail files. The original-set p95 drift regression is material and is not hidden by the better fresh-set number. Held-angle accuracy also remains above the approximately 0.025 rad historical reference.

The initial guard tuning used poses from the original audit distribution. The fresh 64-scenario audit was therefore added as a separate check; it was not used to choose another threshold or train again. This is one training seed and two audit seeds.

## Checks and artifacts

The expanded smoke test passes: 8 actions / 51 observations; wider abduction acceptance on every leg; retained hip, tilt and angular-speed rejections; persistent calibration and wheel-hold behavior; mixed actuator randomization; and repeated full task-state reset with preserved PPO timeout flags. Python compilation and whitespace checks pass. SHA-256 source snapshots match the code and model used by the run; the environment diff records the guard-only change.

- [W&B run](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/9bhxnkmw)
- [Original-seed audit](attempt2_guard/audit64.json)
- [Fresh-seed audit](attempt2_guard/fresh64.json)
- [Guard-only frozen-checkpoint audit](attempt2_guard/pretrain_guard_audit64.json)
- [Guard selection and stress test](attempt2_guard/guard_validation.json)
- [Training curves](attempt2_guard/training_progress.png)
- [Successful 90-second sequence video](attempt2_guard/robot0_sequence.mp4) — selected successful robot 0, recorded randomized-physics motion rendered with nominal visual geometry.
- [Failure details](attempt2_guard/audit64_details.json), [fresh-seed failure details](attempt2_guard/fresh64_details.json)
- [Environment change](attempt2_guard/guard_change.diff)
- Checkpoint: `attempt2_guard/mjx_params`; source/config snapshots are beside it.

Work stopped after this approved attempt. The next decision is whether to polish drift outliers and held-angle accuracy or keep the current result. No W&B runs were deleted during this task.
