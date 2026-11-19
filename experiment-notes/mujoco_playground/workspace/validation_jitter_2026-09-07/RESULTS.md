# Reverse and spin jitter: selected policy, September 7, 2026

The new policy substantially reduces reverse cadence and abrupt motion, improves
both spin directions, and nearly eliminates modeled ring-side overlap in the
matched nominal tests. Forward velocity tracking is slightly worse. This is a
simulation result, with remaining limitations below.

## Selected artifacts

- Run: [W&B 61am09cq](https://wandb.ai/QuadMorph/pupper-leg/runs/61am09cq),
  panel **`eval/video_selected`**.
- Checkpoint: [selected_params](../output/jitter_trials/tip_credit3/walk_2026-09-07_04-04-36/selected_params),
  an exact copy of `params_000129761280` (129,761,280 steps).
- SHA256: `d240a2cd345a3e00eec49fdd2ac8b9697aabb191b6383e68271099edd25acd1f`.
- [32-second, 50 fps video](../output/jitter_trials/tip_credit3/walk_2026-09-07_04-04-36/rollout_selected.mp4).
- [Exported policy](../output/jitter_trials/tip_credit3/walk_2026-09-07_04-04-36/policy_walk_selected.json).
- [Joint-target and yaw traces](jitter_comparison.png), [sampled video frames](selected_frames.jpg).
- [Selection record](selection.json), [raw report](credit_final/report.json),
  [protocol](credit_final/protocol.json), [W&B verification](wandb_selected_verified.json).

The final training run completed 173,015,040 steps. Its highest training-reward
checkpoint remains correctly named `best_params`. The selected 129M checkpoint
had better reverse tracking and lower nominal ring-side overlap in the separate
10-second comparison than the final 173M checkpoint. Use the selected artifacts
above for this result. Selection used these evaluation commands; they are not an
independent held-out validation set.

## What was causing the behavior, and what changed

The earlier policy's reverse motion used about 5–6 Hz steps and rapid target
reversals. Stronger action regularization alone reduced target changes but did
not resolve the high cadence. A swing curve under the old action bounds slowed
the gait at the expense of reverse tracking and sometimes turning. The centered
swing workspace was restrictive; rewarding clearance did not make a longer
path physically reachable.

The selected configuration combines these changes:

1. Penalize the second action difference at weight -0.8, retaining first
   difference -0.30. Both use fixed physical normalization so widening the action
   range cannot make identical joint-target motion appear smoother.
2. Increase knee command scale from 0.5 to 1.1 rad while keeping hip/abduction
   scales 0.5/0.25 and the XML joint limits. Lower the moving torso reference by
   3 mm. A geometry check found a 6 mm apex and 56 mm fore/aft stroke reachable
   with this workspace (including the shortest common length endpoint), with
   under 0.054 mm maximum path residual. This is a constrained kinematics check,
   not an exhaustive combined-DR feasibility guarantee.
3. Replace the old height-area reward with a 6 mm `sin²` curve over 160 ms,
   triggered independently by each foot's liftoff. The remaining curve penalty
   persists through early contact. A small 0.25 credit rewards active compliant
   swings; hovering at floor level over the whole curve still loses reward.
   No external phase input or runtime action filter was added.
4. Increase linear/yaw tracking weights to 4/3 and height weight to -12, so the
   policy follows commands and maintains the intended moving height.
5. Penalize the fraction of feet whose ring sides intersect the floor at weight
   -3, in addition to side depth (-3) and rubbing (-2). This addresses frequent,
   shallow brushing that the squared depth cost tolerated. Bottom allowance
   remains 9 mm, bottom weight -1, and contact tip-lean weight -0.5.

The capsule remains the physical foot. Ring penalties use a transformed,
undeformed outline: they encourage tip support and discourage side contact but
do not simulate TPU deformation or prove how hardware load is shared. The new
frequency cost counts any side probe below the floor, averages over feet and
250 Hz substeps, and then receives the normal outer control-step `dt` multiplier.

The actor still has 144 inputs and 12 outputs. **Use the exported action scales
`(0.5, 0.25, 1.1)` repeated for four legs.** Old runtime scales would execute a
different policy. Geometry DR, parent-frame mount perturbations, capsule/ring
geometry, torque weight, and hardware gains were not changed in this follow-up.

## Matched measurements

Baseline: the previous selected `tip_clearance` checkpoint at 57,671,680 steps.
Candidate: `credit3_129` in the report. Each received 13 fixed commands, three
reset seeds, 10 seconds per command, discarding the first second. Both survived
39/39 rollouts. Pushes, sensor noise, latency, and joint-reset noise were disabled;
the model was nominal. Action evaluation used each checkpoint's saved scales.
Physical foot and motion traces were sampled at 250 Hz; policy commands at 50 Hz.

| Reverse command | Old → selected cadence | Old → selected major airborne duration | XY velocity error, old → selected |
|---|---:|---:|---:|
| -0.15 m/s | 6.25 → 1.95 Hz | 60 → 196 ms | 0.0190 → 0.0167 m/s |
| -0.20 m/s | 6.10 → 2.31 Hz | 72 → 132 ms | 0.0228 → 0.0207 m/s |
| -0.35 m/s | 5.32 → 3.01 Hz | 72 → 152 ms | 0.0414 → 0.0412 m/s |

“Major” swings last over 40 ms and exceed 2 mm height; cadence uses successive
same-foot major liftoffs. This excludes tiny contact flickers. The raw report also
preserves all-event statistics. At -0.20 m/s, torso angular acceleration fell
48%, joint acceleration 34%, and median touchdown speed 58% (0.183 → 0.077 m/s).
At -0.35 m/s, torso angular acceleration fell 39% and touchdown speed 53%.

| Spin command | Mean yaw rate, old → selected | Yaw error, old → selected | Yaw-rate standard deviation |
|---|---:|---:|---:|
| +0.5 rad/s | +0.437 → +0.474 rad/s | 0.0959 → 0.0588 rad/s | 0.1127 → 0.0946 rad/s |
| -0.5 rad/s | -0.362 → -0.482 rad/s | 0.1472 → 0.0542 rad/s | 0.1839 → 0.0649 rad/s |

Right-spin yaw error fell 63%, its rate variation 65%, and torso angular
acceleration 45%. Left-spin yaw error fell 39%, with a smaller reduction in
physical angular acceleration (6%). The yaw traces still contain fluctuations;
the policy is not perfectly smooth.

Across all 13 commands, averaging each command equally:

| Metric | Baseline | Selected |
|---|---:|---:|
| XY velocity error | 0.02066 m/s | 0.02255 m/s |
| Yaw-rate error | 0.05765 rad/s | 0.04235 rad/s |
| Ring-side overlap fraction | 1.8524% | approximately 0.0023% |
| Action second-difference RMS, fixed reference units | 0.2245 | 0.1189 |
| Joint acceleration RMS | 96.61 rad/s² | 78.96 rad/s² |
| Torso angular acceleration RMS | 20.50 rad/s² | 13.19 rad/s² |

## Remaining tradeoffs and scope

Forward XY velocity error increased by 4.8, 5.9, and 7.6 mm/s at 0.15, 0.20,
and 0.35 m/s. Fast reverse median major lift fell from 5.26 to 3.53 mm despite
longer airborne intervals and softer touchdowns. Pure-spin net translation over
the measured nine seconds increased from 12 to 27 mm left and 41 to 49 mm right.
These are material reasons to keep measuring the whole gait rather than select
on training reward alone.

The selected checkpoint's randomized training evaluation reached the full
600-step episode length over 128 environments, reward 75.257. Its reported
ring-side fraction was about 1.4%, higher than in the clean nominal suite. The
near-zero nominal result is therefore not an all-geometry or hardware guarantee.
This turn evaluated eight training candidates, not multiple independent training
seeds of the selected recipe; individual changes cannot all be assigned isolated
causal effects. [Experiment history](PLAN.md) preserves the intermediate findings.

## Verification

- Full workspace suite: **30 passed**, 478.09 seconds; one existing JAXopt
  deprecation warning. [Test log](final_tests.log).
- Tests cover reward clock persistence, shaping credit versus floor hovering,
  action history/reset, instrumented stepping equivalence, geometry/DR, Brax
  wrapping, export parity, and warm-start action/history remapping.
- The selected video decodes to 1600 frames at 50 fps (32 seconds), including
  both spin directions, three reverse segments, lateral motion, and stops.
  Representative frames were inspected; the 250 Hz measurements establish the
  motion statistics. [Video metadata](video_verified.json).
- W&B selected media was downloaded and checked against the local video's
  SHA256, with the checkpoint step and SHA256 verified in the run summary.
- Export completed with the selected checkpoint and matching saved configuration.
