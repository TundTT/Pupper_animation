> Latest update: the user requested a longer, rigid capsule and removal of ring compression. The soft-contact trials below are historical; their candidate selection is superseded. See [the rigid-capsule implementation and checks](RIGID_CAPSULE.md).

# Front-foot dragging: hardware feedback and friction stress test

The higher-friction test exposes a weakness in the deployed checkpoint, but does
not reproduce every detail of the hardware report. Hardware forward walking drags
more than reverse; extreme simulated friction also worsens reverse substantially.
Friction is therefore a useful training/test dimension, not a proven complete
explanation of the simulation-to-hardware difference.

## Deployed checkpoint under stress

Checkpoint: `params_000129761280` from `walk_2026-09-07_04-04-36`, SHA256
`d240a2cd345a3e00eec49fdd2ac8b9697aabb191b6383e68271099edd25acd1f`.
This is the policy ported in robot-code commit `b455cbf`.

All entries below command forward 0.20 m/s. Three seeds, eight seconds per rollout,
first second discarded; nominal geometry, no pushes/noise/latency. Contact and
clearance are sampled at 250 Hz, policy actions at 50 Hz.

| Floor sliding friction | FR / FL major swing peak | FR / FL contact sliding RMS | All-foot ring-side overlap | XY velocity error |
|---|---|---|---|---|
| 1 (original XML) | 2.68 / 2.98 mm | .039 / .047 m/s | 0% | .021 m/s |
| 2 | 2.24 / 3.34 mm | .074 / .074 m/s | 3.52% | .032 m/s |
| 3 | 2.48 / 3.71 mm | .136 / .115 m/s | 22.35% | .038 m/s |

A major swing lasts over40ms and exceeds2mm. These medians exclude the smallest
contact flickers; all-event counts and heights remain in the raw reports. The
policy survives all18 rollouts at each friction level, demonstrating that survival
and aggregate tracking alone miss the low-clearance/contact-quality failure.

[Stress plot](friction_stress.png), [forward video at friction2](deployed_forward_mu2.mp4),
[frames](deployed_forward_mu2_frames.jpg). The video was decoded and checked:400
frames at50fps. Raw reports, protocols, and NPZ traces are in [nominal](nominal),
[mu2](mu2), and [mu3](mu3).

Native MuJoCo contact inspection confirms actual capsule-floor friction values
1/2/3 after the floor override. Increasing friction is not a hard no-slip constraint:
contact remains compliant, policy motion adapts to a different state trajectory,
and sliding can increase for the same unadapted policy. No claim is made that2 or3
matches the actual hardware surface, which was not specified.

## Training trial

A friction-only warm start is running from that exact deployed checkpoint, with
unchanged action bounds, home/gains, and gait/ring reward weights. Sliding-friction
DR changes from uniform[0.4,1.2] to uniform[1.2,2.5]. Nominal showcase floor friction
is2. Other DR draws remain identical for matching random seeds. This isolates the
friction change from a reward redesign, although PPO remains stochastic.

Run: `output/front_drag_trials/friction_only/walk_2026-09-07_14-56-35`,
[W&B lalqv72h](https://wandb.ai/QuadMorph/pupper-leg/runs/lalqv72h).
Requested86.4M steps, learning rate1e-4, five evaluations, seed0. Intermediate,
final, and best videos are enabled. The initial128-environment randomized evaluation
had reward61.415, full600-step length, and14.0% ring-side overlap.

Selection will use individual front-foot lift, contact sliding, ring overlap,
forward tracking, and reverse/turn regressions at both nominal and high friction.
Training reward improvement alone will not establish a hardware fix.

## Code and checks

- Floor friction is configurable in the environment, evaluator, renderer, and
  trainer. The DR range is separately configurable so randomization cannot silently
  erase the intended training change. XML and robot-code deployment remain intact.
- Added per-foot250Hz tangential contact velocity and low-clearance-motion metrics.
  Contact sliding distance integrates material-point tangent speed during capsule
  contact. Motion faster than.05m/s below2mm is a scuff proxy, not a TPU load sensor.
- New friction tests:2passed. Diagnostic/plain stepping and randomized Brax reset/
  step integration:2passed. Logs are in this directory.

## Contact-model caveat discovered during review

Higher friction also changes static penetration with this model's pyramidal
contact cone. A native MuJoCo four-second neutral standing check gives torso
heights142.464/139.960/138.486mm at friction1/2/3, and front capsule bottoms about
-1.53/-4.03/-5.51mm. Raising solver iterations from8 to32 does not change those
results. Thus part of the ring-overlap increase is a change in effective contact
compliance, not evidence that the policy started relying on TPU sides solely
because grip increased. The sweep isolates one model parameter, not a pure change
in traction with identical effective normal stiffness.

As a separate diagnostic, an elliptic contact cone holds the standing height and
penetration constant across friction1/2/3, but at a different baseline height
(139.960mm). This is not used in training or deployment here: changing cone/contact
calibration would itself require a separate matched validation. See
[standing checks](friction_standing_check.json) and
[elliptic diagnostic](elliptic_standing_check.json). The actual hardware surface and
contact compliance remain unmeasured.

## Completed friction-only trial and next experiment

The first run completed86,507,520 steps; training-reward best was64,880,640. Seven
videos were uploaded to W&B. The64M candidate improves nominal forward.20 front
major swing peaks from2.68/2.98 to4.34/5.56mm, and high-friction2 peaks from2.24/3.34
to4.03/3.61mm. But front-left low-clearance motion on friction2 persists, and nominal
forward.35 velocity error rises from.036 to.069m/s. This is not selected as a full
hardware replacement. [Per-foot traces](friction_only_comparison.png); raw reports
are in `friction_final_mu1` and `friction_final_mu2`.

A second trial uses friction[.8,2.5] to include the original floor, swing reference
8mm (previous6mm), planned-swing weight-3 (previous-1), and linear tracking weight6
(previous4). It warm-starts from friction-only64M. All joint scales, ring weights,
height reference/weight, torque, action smoothing, and DR geometry stay unchanged.
An independent constrained kinematics check finds8mm and10mm arcs over a56mm
fore/aft stroke feasible under the current [.5,.25,1.1] bounds, with maximum XZ
residual.052mm at the shortest common length endpoint and negligible nominal error.
This checks geometry, not dynamic feasibility across all combined DR.

The stronger clearance experiment is a combined follow-up; it cannot assign an
isolated causal effect to each of those reward/range adjustments.

### Ring-aware feasibility qualification

The initial fixed-X/Y56mm-stroke IK check only certified capsule reachability.
Checking the ring on that particular path reveals side intersections, so that
path must not be treated as a collision-free reference. The implemented reward
prescribes height only and leaves horizontal placement free. A second constrained
check, allowing fore/aft placement and requiring1mm ring-side clearance, finds
an8mm vertical arc under the same joint bounds: maximum height residual below
.00021mm, lateral adjustment below.031mm, and fore/aft adjustment up to50.4mm over
the checked common-length endpoints. This demonstrates geometric options, not
that the policy follows those exact paths or that every combined-DR model can.
See `ring_clearance_feasibility.json` and `ring_safe_vertical_feasibility.json`.
