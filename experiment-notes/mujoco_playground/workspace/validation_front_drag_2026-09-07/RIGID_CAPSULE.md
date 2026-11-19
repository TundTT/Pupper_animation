# Flush rigid-capsule update

The user's latest request supersedes selection of the previous soft-contact
front-drag candidate: remove ring compression, stiffen capsule contacts, and
extend the tip to the ring's outer bottom edge.

## Implemented model

- Overall capsule length **60 -> 62.649961 mm**. The distal cap center moves
  **2.649961 mm**, from foot-local |Y|=48 to 50.649961 mm. Radius stays 12 mm;
  the proximal cap center stays at |Y|=12 mm. All four legs match.
- The tip's axial extent is calibrated from the mesh-derived outer ring outline.
  This is tip-axis alignment, not an attempt to cover the ring sides with a
  larger collision shape. The ring visual geometry and mass/inertia stay fixed.
- Capsules remain rigid MuJoCo geoms. Both floor and feet now use solref
  `[.008,1]` and solimp `[.99,.99,.001,.5,2]`. No contact softness randomization.
  Contact priority is unchanged so high floor friction still reaches actual contacts.
- `ring_bottom` contributes **zero reward**. Its old diagnostic is retained for
  archived comparisons. Side depth (-3), side overlap frequency (-6), side rubbing
  (-2), and contact-only tip orientation (-.5) remain active.
- Native settling recomputes the nominal reset/height reference for the new
  geometry, approximately **146.72 mm**. Per-slot height adjustment still uses
  randomized neutral geometry projected onto the true floor normal.

## Length randomization around the new geometry

The capsule and ring move as one assembly under upstream leg-length and mount
randomization. The new tip extension exists in every slot; randomization cannot
reintroduce a gap by shortening the capsule independently of its ring.

The engineering prior is retained: common U(.90,.98) times per-leg U(.96,1.04),
combined .864--1.0192 of the 84.455 mm upstream offset. Both ranges are now explicit
training configuration/CLI settings. These are **not** multipliers on the capsule
alone or on the entire hip-to-floor distance. Its new nominal geometry shifts
the resulting total support reach. Shortest/longest four-second standing heights
at floor friction 1 are now **135.38 / 148.33 mm**, versus the older soft model's
131.01 / 144.08 mm. Both longer geometry and reduced sinking contribute.

## Verification

- Native 4-second standing checks: scales .864/.94/1/1.0192 × friction 1/2/3.
  All 12 cases have four floor contacts and zero ring-side cost.
- Largest capsule penetration: **0.0573 mm**. Numerical contact is stiff, not an
  exact zero-penetration constraint. Mild ring-bottom outline overlap remains
  possible from the settled foot tilt; it no longer carries a compression cost.
- Tests check exact axial flush fit, unchanged radius/proximal end/ring geometry,
  actual contact friction and stiffness, custom length-range plumbing, and no
  softening or capsule/ring misalignment under domain randomization.
- [Standing measurements](rigid_flush_standing.json),
  [geometry overlay](rigid_flush_geometry.png), and
  [direct MuJoCo viewer model](rigid_flush_model.xml).

`walk_geometry.load_walk_model` compiles these overrides before MJX conversion,
so collision bounds are rebuilt. Source XML remains available for old-run
reproduction. New runs save `effective_model.xml`, effective contact/geometry
metadata, and source snapshots. Evaluation uses each checkpoint's saved contact
mode; `--foot_model rigid_flush` explicitly requests a transfer test.

The earlier ring6 policy is a warm-start candidate, not a hardware-ready policy
for this changed model. No new export has been placed in robot-code.

## Pre-training transfer test

The ring6 43,253,760-step checkpoint was evaluated explicitly under `rigid_flush`,
floor friction 2, six commands (forward .15/.20/.35, reverse .20, both spins .5),
three reset seeds and six seconds each, with the first second discarded. No pushes,
sensor noise or latency. All **18/18** survived and all six commands measured zero
ring-side overlap. At forward .20, front major-swing median peaks were **6.89/5.47 mm**
and XY tracking error was **.040 m/s**. These are transfer results before retraining,
not a hardware claim or a matched-duration comparison with the older soft trials.

See [the per-command report](rigid_transfer_mu2/report.json).

## New training run

[W&B run wtzmn4vi](https://wandb.ai/QuadMorph/pupper-leg/runs/wtzmn4vi),
`output/front_drag_trials/rigid_flush/walk_2026-09-07_15-59-18`.
Warm-started ring6 43,253,760; requested 86.4M steps, 8,192 environments,
learning rate 1e-4, five evaluations, GPU 1. Training friction .8--2.5,
showcase friction 2, all evaluation videos enabled at 50 fps.

The saved `run.json` confirms `foot_model=rigid_flush`, ring-bottom weight zero,
configured length ranges, and the effective capsule geometry/contact settings.
[Launch command](rigid_flush_launch.json) and [training log](rigid_flush_train.log).

Validation completed: **35 full-suite tests passed** (520.63 seconds), plus the
new effective-model snapshot/reload test passed separately: **36 tests total**.
`git diff --check` and Python compilation passed. Initial PPO evaluation: reward
82.158, full episode length 600, mean tilt 2.31 degrees, zero ring-side overlap,
velocity error .054 m/s. This is the warm-start evaluation at step zero; it does
not yet demonstrate an improvement from retraining.
