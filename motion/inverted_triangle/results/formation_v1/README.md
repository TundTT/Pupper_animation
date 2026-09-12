# Formation screening, September 12, 2026

**Not robust yet: 3/7 development cases pass the current screen.** These are
simulations of the new simultaneous roll; no hardware or actual walking-policy
handoff. User input: rigid legs with about 10 mm total length spread; slight
angle variation. +/-5 degrees is our trial assumption, not a measurement.

[Seven actual rollout videos and audit artifacts in W&B](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/ce3662820cae47f8).

| Development case | Screen | Max tilt (deg) | Min sampled CAD gap (mm) | Min sampled front/rear gap (mm) | Failed gates |
| --- | --- | ---: | ---: | ---: | --- |
| nominal_regression | PASS | 0.864 | 2.789 | 39.866 | none |
| front_short_rear_long | FAIL | 4.714 | 1.184 | 34.010 | tip support |
| right_short_left_long | PASS | 2.489 | 1.177 | 37.755 | none |
| diagonal_length_mismatch | FAIL | 4.825 | 1.177 | 20.438 | walk pose, tip support |
| one_front_leg_10mm_short | FAIL | 1.021 | 2.789 | 28.163 | walk pose, tip support |
| angle_only_assumed_5deg | PASS | 0.950 | 2.331 | 39.126 | none |
| combined_length_and_angle | FAIL | 4.658 | 1.304 | 20.188 | walk pose, tip support, settled |

PASS means only the fixed probe gates passed. CAD sampling is every 0.1 s,
so these numbers are not continuous clearance guarantees. All seven completed
19 s, with zero unintended motor/body floor force and no sampled shin CAD
intersections. The four failed cases must remain failures; an attractive video
or nearly level body is insufficient. These cases were used for development,
not held-out validation. Friction, sensor errors, hardware dynamics and walking
handoff were not varied/validated in this batch.

The one-front-leg-10-mm-short case ends at 0.193 degrees tilt while its short
front tip is 10.31 mm above the floor and carries zero normal force. The diagonal
case ends with a tip about 20 mm above the floor after the body tilts. Shape
variation can create larger ground gaps than the original geometric difference.

## Saved evidence

- `baseline_summary.json`: full reports and per-variant source/geometry hashes.
- `baseline_provenance.json`: baseline code commit, environment and all source hashes.
- `baseline_states_and_traces.zip`: untouched audits, sampled states, traces,
  exact final integration states, configurations and W&B identity.
- `cloud_verified.json`: cloud video byte hashes and artifact states, once checked.
- `local_diagnostics.zip`: all 28 earlier local diagnostic trials, including
  all three rejected settling variants. Each trial's provenance source hash maps
  to exact bytes under `source_sha256/`. No trial is discarded. The archive
  includes the baseline and 8/12 s settling holds, not four equivalent final tests.
- `diagnostic_summary.json`: readable summaries of those 28 trials. Their CAD
  fields are null because these were short local diagnostics without CAD/video.

The online batch ran source commit `8bcab6a`. A later change adds generated STL
hashes to snapshot compatibility checks; it does not change geometry, physics
or commands. Historical final states remain evidence for their original source
version. For continuation with current code, replay from the beginning to obtain
its stricter snapshot format; do not edit old snapshot hashes to force acceptance.

All trials preserve the nominal XML and assets, 9 mm gap, original attachment
and backpack model. Synthetic distal meshes retain nominal mass/inertia.
See `../../FORMATION_ROBUSTNESS.md` for the model's limits and hardware sources.

## Checks and next work

23 selected tests passed: 5 new formation/continuation tests, 3 roll-coordinate
checks, and 15 existing model/continuation checks. The one historical sequential
candidate-file test is excluded because its archived candidate is outside this
sparse checkout; it is not a test of the new roll or geometry mechanism.

Keep the clean coordinated roll as a prior and develop bounded per-leg feedback
across uneven geometry. The three elementary settling trials do not solve the
suite. A small learned residual is an available next option on the PC; no policy
was trained here. Freeze candidate and gates before fresh held-out testing,
then validate continuous entry into the actual walking policy on the same
imperfect legs. Use `../../ROLL_TO_STAND_PC_HANDOFF.md` as the PC task.
