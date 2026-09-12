# PC simulation result — September 12, 2026

**The continuous four-flip plan passes nominally and in all 20 required friction/seed replays. Broader stress testing passes 7/16 cases; full uncertainty robustness is not achieved.**

The robot starts on four rigid point-up triangles and finishes in the audited four-tip stance. The original shin tip and additional 9 mm axial gap are unchanged. This was simulation only; no robot connection, deployment, calibration change or policy activation occurred.

[W&B review and videos](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/79b7226026be45d4) · [Nominal continuous rollout](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/5d30198204624884) · [All 114 verified experiment runs](wandb_runs.json)

## Accepted immutable plan

Order and hub directions: **back-right −π → back-left +π → front-right −π → front-left −π**. The actual integration state, velocities, actuator state and commanded references carry between stages; the saved state hashes match exactly at every boundary. See [continuation_chain.json](continuation_chain.json).

- Plan: [plan/plan.json](plan/plan.json), SHA-256 `ae132fcb1de919d548cbee61bfa36f09aeffd4f49dc0b424995b233a05d9295e`.
- Tested source commit: `7a923b7409300fa09a7033b2dcc7c6af2af596c1`; 14 model/continuation tests passed.
- Model SHA-256: `768cbeb0bfbaab0c898e6078d0b778ee2968718d98bdb2857e13538daeb607a8`. CAD/XML hashes are in [source_manifest.json](source_manifest.json).
- Nominal rollout: **122.327 simulated seconds / 63,610 physics steps** at 520 Hz; [complete actual video](continuous_rollout.mp4).
- Dependencies: [requirements.lock.txt](requirements.lock.txt); Python 3.12.3, MuJoCo 3.3.7, NumPy 2.2.6, SciPy 1.16.2. Full environment and source hashes are in [summary.json](summary.json).

| Flip | Rotation floor gap (mm) | Refined CAD gap (mm) | Minimum support (N) | Peak tilt (deg) | Peak requested torque (Nm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| back_r | 7.954 | 3.867 | 2.216 | 2.151 | 0.429 |
| back_l | 7.997 | 2.776 | 5.001 | 2.727 | 0.511 |
| front_r | 6.951 | 2.491 | 2.625 | 6.664 | 0.959 |
| front_l | 10.480 | 3.838 | 2.537 | 6.182 | 0.916 |

Maximum nominal landing descent is 14.825 mm/s against the 25 mm/s gate. Motor/body floor support is zero. All original torque, speed, tilt, support, tracking, landing and CAD gates remain enforced.

Final loads in front-right, front-left, rear-right, rear-left order are 9.714 N, 5.489 N, 6.251 N, 9.829 N. Final long-tip heights are 0.479 mm, -0.017 mm, 0.209 mm, -0.031 mm (the screening tolerance is ±3 mm).

## CAD and visual review

The limiting detailed pair is the front-right shin against the rear-right motor assembly, with **2.491 mm** sampled clearance. Each phase minimum and both same-side wheel minima were refined at the 1/520-second physics interval within ±0.125-second windows. Reintegrated states matched the saved audit exactly. The first three stages are unchanged from the previously reviewed prefix; [prefix equivalence](cad/prefix_equivalence.json) records exact equality.

| Flip | Right front/rear wheel minimum (mm) | Left front/rear wheel minimum (mm) | Three-view key frames |
| --- | ---: | ---: | --- |
| back_r | 21.804 | 50.373 | [Initial, lift, half-turn, final](cad/01-back_r/views.png) |
| back_l | 78.705 | 64.956 | [Initial, lift, half-turn, final](cad/02-back_l/views.png) |
| front_r | 18.585 | 100.585 | [Initial, lift, half-turn, final](cad/03-front_r/views.png) |
| front_l | 40.876 | 20.962 | [Initial, lift, half-turn, final](cad/04-front_l/views.png) |

Finite sampling is not continuous collision proof. Floor-only convex hulls and independent nonconvex CAD checks remain separate. Polymer compliance, spacer mass, measured friction and calibration uncertainty are not certified by these rigid-model results.

## Fixed-plan robustness

| Friction | Perturbation seeds | Passing continuous sequences |
| --- | --- | ---: |
| 0.50 | 1–5 | 5/5 |
| 0.65 | 1–5 | 5/5 |
| 0.80 | 1–5 | 5/5 |
| 1.00 | 1–5 | 5/5 |

Each initial seed perturbs all eight proximal coordinates by uniform ±0.01 rad. Every replay uses the same plan and propagates the actual preceding end state. No nominal checkpoint is restored between legs.

Additional sensitivity cases use mass/inertia ±10%, trunk COM offsets ±3 mm horizontally/±2 mm vertically, 5/10-step command delays (9.62/19.23 ms), 1.5 Nm available torque, PD gains ×0.8/×1.2, and initial height ±2 mm with roll/pitch ±1°. Five combined seeds use heavier mass, shifted COM, delay, lower gains/torque and a low tilted start. These are explicit sensitivity assumptions, not measured hardware distributions. The rationale and exact configuration are in [the grid configuration](batches/final-validation-v5/config.json).

**7/16 additional cases passed.** Both COM cases, both delay cases, half torque, and both start-offset cases passed. Remaining failures are:

| Condition | First failing leg/phase | Measured gate failure |
| --- | --- | --- |
| [combined-seed-22](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/ac0020bba7704ddf) | back_r / angle_hold | Minimum rotation clearance 4.909 mm; minimum 5 |
| [combined-seed-23](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/bf52c2fcd4c34477) | back_r / rotate | Minimum rotation clearance 4.892 mm; minimum 5 |
| [combined-seed-24](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/aa815c75e70a400b) | back_r / rotate | Minimum rotation clearance 4.874 mm; minimum 5 |
| [combined-seed-25](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/6a4dcd1cbbe747a1) | back_r / angle_hold | Minimum rotation clearance 4.913 mm; minimum 5 |
| [combined-seed-26](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/4d05f55d98f04e37) | back_r / rotate | Minimum rotation clearance 4.886 mm; minimum 5 |
| [gains-high](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/e16b47bf1dd243dc) | front_l / land | Peak near-ground descent 102.734 mm/s; limit <25 |
| [gains-low](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/b540387598f743dc) | back_l / lift | Peak motor/body floor force 4.947 N; limit <0.2 |
| [mass-high](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/9d8ed647b27642c4) | back_l / clearance_hold | Peak motor/body floor force 1.566 N; limit <0.2 |
| [mass-low](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/d287bb93e0164137) | front_l / land | Peak near-ground descent 114.082 mm/s; limit <25 |

Exact first sampled failure times and values are in [stress_failure_details.json](stress_failure_details.json). Landing failure times were derived from the retained actual traces where the original audit did not record that timestamp. A separate, unaccepted duration scout found that slowing the final landing to 12 seconds still exceeded the descent gate; that scout has no CAD/video acceptance claim.

[All 36 results with run links](validation_results.csv) · [Failed lower-gain continuous replay](failed_gains_rollout.mp4)

## Search history and preserved failures

The saved rear-right flip was reproduced on clean source commit `5fb4567`. The prescribed nine-parameter Powell sequence failed at the second front-left flip on support and CAD gates. A rear-first alternate order passed two connected flips. Body-shift and independent landing targets enabled the third and fourth flips.

A lower-cost lift revision then intersected the rear-right motor assembly; its independent audit rejected it. The final optimizer includes those motor/body pairs in its cost, and a saved regression check verifies the previously missed collision is now detected. The accepted plan retains the clear lift path and uses two focused landing searches across friction conditions.

Earlier plans failed high-friction tracking, including a final-leg error of 0.0350187 rad against the strict 0.035 gate. That failed audit and [its actual video](earlier_tracking_failure.mp4) are preserved. The corrected plan passed the same seed with 0.00950 rad final-leg error. No threshold was relaxed.

Every experiment audit, candidate, configuration and continuation state is retained under [evidence/](evidence/); earlier plans and exact local drivers are also included. Full traces, evaluation streams and all actual videos remain in the local isolated checkout and the verified W&B artifacts/Media indexed by [wandb_runs.json](wandb_runs.json). Video MD5 values were matched to local bytes and every indexed artifact was COMMITTED in the cloud. [evidence_sha256.json](evidence_sha256.json) hashes this review bundle.

## Leg-policy compatibility and remaining work

**Direct leg-policy entry is not validated.** The final proximal posture differs from `policy_walk_v2.json`, and the rear-left commanded hub coordinate is 7.283185 rad versus the policy reference of 1 rad. The physical hub orientation agrees modulo 2π, but silently resetting or clamping that reference would not be an acceptable handoff. See [policy_compatibility.json](policy_compatibility.json).

A reference-preserving stance transition and measured-state executor remain necessary before a policy/hardware handoff: encoder/IMU tracking, low-speed progression, bounded landing, timeout/abort behavior and valid calibration. Simulation contacts are audit signals, not physical foot-load sensors. No hardware-ready claim is made; the failed uncertainty cases remain unresolved.

## Replay

From the repository root in the fresh environment described in [VALIDATION.md](../../VALIDATION.md):

```sh
python -m motion.inverted_triangle.sequence --plan motion/inverted_triangle/results/pc_20260912/plan/plan.json --output runs/inverted_triangle/review-replay
python -m motion.inverted_triangle.robustness --plan motion/inverted_triangle/results/pc_20260912/plan/plan.json --output runs/inverted_triangle/review-grid --workers 12
```

Use a new output directory each time. On headless Linux prefix the command with `MUJOCO_GL=egl`. These commands log online to the documented QuadMorph W&B project.
