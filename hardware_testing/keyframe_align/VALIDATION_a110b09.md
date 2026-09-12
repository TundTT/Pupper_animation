# Keyframe acceptance — a110b09, September 11, 2026

**22/22 simulation scenarios passed. No training or physical motion was run.**
Frozen source: `a110b09dc6b6da5ae577ef021277bae97b08a060`. Every final audit reports
clean source and unchanged source/configuration/library hashes during execution.

[W&B run, Media rollout and audit artifact](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/e276b6aaa0a74a5b)
contains the actual nominal video and all 22 final audits. The rejected earlier
matrix remains attached under `failed-development-matrix` and is labeled
`development_audit`; its failures are not hidden or counted as final passes.
The remote Media file and summary were verified through the W&B API.

## Acceptance scope

- Nominal all-wheel sequence.
- Eight IMU roll/pitch bias combinations in {-0.04, 0, +0.04} radians.
- Four random starting-angle seeds with synthetic 0.06 Nm hub friction.
- Four more seeds combining that friction with all four bias corners.
- Cancellation in SHIFT, LIFT, ROTATE, LOWER and RECENTER.
- Nine unit tests, including large touchdown disturbance, retry latches, stop,
  calibration preservation, and 600 independent MuJoCo geometry comparisons.
- The standalone C++ full-sequence/stop test passed natively and as a statically
  linked ARM64 executable under QEMU. This is not a Pi realtime performance test.

| Measured across final simulations | Result |
| --- | --- |
| Largest final angle error after the complete non-cancelled sequence | 0.47° |
| Smallest actual modeled floor clearance during rotation | 19.73 mm |
| Smallest conservative wheel-to-wheel spacing | 16.59 mm |
| Smallest conservative wheel-to-body spacing | 18.99 mm |
| Largest descent speed within 8 mm of ground | 24.34 mm/s |
| Nominal duration per wheel, including rotation and final verification | 20.43–20.68 s |

The pass predicate now requires final-angle error below 0.035 rad (~2°), angular
settling, all-wheel target retention, descent speed below 30 mm/s, positive
collision margins, actual rotation clearance above 5 mm, no timeout/failure, and
return to hold. Cancellation audits require recovery rather than alignment to a
target the operator cancelled. These are explicit engineering test thresholds;
the physical tolerance needed for morphing has not yet been measured.

## Why the changes help

The old floor proxy subtracted the highest support-wheel bottom. An unloaded
support wheel could therefore raise the inferred floor. For a level rigid floor,
the ground cannot lie above the lowest support-wheel bottom without penetration.
The new estimate subtracts an upper bound for that lowest support bottom from a
lower bound for the active wheel, using modeled radii 48 ± 2.5 mm. It retains the
10 mm controller gate. This is a geometry-bound correction, not a contact sensor
or permission to use the behavior on uneven ground. The independent MuJoCo test
checks that the bound does not overestimate the nominal geometric clearance.

Rear hip apex magnitudes increase from 1.05 to 1.2 rad to pass the bias corners.
Shift, lift and recenter reference durations decrease from 2 to 1.5 seconds;
landing stays at 3 seconds. Velocity/acceleration bounds are retained.

Wheel holding follows the actual calibrated target during descent, verifies the
post-touchdown angle, and preserves completed targets during later wheel moves.
A disturbance above 0.10 rad (~5.7°) invalidates completion without a grounded
rotation retry. The current angle is held instead; an operator must request a
new attempt. The failed-wheel mask distinguishes failure from success.

## Reproduce and continue

Use the build/environment instructions in [KEYFRAME_ALIGNMENT.md](../../KEYFRAME_ALIGNMENT.md).
The saved run used MuJoCo 3.6.0, NumPy 2.4.6, GCC 13 and the exact C++ controller.
The simulation obtains angular velocity directly in the regular body frame via
[MuJoCo's object-velocity API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-objectvelocity).

```bash
python -m motion.keyframe_align.audit_suite --output /tmp/keyframes-new --video
python -m motion.keyframe_align.upload_audits /tmp/keyframes-new
```

The first command runs CPU simulation and saves evidence locally. The second
uploads the saved evidence using the shared W&B logger and verifies remote Media.
It defaults to online; `--wandb offline` explicitly retains an offline SDK run.
Never edit core/configuration/library while the matrix runs. Failed cases remain
in the output directory and must be uploaded too.

The robot-code adapter is isolated on `codex/keyframe-robot-integration` and uses
identical core/configuration files with a hash manifest. That branch's
`KEYFRAME_LAB.md` records software checks and remaining target-specific checks.
The Pi was unreachable during preparation; no deployment or hardware startup is
included in this result. Physical calibration/ring marks, ground assumptions,
tracking, actual timing and motor behavior still need a supervised first trial.
