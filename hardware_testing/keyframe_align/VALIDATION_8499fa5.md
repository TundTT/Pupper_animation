# Keyframe prototype validation — 2026-09-11

Controller source: `8499fa562e7587f12ec589a2223f6926d323373b` on
`codex/align-motion-v2`. All four final runs used clean source and binary SHA256
`caffea6cd7a4407107a32980d37754a4ecf50aa2945a8dc9fe36e5e60a4188b9`.
This is native MuJoCo CPU simulation, with no training, checkpoint or robot motion.

[W&B run and Media video](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/ebcfaec44a3e4d42)
contains the actual 93-second nominal rollout (`motion/keyframes`) and a
`motion-audit` artifact with configurations, traces and audits, including failures.
Upload completion and the remote video file were verified through the W&B API.
Overall status is explicitly **failed; hardware not validated**.

## Results

The standalone C++ build and seven controller tests passed. Native simulation
used MuJoCo 3.6.0, NumPy 2.4.6 and the existing wheel model. Timing below includes
shift, lift, feedback rotation, lower and recenter; it is not just lifting time.

| Final scenario | Result | Meaning |
| --- | --- | --- |
| Nominal, seed 0 | Completion/clearance audit passed | All four returned to hold in about 21.1–21.4 seconds each. |
| Synthetic joint friction 0.06 Nm, seed 1 | Passed | All four completed, final errors below 1 degree. This friction is a test assumption, not a measured motor property. |
| Cancellation during lift | Passed | Each returned to hold without a false completion flag. |
| IMU roll and pitch both biased +0.04 rad | Failed | Front-right never rotated, timed out, then lowered; other wheels completed. |

Nominal minimum actual simulated floor clearance during rotation was 19.8 mm;
minimum modeled wheel-to-wheel gap across the sequence was 16.6 mm. Maximum
near-ground descent speed was about 24 mm/s. These measurements describe the
simulation and its geometry, not guaranteed physical clearances.

Nominal final errors after lowering were FL +0.19°, FR −2.44°, BR −0.10°,
BL +0.12°. The front-right result exposes a separate limitation: the completion
flag checks angle settling while raised, then descent; it does not recheck the
angle after touchdown. The current audit reports that angle but does not include
a final-angle or descent-speed threshold in its pass predicate. Thus its nominal
`passed: true` is a completion/clearance result, not full acceptance of alignment.

## Remaining work before porting

1. Reconcile the front floor-clearance estimate with recorded body tilt and modeled
   support geometry. Tune coordinated support/lift poses with clearance margin
   across perturbations, including both signs of IMU error. Do not bypass the gate
   or lower its threshold simply to make this audit pass. Keyframes alone do not
   correct the existing front-clearance sensitivity.
2. Maintain and verify the calibrated angular target through lowering. Decide the
   permissible post-touchdown error, enforce it in completion and audit checks,
   and avoid trying to rotate a grounded wheel as a hidden recovery step.
3. Expand acceptance to final-angle retention, descent-speed limits, varied wheel
   starts and interruptions throughout the sequence. Run video from the exact
   accepted source/library/configuration, preserving failed evidence.
4. Only after these pass, add the robot-code adapter and complete the calibration,
   ROS, ARM64 and pre-lab checks in [KEYFRAME_ALIGNMENT.md](../../KEYFRAME_ALIGNMENT.md).
   No adapter or deployment is included in this prototype.

Attempts using a static optimized pose, a front-hip offset, and coordinated
front/rear hip offsets did not solve the clearance/balance issue. They were
rejected rather than substituted as defaults. Artifact `nominal-2` is explicitly
an earlier dirty development checkout at e663e03; the four `final-*` audits are
the frozen 8499fa5 baseline. The offset candidates have separate configuration
hashes and must not be confused with the default configuration.

Local full evidence and video:
`C:/Users/tundt/Desktop/quadmorph-lab-recordings/keyframes-8499fa5/`.
No historical calibration files are included in this committed evidence.
