# Lift and alignment with the backpack and 9 mm spacing

September 12, 2026: the new candidate passes **26/26 CPU simulation scenarios**.
It retains the successful hardware position PID and needs no neural training.
Physical performance with this morphology is still pending.

Development branch: `codex/backpack-align`. Robot adapter:
`codex/backpack-robot-integration`, based on `robot-code` f6981f7. Do not deploy
the older ROS stack in the simulation branch. The authoritative model is
[model.xml](training/wheel_align/model.xml), SHA256
`4d23ca9de08c78b40cabdfce24a41649a9695132b5900f0823f64f801048bcbe`.
The operator's corrected geometry was preserved. Its actuator representation
was changed to torque motors so simulation explicitly applies the same
position/velocity/effort law used by the robot.

## Changes and their reason

The unchanged lab poses failed with the backpack: the fronts did not clear the
wheel-spacing gate and the rears tilted enough to lose the rotation gates.
Offline numerical fitting adjusted all four support poses, including support
polygon margin in the rear-pose objective. Full dynamic rollouts, rather than
the fitting score, selected the candidate. Runtime is still deterministic.

The exact successful robot controller core is now also used in the simulator.
All 12 joints receive angle targets. Wheel P=4, D=0.15, near-target I=0.5,
integral torque cap=0.10 Nm and window=0.10 rad are unchanged. The simulator
applies proximal P=5/D=0.25 and the hardware-style 3 Nm torque bound at 520 Hz.
These values describe the software command contract; they are not measured
motor identification or a guarantee that simulated torque matches physical torque.

[config.json](motion/keyframe_align/config.json) contains the candidate poses
and gains. The original successful hardware settings are preserved in
[lab_baseline_config.json](motion/keyframe_align/lab_baseline_config.json).
Landing reference duration increased from 3 to 4.5 seconds, and recentering
from 1.5 to 2 seconds, to keep the simulated near-ground descent below 30 mm/s.
The 5 mm floor gate, angle tolerances and rotation speed limit are unchanged.
Filters and settling can extend configured phase durations.

Clearance now includes the original body box plus all three backpack collision
boxes and the 39.35 mm wheel-center offset. Generated values live in
[model_geometry.hpp](motion/keyframe_align/model_geometry.hpp). After another
XML correction, regenerate and re-audit; do not just replace recorded hashes.

## Evidence

[W&B run and Media](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/f5f9c4a12b7348a6)
contains the actual complete simulation video and the evidence archive, including
failed development audits. [Local video](hardware_testing/backpack_align_2026-09-12/rollout.mp4).

The 26 cases cover nominal motion, eight IMU bias directions, four randomized
starting-angle/friction cases, four combined cases, cancellation in five phases,
backpack mass at 80% and 120%, an added 20 g per wheel, and a combined heavier case.
Bias is applied to IMU input; it is not a physical slope. Added wheel mass follows
the existing wheel COM/inertia distribution and is only a sensitivity assumption.
No slope, deforming hot polymer, heater actuation or physical sensor failure is
represented by these dynamics cases.

Across the matrix, rotation retained all three support contacts and at least
20.5 mm actual modeled floor clearance. Peak body tilt was 3.54 degrees, peak
commanded joint effort 0.714 Nm, and peak near-ground descent 26.1 mm/s. Nominal
FL/FR/BR/BL final errors were +0.98/-0.95/-0.54/+0.55 degrees; full 180-degree
cycles took 22.5/23.2/22.6/23.1 seconds. All four completed without a timeout.
Cancellation cases correctly return to hold and are not claimed to align.

[suite.json](hardware_testing/backpack_align_2026-09-12/suite.json) and per-case
reports record source and binary hashes. `evidence.tar.gz` preserves exact tested
source bytes, raw traces, the failed unchanged baseline, two failed intermediate
candidates and fitting records. The first baseline attempt failed during log
provenance collection; its console log is preserved and the identical physical
case was rerun to save the failed audit. Source was dirty during evaluation;
hashes and the snapshot, not the base commit alone, identify what was tested.
Subsequent fitting CLI cleanup, documentation and upload tooling do not change
the tested motion/controller/model. Nine Python controller/geometry tests and
the portable C++ test passed. Robot ROS checks are documented in its handoff.

## Reproduce or revise locally

Use Linux/WSL, Python 3.12 and a C++17 compiler. No GPU is needed.

```bash
python3 -m venv .venv-keyframes
.venv-keyframes/bin/pip install -r motion/keyframe_align/requirements-backpack.txt
source .venv-keyframes/bin/activate
python -m motion.keyframe_align.generate_geometry --check
cmake -S motion/keyframe_align -B /tmp/backpack-keyframes-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/backpack-keyframes-build -j2
export KEYFRAME_ALIGN_LIBRARY=/tmp/backpack-keyframes-build/libkeyframe_align.so
ctest --test-dir /tmp/backpack-keyframes-build --output-on-failure
python -m pytest motion/keyframe_align/test_controller.py -q
MUJOCO_GL=egl python -m motion.keyframe_align.audit_suite --output /tmp/new-backpack-audit --workers 2 --video
```

Use a new output directory each time; retain failures. Do not edit hashed source
while the suite runs. Optional `fit_pid_support --leg 2 --config <input.json>
--output <fit.json>` generates a rear-right candidate; 0/1/2/3 means FR/FL/BR/BL.
The result is not automatically installed or accepted. `fit_pid` is the earlier
objective without support-polygon margin; it is retained for experiment history.

The saved September 12 evidence can be uploaded without rerunning motion using
`python scripts/upload_backpack_audit.py`. It uses existing W&B credentials and
the shared logger, defaults online, and supports `--mode offline`. Its saved run
identity avoids creating a duplicate during retry. An offline run is not an upload.

## Hardware facts and remaining input

- Backpack mass 0.60191707499 kg and inertia come from the supplied CAD report;
  mounting revision 3 was visually approved. See the precise source distinctions
  in [the module README](models/heating_module/README.md). Total modeled mass is
  3.81991707499 kg before optional sensitivity mass.
- The 9 mm change translates the terminal wheel assembly outward along its hub
  axis; joint origins, axes and CAN ordering stay as described by the source
  model and robot `components.xacro`. Printed spacer mass remains unspecified.
- The position PID baseline is the operator-reported successful floor sequence
  recorded in robot-code `hardware_testing/keyframe_align/LAB_BASELINE_20260912.md`.
- New-model physical validation requires confirming the actual fitted backpack
  and spacers, then the documented supported homing pose. Start supported, enter
  stance, and place on the floor afterward. The desired saved pose does not
  eliminate live-session encoder calibration after a restart. Heating stays manual.
