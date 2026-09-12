# Post-cooled inverted-triangle development

This module begins with four rigid, already-transformed, point-up shins. Heating,
deformation and hardware motion are outside its scope. Use the original tip length
and the approved additional 9 mm axial gap. Do not silently enlarge the gap.

Read README.md and PC_HANDOFF.md before extending the experiment. Work only on
codex/inverted-triangle or a user-requested derivative. Preserve robot-code and
the historical alignment/training sources. No SSH, robot startup, deployment,
calibration changes or policy activation is authorized by a compute handoff.

Floor-only convex hulls are deliberately separated from the nonconvex CAD
self-clearance audit. Do not count a visually intersecting model as passing,
disable a failed audit, weaken criteria to obtain PASS, use qpos assignment as
dynamic validation, introduce ground-truth contact/base position into a deployable
controller, or reset to a fresh four-inverted state between flips in a full-sequence
claim. A single-flip pass is not a full-sequence or robustness pass.

Maintain actuator limits and measured-state conventions. Search objective values
are candidate-selection aids, not acceptance. Preserve failed candidate reports.
Record source/model/candidate hashes, environment versions, seeds and simulation
steps. Reuse training/wandb_logging.py; online W&B is the compute-run default, and
actual rollout videos must appear as wandb.Video. Explicit offline/disabled modes
are allowed for local iteration. Report upload status accurately.

Do not modify code/model/configuration during a running audit. Rerun affected
checks when they change. Source provenance must remain reproducible.
