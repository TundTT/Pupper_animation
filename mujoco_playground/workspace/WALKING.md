## Latest heating-backpack policy (September 13, 2026)

The latest trained leg policy is in
[`trained_policies/backpack_2026-09-13`](../../trained_policies/backpack_2026-09-13/README.md).
This release supersedes the gap-only checkpoints discussed in the historical notes below.
Its matching robot-code controller export is identified by `policies/latest.json` there.

# Capsule-foot walking on the `leg` branch

The active entry point is `python -m workspace.train` (also `workspace.train_walk`).
It trains velocity-commanded walking with all 12 joints position controlled.
The wheel branch was the reference for remote setup and PPO sizing; its wheel
velocity actuators and wheel reward are not used here.

## Model and starting pose

New walking runs use `foot_model=rigid_flush`. The trainer compiles a 62.649961 mm
capsule from the source XML: the distal cap extends 2.649961 mm to the outer ring's
axial bottom, with the 12 mm radius and proximal endpoint unchanged. Floor and
foot contacts use solref `[.008,1]`, solimp `[.99,.99,.001,.5,2]`; contact softness
is no longer randomized. The home keyframe is re-settled at about 146.72 mm torso
height. Each run saves `effective_model.xml` for direct viewing, plus the original
source `model.xml`. Open the effective model to inspect the new physics.
Existing checkpoints without a `foot_model` field explicitly retain `legacy_soft`
when evaluated; `--foot_model rigid_flush` requests a deliberate transfer test.

The source `pupper_v3_complete.mjx.position.xml` remains the legacy model.
Its default state now stands with zero actuator inputs; the `home` keyframe
contains the settled pose. Nominal joint targets in FR/FL/BR/BL order are
`[1, 0, -1, -1, 0, 1, 1, 0, -1, -1, 0, 1]` radians. All four feet support the
robot at about 0.1425 m torso height. The offline 20-second checks require
four capsule contacts, less than 2 degrees tilt, and no knee/body floor contact.

**XML controls now mean joint-angle offsets from home**, in radians. Zero holds
home; an absolute desired joint angle must be converted to `target - home`.
Body frames and `joint ref` were rotated together, preserving FK for the same
absolute joint angles. Mesh mounts, mass and inertia are unchanged. Gains are
kp=5, kd=0.25, with the existing 3 Nm force limit. Eight solver iterations replace
the old single iteration. The trainer reads home/gains/limits from the XML.

The environment uses MJX directly through the Brax Env API, because Brax's MJCF
importer rejects nonzero joint reference values. Do not run the legacy leg-lift
environment against this model. Its old sources remain as reference.

## Rubber ring reward

The capsule is the physical supporting foot. The mesh ring remains visual only.
`ring_outline.json` contains 96 points per foot: the outer perimeter sampled on
both axial edges. `walk_geometry.py` transforms these points with each actual
mesh mount, including the left-side clocking. The STL hash is checked before use.
This is an undeformed-outline **ground clearance proxy**, not contact force or a
rubber deformation simulation. It supports a plane through the world origin,
including the randomized floor tilt;
obstacles/heightfields need a corresponding terrain-distance implementation.

- `ring_side`: side penetration/clearance violation, measured per foot.
- `ring_rub`: world-XY speed squared, weighted by side-point proximity to the floor.
- `ring_bottom`: disabled (weight zero); retained only for legacy diagnostics.
- `ring_side_contact`: fraction of feet with any side probe below the floor,
  averaged over substeps, with weight -6. This charges frequent shallow side
  brushing in addition to the depth and rubbing costs.
- `ring_side_fraction` and `ring_penetration_m`: diagnostics, not policy inputs.

The lower arc has no compression reward or penalty. The capsule is a rigid shape
with stiff numerical contacts; the visual ring has no load-bearing collision.
Side costs use the maximum violation per foot so weights do not depend on point count. Ring costs
are averaged over the simulation substeps. These weights/allowances are starting
values for hardware calibration, not measurements of rubber stiffness or wear.

Foot support uses actual capsule-floor contacts. Slip uses the velocity of the
lowest capsule surface point; the foot sites now mark the lower cap centers.
Other rewards track XY velocity/yaw, discourage slipping, penalize body/knee
contact and abrupt actions, and reward bounded swing time at touchdown. Commands
span vx [-0.35,0.35], vy [-0.15,0.15] m/s, yaw [-0.8,0.8] rad/s.
No imposed gait phase: PPO must discover a gait. Rewards do not guarantee success.

The actor uses four 36-value history frames (144 inputs), matching the existing
locomotion layout: angular velocity, projected gravity, XY/yaw command, desired
upright direction, joint offsets, last actions. Ring information is reward-only.
Friction, gains, mass/inertia and small symmetric CoM shifts are randomized;
sensor noise and one-step action latency are included. No wheel-specific or
legacy leg-lift CoM correction is inherited.

## Remote setup: two RTX PRO 6000 Blackwell GPUs

The `origin/wheel` README records two 96 GB Blackwell cards and successful runs
on one visible GPU. Confirm the actual remote GPUs/driver with `nvidia-smi`.
Use a separate environment to preserve the working wheel setup:

```bash
cd Pupper_animation/mujoco_playground
bash workspace/tools/setup_walk_gpu.sh

# Geometry/standing checks, then a tiny end-to-end PPO integration check.
.venv-walk/bin/python -m pytest workspace/tests -q
CUDA_VISIBLE_DEVICES=0 .venv-walk/bin/python -m workspace.train --smoke

# A short GPU training trial. Defaults keep batch_size*num_minibatches=8192,
# divisible by 512 environments. Evaluate before committing to the full run.
CUDA_VISIBLE_DEVICES=0 .venv-walk/bin/python -m workspace.train \
  --num_timesteps 300000 --num_envs 512

# Full run, detached so an SSH disconnect does not stop training.
CUDA_VISIBLE_DEVICES=0 nohup .venv-walk/bin/python -u -m workspace.train \
  --use_wandb > walk-train.log 2>&1 &
disown
```

Defaults: 200M environment steps, 8192 environments, 15 evaluations, ELU network
128/128/128, 50 Hz policy. For a second independent seed, use GPU 1 with `--seed 1`
and a different log filename. A second GPU is optional; one job defaults to one
visible GPU just as the wheel runs did. `--smoke` is tiny and is not a useful policy.
W&B is optional and requires the remote account already be logged in.

**Use `.venv-walk/bin/python` directly. Do not use `uv run` or `uv sync` on the
existing wheel environment.** The wheel notes document incompatible lockfile
re-syncs disabling CUDA. This setup pins JAX/jaxlib/CUDA plugin versions together;
local CPU verification does not certify the remote driver, so the GPU check is
required before a long run. No SSH credentials or remote paths are hardcoded.

## Outputs, evaluation, and export

The current selected candidate is **21,626,880 steps** from
`output/stride_trials/preferred_long/walk_2026-09-07_16-48-35` (W&B `zw331stu`).
Use `selected_params` and the W&B **`eval/video_selected`** panel. It warm-starts
directly from the user's preferred `walk_2026-09-07_15-29-41` ring6 policy.
The later highest-reward checkpoints shorten the stride again, so `best_params`
is not the selected policy. See [stride and clearance results](validation_stride_2026-09-07/RESULTS.md).

Defaults now use a **16 mm / 320 ms** swing curve, shape weight **-6**, per-foot
error weights **3/3/1/1** (FR/FL/BR/BL), and a **12 mm** touchdown peak threshold
with shortfall weight **-.5**. Front weighting changes the error cost only; the
swing bonus, action bounds, ring-side penalties and rigid contacts are unchanged.
Action-acceleration weight is **-1.6** to discourage abrupt reversals. Saved old
policies explicitly retain equal foot weighting when their metadata lacks it.

At forward .20 m/s and friction2, the selected policy has roughly **98 mm** strides,
**2.17 Hz** cadence, **224 ms** major swings and **9.2/11.3 mm** front peaks.
All39 flat direction/seed trials survive and three10mm obstacle courses complete,
versus one of three for the preferred policy under the same rigid physics.
Some side strikes and ring-side overlap remain; this is not hardware validation.

The earlier front3 candidate reached13--14mm front peaks, but its3.57Hz/48mm
strides were rejected by the user as too tap-like. It is superseded. Its
[clearance-only results](validation_clearance_2026-09-07/RESULTS.md) remain available.

`evaluate_walk_obstacles.py` adds real stiff transverse bars for an independent
native MuJoCo stress test, without terrain observations. It reports course
completion, capsule side/top contacts and ring-side overlap. Compare both
policies with this evaluator; it is not bit-identical MJX evaluation.

The reverse-walking follow-up uses 20% stand, 20% forward, 20% reverse, 10%
lateral, 10% yaw, and 20% mixed commands. It adds a stronger signed air-time
incentive, a soft 4 mm swing-peak goal, contact-only tip-lean and touchdown
approach-speed penalties, and speed-dependent tracking tolerances. See
[the ring explanation and trial protocol](validation_reverse_2026-09-07/RING_AND_TRAINING.md)
for the intended tip support, unchanged ring settings, and exact new penalties.
The soft-contact front-drag experiment produced **43,253,760 steps** in
`output/front_drag_trials/ring6/walk_2026-09-07_15-29-41`. It is now superseded by
the requested flush rigid-capsule physics and is only a warm-start candidate.
That experiment raised the training friction range to **0.8–2.5**, swing reference
to **8 mm**, swing-shape weight to **-3**, linear tracking to **6**, and side-contact
weight to **-6**. Action scales/home/gains are unchanged. Nominal XML friction stays
1; `--floor_friction 2` produces the high-friction review used in this run. The
trainer passes `--friction_range MIN MAX` explicitly into domain randomization.

Front lift improves, with some tracking and fast-motion tradeoffs. Higher friction
also changes penetration under the current contact model, so it is a stress test
with a compliance confound. See [full front-drag results and videos](validation_front_drag_2026-09-07/RESULTS.md).
This candidate has not been tested on hardware or copied into `robot-code`.

The earlier jitter follow-up selected the **129,761,280-step checkpoint** from
`output/jitter_trials/tip_credit3/walk_2026-09-07_04-04-36`. Use `selected_params`
and `policy_walk_selected.json`; `best_params` still denotes the checkpoint with
the highest training evaluation reward. See [measured results, limitations, and
video](validation_jitter_2026-09-07/RESULTS.md). W&B run `61am09cq` contains the
chosen video under **`eval/video_selected`**.

That recipe introduced a second action-difference penalty (-0.8), a 160 ms,
6 mm swing reference triggered by each foot's liftoff, and a small credit for
following that reference. Early contact cannot cancel the remaining curve cost.
The old height-area reward is disabled. Knee action scale is now **1.1 radians**
(hip 0.5, abduction 0.25), with a 3 mm lower moving height reference. The height
weight is -12, linear/yaw tracking weights are 4/3, and side-depth/side-contact
weights are both -3. These changes make longer swings feasible and discourage
shallow ring-side brushing. The 144-input/12-output layout is unchanged, but
deployment **must use the new exported action scales**. No external gait clock
or runtime action filter was added.

The previous reverse follow-up completed 86.5M steps; its 57.7M-step checkpoint improved
slow reverse tracking and clearance, but retains short reverse swings and has
more ring-side overlap than the previous policy. It is an experimental candidate;
[measured results and the review video](validation_reverse_2026-09-07/RESULTS.md)
include the regressions. The latest checkpoint supersedes that candidate.
`evaluate_gait_suite.py` evaluates 13 fixed commands over three seeds and records
250 Hz foot traces. Showcases now append faster reverse and lateral segments to
the original 16-second command sequence; the current 32-second review also includes
both spin directions and a final stop.

The earlier September 7 audit follow-up raised only the action-rate reward weight to
`-0.30`; torque (`-0.0002`), air time (`0.3`), foot slip (`-0.6`), the 28 mm
clearance cap, joint action bounds, and 9 mm bottom allowance retain their values.
The observed gait had short rounded swings below the cap; this is a roughness
experiment, not a claim that the old policy dwelled at maximum clearance.

Length DR uses configurable `leg_length_common_range=(.90,.98)` times
`leg_length_per_leg_range=(.96,1.04)`, giving `0.864–1.0192` upstream-offset scale.
It acts around the extended foot assembly, moving the ring and capsule together
and preserving the flush tip at every sampled length. The 2.65 mm extension is
present at both length endpoints; capsule length is not randomized independently
of the ring. New four-second flat standing heights at friction 1 span roughly
135.38–148.33 mm, with nominal 146.71 mm. Mount perturbations remain parent-frame X/Y
rotations of ±0.07 rad. Brax holds these physical parameters fixed per environment
slot across episodes. At reset, the torso-height reference is adjusted using the
change in neutral capsule-bottom projections for that slot's actual geometry.
Torso-height reward and low-height termination use distance along the floor normal.
The geometry and reward updates together constitute a combined training run;
comparison with the old checkpoint cannot isolate action-rate causality.

Training evaluation and showcases disable pushes; showcases render every control
step at 50 fps. Standalone evaluation also disables pushes by default; pass
`--with_pushes` to test the saved training disturbance probability. Evaluation
rewards therefore are not directly comparable with older disturbed evaluations.
Run folders include source snapshots, warm-start provenance, evaluation settings,
and `best_checkpoint.json`. Best parameters are saved at each improvement so an
interruption does not lose selection of the strongest evaluated checkpoint.
Warm starts default to learning rate `5e-5`; `--learning_rate` overrides it.
Videos are enabled by default, including W&B panels `eval/video`,
`eval/video_final`, and `eval/video_best`. For shorter trials, use
`--final_videos_only` to skip intermediate rendering while retaining final and
best videos. `--no_eval_videos` explicitly disables **all** videos. The chosen
mode and W&B run path are saved in `run.json`.

To attach a locally rendered video to its original finished W&B run:

```bash
.venv-walk/bin/python -m workspace.upload_walk_video \
  --params workspace/output/<run>/best_params \
  --video workspace/output/<run>/rollout_best.mp4 \
  --run QuadMorph/pupper-leg/<run_id>
```

This appends `eval/video_best` with checkpoint provenance and preserves existing
training metrics. The run must already exist and match the checkpoint directory.

Each run writes `run.json` (settings, dependency versions, hashes, home pose),
`metrics.jsonl`, intermediate `params_<step>` files, and final `mjx_params` under
`workspace/output/walk_<timestamp>/`. Intermediate weights survive a stopped run.
`--init_params PATH` warm-starts weights; it does not restore optimizer state.
The saved XML snapshot needs its corresponding STL assets to be loaded elsewhere.

```bash
.venv-walk/bin/python -m workspace.evaluate_walk \
  --params workspace/output/<run>/mjx_params --command .2 0 0 \
  --video workspace/output/<run>/forward.mp4
.venv-walk/bin/python -m workspace.export_walk \
  --params workspace/output/<run>/mjx_params
```

Evaluate stop, forward, reverse, turning, and combined commands. Check episode
length, velocity error, tilt, foot contacts, ring-side fraction, ring penetration,
and video together. Evaluation refuses a model whose hash differs from training.
Exports use the locomotion observation layout and absolute home joint angles;
ring probes are not exported as sensors. Export is not hardware deployment or
hardware validation. The robot must use matching home/action scales/gains and
upright-only orientation commands until separately trained otherwise.

Export folds normalization in double precision and substitutes the constant
upright-command inputs before writing weights, avoiding cancellation from their
near-zero variance. Evaluation applies the same observation normalization as PPO.

## Verification performed locally

Both raw XML startup and the home keyframe held for 20 seconds on four capsules
with no body/knee contacts; maximum tilt was 0.111 degrees. Geometry and inertial
values are unchanged, and FK matched the pre-change model at 20 random joint
poses. Tests cover side rubbing, bottom allowance, tilted rings, point velocities,
JIT rollout/reset behavior, randomized Brax wrapping, and exported MLP parity.
A 64-step CPU PPO integration run completed and saved intermediate/final weights;
it is only a pipeline test, not a walking policy. Subsequent CUDA training and
direction-specific evaluation results are recorded in the September 7 validation
directories linked above.
