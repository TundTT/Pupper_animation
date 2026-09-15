# Capsule-foot walking on the `leg` branch

The active entry point is `python -m workspace.train` (also `workspace.train_walk`).
It trains velocity-commanded walking with all 12 joints position controlled.
The wheel branch was the reference for remote setup and PPO sizing; its wheel
velocity actuators and wheel reward are not used here.

## Model and starting pose

Drag the existing `pupper_v3_complete.mjx.position.xml` into MuJoCo as usual.
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
rubber deformation simulation. It currently supports only flat ground at world Z=0;
obstacles/heightfields need a corresponding terrain-distance implementation.

- `ring_side`: side penetration/clearance violation, measured per foot.
- `ring_rub`: tangential speed squared where a side point is touching the floor.
- `ring_bottom`: bottom compression beyond a 6 mm allowance.
- `ring_side_fraction` and `ring_penetration_m`: diagnostics, not policy inputs.

The lower arc of the ring permits incidental compression. This includes the
flattened bottom next to the leg tip; ordinary standing penetrates the undeformed
outline by about 4.4 mm and incurs zero ring penalty. Side and bottom costs use the
maximum violation per foot so weights do not depend on point count. Ring costs
are averaged over the simulation substeps. These weights/allowances are starting
values for hardware calibration, not measurements of rubber stiffness or wear.

Foot support uses actual capsule-floor contacts. Slip uses the velocity of the
lowest capsule surface point; the foot sites now mark the lower cap centers.
Other rewards track XY velocity/yaw, discourage slipping, penalize body/knee
contact and abrupt actions, and reward bounded swing time at touchdown. Commands
start conservatively at vx [-0.2,0.4], vy [-0.1,0.1] m/s, yaw [-0.6,0.6] rad/s.
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
it is only a pipeline test, not a walking policy. Remote CUDA training has not
been launched or verified from this machine.
