# Alignment endpoint to roll-to-stand

This branch adds a simulation prototype for the newly requested entry motion.
The Pi can remain off. Nothing here starts ROS, heats a limb, deploys a controller,
or authorizes motors. The previous hardware controller has not been replaced.

The required sequence is: finish alignment and lowering; manually form/cool the
four rigid limbs; start from the resulting measured posture; smoothly enter a
usable posture; roll all four hubs forward together; settle near the walking
policy's default posture. The entry must tolerate modest posture errors and
incomplete compression. Do not require the operator to recreate the old exact
simulation keyframe. Do not silently reset joint state to that keyframe.

## Sources and remaining uncertainty

- User-confirmed: cooled limbs are rigid; approximately 10 mm difference between
  longest and shortest, small angular variation, and the additional 9 mm axial
  spacing in the XML is correct. The lower central underside can remain above
  the floor; less compression also shortens the upper point. Heating is manual.
- Pinned triangle parent: `221e16681952bb9350bd2c1019c2b092c347432a`.
  `source_manifest.json` retains the original tip, 9 mm gap, and approved backpack.
  `core.verify_sources()` still verifies the original XML and all nominal assets.
- Actual endpoint captured locally: backpack **deterministic keyframe** controller
  `a7e3bb1e67f73b4fb14ee6658d66d0263fd2bfd8`,
  `motion/keyframe_align/{simulate.py,controller.hpp,config.json}`.
  Capture records qpos, qvel, calibrated wheel homes, final proximal commands,
  masks, model/config/binary hashes, and source identity. Its joint anchors, axes,
  and body rotations match the triangle model at common neutral (error <1e-12).
  That comparison does not certify mesh shape, inertia, or physical calibration.
- **Do not conflate alignment versions:** robot-code `25301741` full launch still
  selects `policy_wheel_align_motion_v5.json` for its legacy alignment action.
  Its `wheel_align_motion.hpp` returns the eight proximal targets toward
  `[1,0,-1,0,1,0,-1,0]`, but this is a command, not the measured endpoint.
  The keyframe replay is not proof of a v5 policy handoff. If v5 is the selected
  upstream behavior, replay its actual checkpoint and export equivalent records;
  `native_audit.py` with zero residuals is not that replay.
- Hardware contract: `robot-info` commit
  `e9b04173b034d595a85e6147d73a45cdfe9393e3`, specifically
  `robot_info/HARDWARE.md`, `POLICY_INTERFACE.md`, and `PRE_LAB.md`.
  Preserve canonical FR/FL/BR/BL joint order, proximal soft limits, 3 Nm effort,
  and the 520 Hz controller/52 Hz feedback contracts. The default leg profile's
  hub position limits do NOT cover the inverted phase; eventual deployment must
  use the reviewed triangle-roll profile and its continuous hub coordinates.
  The unbounded simulation hub sentinel is not permission to override hardware
  limits. IMU/contact estimates must match actual available signals.
- `handoff_shape.py` is a **synthetic rigid shape family**, not measured polymer
  mechanics. Support extension 0..5 mm is an input to a smooth mesh deformation,
  not a direct world-floor gap. Tip shortening is correlated with it; explicit
  development cases include the full 10 mm between-limb spread. Trial bends are
  +/-3 degrees. Attachment region and axial coordinates are fixed. The same
  changed mesh feeds visual geometry and detailed CAD checks; two convex half
  hulls approximate floor contact. Contact convergence is still unvalidated.
  Nominal mass/COM/inertia are retained, with separate dynamics stress available.

## Implementation

`handoff_entry.py` starts from measured joint state and the previous proximal
position commands, holds measured hub angles during entry, then performs the
simultaneous roll. All phases share one velocity/acceleration limiter. Entry
requires measured settling and reports failure on timeout. Support correction
uses the parent's bounded feedback. This is a Python simulation controller;
portable C++ parity and the robot adapter remain subsequent work.

`capture_alignment_terminal.py` replays an unchanged keyframe checkout and
exports its actual terminal state. Failed alignment records are saved and rejected
by `handoff_probe.py`. The wheel-to-triangle model boundary uses calibrated hub
phase offsets once; no live encoder/state wrapping. At this **post-cooling model
boundary only**, root Z is positioned so the lowest intended foot reaches the
floor. Root XY/orientation, joints, and velocity are carried through. Heating and
the intervening physical deformation are not simulated. There are no further
physics-state resets through entry, roll, and standing. Do not describe this as
continuous wheel-to-cooled-triangle physics.

## Run on the PC

Use separate worktrees; preserve previous accepted and failed experiments.
Fetch `codex/alignment-roll-handoff-20260913` and record its exact HEAD before
running. Do not merge it into robot-code yet. LFS assets and the frozen walking
policy are required:

```bash
git lfs pull --include='motion/inverted_triangle/assets/**,ros2_ws/src/neural_controller/launch/policy_walk_v2.json'
python3 -m venv .venv-handoff
.venv-handoff/bin/pip install -r motion/inverted_triangle/requirements.txt
.venv-handoff/bin/python -m pytest motion/inverted_triangle/test_handoff.py motion/inverted_triangle/test_formation.py motion/inverted_triangle/test_model.py motion/inverted_triangle/test_roll_to_stand.py -q
```

Keep the keyframe simulator in its own environment: it pins MuJoCo 3.6.0/Numpy
2.4.6, while triangle replay pins MuJoCo 3.3.7/Numpy 2.2.6. Record both versions;
the explicit model boundary is not a bitwise cross-version continuation.

```bash
# In a separate alignment worktree at a7e3bb1e67f73b4fb14ee6658d66d0263fd2bfd8:
git lfs pull
python3 -m venv .venv-alignment
.venv-alignment/bin/pip install -r motion/keyframe_align/requirements.txt
cmake -S motion/keyframe_align -B /tmp/handoff-keyframes -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/handoff-keyframes -j2
export KEYFRAME_ALIGN_LIBRARY=/tmp/handoff-keyframes/libkeyframe_align.so
MUJOCO_GL=egl .venv-alignment/bin/python /ABS/HANDOFF/motion/inverted_triangle/capture_alignment_terminal.py \
  --alignment-root "$PWD" --seed 0 --video --output /ABS/RESULTS/alignment-0

# Back in the new handoff worktree:
MUJOCO_GL=egl .venv-handoff/bin/python -m motion.inverted_triangle.handoff_probe \
  --terminal /ABS/RESULTS/alignment-0/terminal.json --output /ABS/RESULTS/handoff-0
```

The probe logs online to W&B by default, including its actual continuous video
as `wandb.Video`, integration records, and configuration/source hashes. Reuse
existing credentials. `--wandb offline` and `--wandb disabled` are explicit local
options; report the actual upload status. `--synthetic` is only a preliminary
test, never substitute evidence of an alignment handoff. `--no-cad` is a quick
diagnostic and cannot pass engineering acceptance. Nonzero exit preserves results.

For each entry in `handoff_development_cases.json`, write its `config` object to
a JSON file and pass `--config FILE` with a real `--terminal` record. Keep separate
output directories. Preserve failed trials and render them. For WSL unable to
read Windows Git metadata, the capture adapter optionally accepts a host-generated
`--source-record` containing `source_commit`, `source_dirty`, and a relative-path
SHA256 map `files`; it verifies the recorded files before replay.

## Work for the PC agent, in order

1. Reproduce the saved local observations and select/document the intended
   upstream alignment implementation. Collect real successful endpoints across
   seeds and the existing alignment perturbations, including full lowering and
   settling. Carry the previous proximal command; never import hub torque/velocity
   outputs as hub position targets. Validate calibrated hub-phase mapping.
2. Resolve the floor-contact counterexample before parameter optimization: at
   ideal full compression with neutral upper joints, detailed lower-housing CAD
   lies about **1.118 mm below the intended foot plane**. This is present at the
   starting boundary, not caused by a violent roll. The user's earlier physical
   estimate was about 5 mm clearance; keep that discrepancy explicit. Inspect
   actual endpoint geometry and partially formed support shapes. Do not fix this
   by deleting motor contacts, weakening force gates, teleporting the body after
   initialization, or inventing a physical dimension. If no contact-free cold
   start exists for an endpoint, report it as infeasible and identify whether the
   supported posture must be established before manual formation. The operator
   may ultimately need to verify the cold geometry; no powered test is needed now.
3. Compare a short measured-state entry with a coordinated direct roll from the
   endpoint. Use CPU trajectory fitting/search first. Optimize full simulated
   dynamics, including initial errors, command/gain continuity, contact, and
   final support. More compute/RL is available if this fails; preserve a deployable
   observation/action contract before any training. Do not assume perfect contact
   sensors or per-limb geometry measurements on the Pi.
   The clean-commit local diagonal 10 mm trial already times out in entry with
   the rear-right hip about 7 degrees away from its target. Reproduce this saved
   counterexample; consider load-aware support during entry or a feasible posture
   region instead of making every differently formed limb track one exact pose.
4. Check contact approximation convergence against a finer decomposition of the
   SAME shape, including a zero-variation control versus the original nominal
   hull. Validate that changed inertia assumptions do not determine success.
   Keep source geometry and 9 mm mounting unchanged in accepted comparisons.
5. Run the development matrix including 10 mm diagonal/front/rear/side length
   contrasts. Freeze the candidate, then generate fresh held-out endpoint,
   shape, friction, sensor, and dynamics cases. Re-run the parent's friction/stress
   suites. Preserve failures; no cherry-picking successful seeds.
6. Keep the existing engineering gates: no motor/body floor load, bounded effort,
   measured and commanded speeds, acceleration, tilt, dense nonconvex shin/shin
   (especially front-to-rear), shin/motor/body/backpack clearance, and at least
   three seconds of settled four-tip support close to the walking default.
   Dense replay must reproduce every physics step; do not audit only planned poses.
7. After entry-to-stand succeeds, validate the actual frozen walking-policy
   activation/history/fade/handoff without resetting body/joints/velocity. The
   parent already has known walking failures; final pose similarity is not proof
   they are fixed. Export/port the candidate to the reviewed robot interface only
   after simulation evidence, with parity and lifecycle/stop checks.

Log all experiments to entity `QuadMorph`, project `wheel-leg lift and align
triangle base`. Publish actual videos in Media, configurations, model/source
hashes, environment steps, and optimizer/checkpoint metadata when applicable.
Commit and push results plus an honest acceptance matrix. Return the exact
commit, one continuous real simulation video, failure links, and a concise account
of remaining hardware work. No robot access, heater activation, or motor motion.
