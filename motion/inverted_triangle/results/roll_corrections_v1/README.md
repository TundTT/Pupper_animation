# Roll-to-stand corrections: partial success, not full acceptance

The original dynamics failures are corrected: **20/20 friction/seed and 16/16
stress cases pass the roll-to-stand gates** on the backpack model. Formation
robustness and walking transfer remain incomplete. No hardware actions occurred.

[W&B review, case table and actual continuous video](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/03376ccd9cd844aa)
collects the evidence. Individual results and links are in
[evidence/case-index.json](evidence/case-index.json). The
[nominal 66-second continuous rollout](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/4f5d491a77534519)
and [front/rear mismatch rollout](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/686ce5bef5894c4a)
continue actual positions **and velocities**, without resetting the robot between
roll, standing and walking.

| Evaluation | Direct/reference | Final correction |
|---|---:|---:|
| Original friction/seed grid, roll | 14/20 | **20/20** |
| Original stress cases, roll | 8/16 | **16/16** |
| Seven formation development cases, roll | 3/7 | **5/7** |
| Seven development handoffs | 0/7 with unmodified walking actions after support correction | **3/7** |
| Fresh validation conditions, roll | — | **19/25** |
| Fresh validation conditions, continuous handoff | — | **5/25** |
| Fresh independent shape + sensor/dynamics handoffs (subset above) | — | **0/12** |

The 30-case final coverage batch contains **25 fresh conditions and five exact
development repeats**. Its combined result is 23/30 for rolling and 7/30 for the
handoff. The directory prefix `held-out` does not make the five repeats fresh;
the index's `role` field identifies them. Length permutations and common offsets
are separated, and all six centered angle permutations are tested. The assumed
±5° bend range is an engineering sensitivity, not a user measurement. Centered
lengths are ±5 mm, representing the requested 10 mm total spread.

**What changed.** The selected trajectory remains a two-second initial hold and
12-second simultaneous forward quintic roll. Doubling damping during that phase
addresses tracking and support-transfer speed spikes; settling restores the
original gains. A 32-second bounded settling controller uses only joint/IMU
samples, nominal CAD kinematics and command history. It begins with a vertical
PD-effort projection and gradually blends to an effort-magnitude estimate,
equalizing relative estimated loads. Actual shape, contact forces and base
position are not actor inputs. Reference offsets are capped at 0.075 rad and
command offsets at 0.2 rad; command rate limits and the actual 0.1 rad pose gate
remain enforced. Delayed command history is initialized after the one-time left
hub winding choice, before integration.

The failed searches explain the choices: support feedback alone passed 26/36
original cases, leaving ten speed failures. Merely increasing roll duration to
20 seconds did not resolve the targeted dynamics set. The final damping schedule
and support adaptation passed all 36, with worst measured joint speed **1.493
rad/s**, requested torque **0.931 Nm**, tilt **1.555°**, dense CAD gap **2.783 mm**,
and zero motor/body floor force. A separate geometric review measured 4.292 mm
sampled motor/body/backpack floor clearance in nominal development and 1.268 mm
in combined stress seed 22. These sampled distances are diagnostics, not new
gates; the old airborne-leg floor rule does not apply to this ground-roll task.

For walking, the pinned network is unchanged. The simulated executor multiplies
its normalized actions by **[0.72, 0.78, 1.0] per leg**, before the export's
[0.5, 0.25, 1.1] action scales, and stores the effective previous action in history.
This wrapper is **absent from the current runtime**. Unmodified actions produced
CAD shin intersections in all seven development handoffs. Uniformly reducing
motion often stopped useful walking; the selected correction passes nominal,
front/rear length mismatch and right/left length mismatch. Nominal forward speed
is 0.0584 m/s for a 0.1 m/s command, with a 2.148 mm dense CAD minimum. This meets
the declared ±0.05 m/s check, not accurate speed tracking.

**Remaining formation constraint.** The diagonal and combined development shapes
still fail four-tip support inside the 0.1 rad pose gate. A conservative numerical
FK enclosure over that joint cube and the complete eight-degree tilt cone gives
rigid-floor contact-height separation lower bounds of **1.151 mm** and **0.634 mm**.
It includes interpolation and orientation-grid error allowances. This is not
formal interval arithmetic or a separate proof about every compliant MuJoCo
equilibrium. Coplanar optimization witnesses require 0.1506 and 0.1419 rad;
those are found witnesses, not certified global minima.

Larger-reference diagnostics achieved all other roll gates with maximum pose
errors over the last three seconds of **0.2115 rad (diagonal)** and **0.1787 rad
(combined)**. Their terminal errors were 0.2021 and 0.1660 rad. These provide
concrete dynamically supported examples outside the permitted stance, not a
claim that those departures are minimal. All six expanded-reach trials remain
**failed** under the original pose gate, with actual videos:
[diagonal example](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/ab9cfcb88a28403e),
[combined example](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/435330a9f9c74c75).

**Audit scope and fidelity.** All original roll gates are retained. Every physics
step records commands, gains, positions and velocities, and measures requested
torque, joint speed, floor force and contact impulse. Independent command/gain
replay reproduces positions and velocities exactly. CAD uses the original
nonconvex meshes for every shin pair and each shin against motors, body and
backpack. Sampling is 52 Hz, refined to 520 Hz within 50 ms of contact changes and
ten smallest sampled gaps. This is dense sampled CAD, not a mathematical
continuous-collision certificate. Conservative AABB pruning matched the original
checker exactly on 103 recorded walking states and separate collision fixtures.

Walking has separately declared checks: unchanged physical model and a passing
transition, completion, tilt ≤8°, requested torque ≤3 Nm, unintended floor force
≤0.01 N, CAD gap ≥1 mm without intersections, three seconds of settled four-tip
support and ≤0.1 rad pose error at zero command, then mean body forward speed
within 0.05 m/s of 0.1 and lateral speed within 0.04 m/s. The roll's command-slew
and 2 rad/s measured-speed limits are **not walking-stage gates**. Peak gait joint
speeds in the three passing development cases are 6.56–7.63 rad/s; these results
must not be described as satisfying the roll speed limit throughout walking or
as hardware readiness.

Inference matches the saved 128 reference cases to **4.255e-6**, below 3e-5.
The adapter follows the runtime's measured-pose ramp (2 s), action fade (2 s),
newest-first four-frame history, fixed desired orientation [0,0,1], negative-Z
projected gravity, unwrapped joint positions and 520/10 Hz scheduling. Training
used 50 Hz control and 250 Hz physics; this uses MuJoCo CPU at 520 Hz, not MJX.
The training model had 3.19072 kg explicit inertial mass; evaluation retains
3.792637 kg including the 0.601917 kg backpack. The explicit `rigid_flush`
matched-state counterfactual retains that backpack and changes contact geometry;
it is not a full recreation of the training environment and also failed CAD.

The fresh sensor/dynamics failures remain failures. Joint/IMU biases, noise and
delay are injected into controller updates. Walking's activation-pose capture
uses ideal joint positions; subsequent observations are perturbed. Thus these
are conditional sensor sensitivity tests, not a complete noisy activation or
hardware sensor model. Synthetic shape changes retain nominal inertia and do
not represent every possible bend or twist.

**Provenance and preservation.** This derivative starts at
`2f4c5c3fbe50a65cb1b6ad12ff6a748e4c584f59`. Candidate parameters and gates were frozen
at `c1d8342bd9b161b1fcfb3061efb12e7ed594cd3e` before the final validation batch.
No post-held-out tuning occurred. Nominal XML, original tip assets, 9 mm gap and
source manifest are byte-unchanged. Model SHA256 is
`c274c1b3ddba73b58c89e8dded10d2d2d8e0fda37af77f631d4bedefa2d177e2`;
walking export SHA256 is
`854ac8ba4ffc305079b7f6f7b52187a211413c3cdb18f0de016dd819ff2450a8`.
The selected checkpoint is step 12,779,520. No new RL policy was trained.

All 397 local optimization trials retain their audits and actual recorded-state
videos in W&B; their missing CAD gates remain incomplete. Online full audits,
failed handoffs and expanded-reach failures are also retained. The interrupted
superseded feedback run has two complete audits and one partial audit; its
corrected count metadata and previous metadata are both preserved. The compact
[evidence archive](evidence/all-audits-and-configs.zip) holds exact audits/configs;
full traces and integration states are in the linked W&B artifacts and local run
directories. Cloud verification covers byte counts/MD5 against local videos and
committed artifact states; the selected video was also downloaded and its SHA256
matched. See the verification JSON files in `evidence/`.

All **31 Python tests pass**. The standalone C++ test could not run because this
PC environment lacks `c++`; historical C++ results are not presented as new tests.
Historical source versions for exploratory controllers are retained. Replaying
an old search requires its recorded source hashes/revision, not silently running
its monkeypatch driver against a newer implementation that already includes it.

Reproduce with the pinned requirements lock and Python 3.12. From the repository
root, set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONPATH=.` and
use fresh output directories:

```sh
python -m pytest motion/inverted_triangle -q
python motion/inverted_triangle/results/roll_corrections_v1/drivers/batch_roll.py \
  --configs motion/inverted_triangle/roll_validation/original_families.json \
  --controller motion/inverted_triangle/roll_validation/selected_controller.json \
  --output runs/roll-original-validation --workers 12
python motion/inverted_triangle/results/roll_corrections_v1/drivers/batch_roll.py \
  --configs motion/inverted_triangle/roll_validation/held_out_shapes_sensors.json \
  --controller motion/inverted_triangle/roll_validation/selected_pipeline.json \
  --output runs/roll-walk-held-out-validation --workers 12 --walking
```

A future hardware executor would need the declared gain schedule, bounded support
adaptation, action wrapper and effective-action history, a verified IMU mounting
transform/xyzw-to-wxyz adapter, and one shared measured model-to-encoder mapping
with preserved unwrapped hub coordinates. It must preserve controller-boundary
state and command history without re-homing or changing encoder zeros. None of
those executor changes has been deployed. Geometry compatibility and failed
sensor/walking validation must be resolved before any hardware proposal.
