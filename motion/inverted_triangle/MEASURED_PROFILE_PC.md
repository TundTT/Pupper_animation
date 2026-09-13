# Validate measured cold geometry

Continue from `codex/measured-cold-geometry-20260913`, based on PC resolution
`d67ad42`. Read `physical_measurements_20260913.md` and `measured_dimensions.json`.
The operator confirmed actual cooled, rigid, point-up limbs and measured their
hub-to-top/base extents. This supersedes treating the old fully compressed shape
as the only representative physical cold geometry. Preserve the old model and
its failures as historical evidence.

The provisional 57 mm top / 35 mm bottom fit predicts +7.648 mm initial housing
clearance at the saved simulated endpoint, instead of -1.898 mm. The operator's
rough 12.7 mm floor gap was photographed at a different, not model-calibrated
measurement pose; do not force those two readings to match by shifting root Z.

Two quick native rollouts used clean source `113c6d5`:

- Uniform measured baseline: all executed dynamic/standing gates pass, zero
  motor/body floor load, final maximum joint error 0.0642 rad.
- Measured pairs assigned in sample order to FR/FL/BR/BL as an explicitly
  hypothetical trial: same executed gates pass, zero motor/body floor load,
  final maximum joint error 0.0781 rad. The user did not identify anatomical legs.

Both OMITTED dense self-CAD auditing and therefore have engineering_pass=false
and acceptance_complete=false. This is not hardware or walking acceptance.
Videos/integration artifacts were uploaded and verified in W&B:

- [Uniform measured baseline](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/202de0748cdf44ff)
- [Measured pairs in one hypothetical arrangement](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/f8ef53bbe1844734)

40 focused model/formation/entry/roll checks pass, including fitting all four
measured extent pairs, preserving the mounting region/width/9 mm axial gap, and
exact dynamic snapshot continuation. A standalone generated XML with its assets
also loads. The fit preserves nominal inertia and only changes Y coordinates;
it is not a scan, a full contour reconstruction, or a validated material model.
No measured angular fit is provided; nonzero measured-profile bends currently
raise an explicit error. Implement any angle perturbations as labelled synthetic
tests and preserve their geometry/provenance consistently.

## Next run

1. Fetch the now-published historical alignment source:
   `codex/alignment-source-a7e3bb1` at
   `a7e3bb1e67f73b4fb14ee6658d66d0263fd2bfd8`. The prior unavailable-source
   blocker has been resolved. Use a separate checkout/environment to replay it;
   retain fresh successful endpoints and reject failures as before. Do not
   substitute the different published backpack controller or zero policy residuals.
2. Run `measured_profile_plan.json`: the uniform baseline plus all 24 anatomical
   assignments of the four measured pairs. Preserve each top/base pair. Use the
   full CAD audit; the included saved endpoint is only the first stage:

```bash
MUJOCO_GL=egl python -m motion.inverted_triangle.handoff_batch \
  --plan motion/inverted_triangle/measured_profile_plan.json \
  --terminal motion/inverted_triangle/results/handoff_local_20260913/alignment-terminal-0.json \
  --output /ABS/NEW-RESULTS/measured-profiles --cad
```

3. Validate against fresh successful alignment endpoints, then held-out coupled
   formation, ruler uncertainty (sample 2 top 57–58; sample 3 base 31–32), friction,
   sensors and dynamics. Keep the earlier approximately 10 mm between-limb
   robustness target as an extended test; four samples are not the full population.
4. Compare 2/4/8 contact parts for the same fitted mesh; retain mass/COM/inertia
   sensitivity and all existing force, tilt, speed, support and CAD gates. Check
   front-to-rear limb, motor, body and backpack collisions on integrated states.
5. Only after standing acceptance, test the frozen walking policy's actual
   activation/observation/history/fade and continuous-coordinate handoff. Its
   original training geometry has not been changed. Final joint-pose similarity
   does not validate its behavior on these fitted shapes.
6. Publish W&B Media videos, artifacts and failed audits; commit/push an honest
   matrix and return one continuous rollout. Hardware-interface parity and a
   reviewed robot adapter remain required before a physical motion test.

No robot, motor, heating, or calibration actions. The robot's measurement hold
is not an upstream controller or a walking calibration to import into simulation.
