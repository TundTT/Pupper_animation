# Rigid formation variation: scope and current failures

User clarification on 2026-09-12: cooled legs are rigid; expect approximately
**10 mm from longest to shortest**, plus slight angle variation. The main mixed
length tests use -5/+5 mm, not +/-10 mm on all legs. One-short-leg testing uses
[-10, 0, 0, 0] mm, which also has a total spread of 10 mm. The +/-5 degree angle
range is an explicit engineering trial assumption, not a user measurement.

This extends the new simultaneous forward roll, not the old sequential flip.
The old 60/60 result does not establish robustness of this different motion.
No robot SSH, deployment, motor activation or training occurs in these probes.

## Geometry and provenance

`formation.py` creates separately identified MJCF variants in memory. It changes
only the distal portion beyond 40 mm in negative local mesh Y of the original
`CustomLegFoot.stl`, using a smooth taper to the nominal tip at 62.694 mm.
Length moves the distal tip along that axis; angle bends the distal shape in the
mesh XY plane. The rest of the attachment is unchanged. This is synthetic shape
sensitivity, not a polymer forming model or a measured scan. The error parameter
is a mesh tip change, not necessarily the same change in world standing height.

The visual mesh, convex floor envelope, nominal foot site and detailed FCL CAD
checker use the same modified geometry. MuJoCo recompiles each model with the
new mesh assets. Axial coordinates are unchanged, preserving the 9 mm additional
outward spacing. Original XML/STL bytes and their manifest stay pinned and
untouched. Reports contain the parent hash, derived XML hash, generated mesh
hashes, exact formation parameters and unchanged-inertia approximation. Mass,
COM and inertia retain their nominal values; redistribution is not modeled.
In-plane angle errors do not cover sideways twisting or all imperfect shapes.

Sources: `model.xml`, `source_manifest.json`, `assets/CustomLegFoot.stl`,
`models/heating_module/{README.md,spec.json}` and the user's clarification above.
See `ROLL_TO_STAND.md` for CAD/backpack provenance and the selected walking export.
MuJoCo's supported XML-and-assets loading is documented in its
[Python bindings](https://mujoco.readthedocs.io/en/stable/python.html).

## What the first diagnostics found

Seven fixed development cases are in `formation_cases.json`: nominal, front/rear
length difference, right/left difference, diagonal difference, one front leg
10 mm short, angle-only +/-5 degrees, and combined lengths with +/-3 degrees.
All finished the 19-second baseline rollout without the motor/body floor force
gate failing. Only nominal, side-to-side length difference and angle-only passed
all non-CAD gates. Front/rear difference failed sustained four-tip support;
diagonal, one-short-leg and combined cases failed additional endpoint checks.

The one-short-leg case finished nearly level (0.193 degrees) while its shortened
front tip was 10.31 mm above the floor and unloaded. This demonstrates why an
IMU-level condition alone cannot certify standing on four tips.

Three exploratory settling variants were also tried locally; none solved the
seven-case set. These are rejected development trials, not validated controls.
The retained `settling_feedback.py` is the second variant, disabled by default.
It uses nominal kinematics, ideal joint position/velocity and an ideal IMU, with
a gravity-corrected PD torque estimate. It receives neither the real variant's
lengths nor simulator contact forces. The estimate is not a foot-force sensor.
Its per-joint correction cap is 0.08 rad; command speed/acceleration and the
existing endpoint gates remain enforced. The first variant's target tracking
oscillated; the second uses gradual error tracking. The third increased reach
rate and subtracted a common extension, but also failed. Longer holds did not
establish success. Do not deploy any of them.

Unlogged short diagnostics explicitly used `--wandb disabled`, no video or CAD.
Their `sampled_cad: null` means incomplete validation even when other gates pass.
They are preserved along with source hashes and failures. A separate online
baseline rerun supplies actual videos and sampled CAD; see its result report.
The 0.1-second CAD sampling is screening, not continuous collision certification.

## Reproduction

Use the pinned dependencies. From the repository root:

```shell
python -m pytest motion/inverted_triangle/test_formation.py motion/inverted_triangle/test_roll_to_stand.py -q
python -m motion.inverted_triangle.roll_to_stand --config motion/inverted_triangle/formation_cases.json --output runs/formation-baseline-new --cad --video --wandb online
```

Output directories must be new. Geometry variants are opt-in; a run with no
`formation` retains the original nominal path. `final_state.json` records the
actual integration state and last command for continuation with the same model
and dynamics. The sampled video states alone are insufficient for policy handoff.

## Next implementation target

Keep the coordinated roll as the nominal motion. Optimize a small feedback
correction around it across independently varied rigid shapes. First determine
whether useful corrections can satisfy the existing 0.1 rad walking-pose error
gate, four supporting tips, settling, torque, tilt and collision requirements.
If that pose tolerance is physically incompatible with a geometry case, report
the evidence and required departure; do not silently loosen a gate to get a pass.

Actual contact forces, geometry errors and simulator base position may be used
for optimization/reward/audit, never secretly as hardware controller inputs.
An actor/controller should use only supported IMU/joint feedback and command
history. If a small conventional correction cannot handle partial support, the
PC can train a bounded residual policy with these geometry variations. Its
purpose is support adaptation, not replacement of the clean nominal motion.
Test actual walking-policy entry under the same imperfect geometry. Pose
closeness alone does not validate handoff, and nominal-policy robustness to these
shape errors is currently unknown.
