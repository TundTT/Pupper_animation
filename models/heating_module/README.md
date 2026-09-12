# Approved heating backpack

Placement revision 3 was visually approved by the user on September 12, 2026.
The long plate runs across the robot; the lower bracket remains on the strip
marked in the user's top-view drawing. The fixed body pose in `base_link` is
`pos="-0.036349545 0.000031565 0.032506671263"`,
`quat="0.5 -0.5 -0.5 0.5"` (wxyz).

`spec.json` is the editable source. `Heating_module.stl` is the supplied mesh,
unchanged, with SHA-256
`40d4b45c2943272e69801393df76551f3829fc41205d79f156bd49957b84a9b1`.
`apply.py` supplies both an idempotent XML editor and `add_to_tree` for generators.
`gap.py` applies or recognizes the additional 9 mm hub-axis spacing. An explicit
`quadmorph_additional_hub_gap_m` numeric value prevents applying the gap twice.
Changing the fixed body's pose moves the visual mesh, COM, full inertia tensor
and three box contact proxies together. There is no added joint or actuator.

## Sources

- User-supplied `Heating module.stl` in Downloads.
- CAD screenshot `codex-clipboard-33290f0e-6ddb-42bf-b764-6944bb699341.png`:
  mass 1.327 lb = 0.60191707499 kg; COM `[0.12, -1.72, 0.248]` inches.
  The complete centroidal inertia matrix, including cross terms, is in `spec.json`.
- Side and top drawings listed in `spec.json`, followed by the user's explicit
  approval of revision 3. The mounting position is visually approved, rather than
  a precisely measured mounting coordinate.
- Millimeter STL scale and a shared CAD frame are inferred from the mesh volume
  and COM comparison. The user has not separately confirmed the export frame.

Explicit mass/inertia avoid assuming a watertight, uniform-density STL. Zero
geom density prevents double counting. Added spacer mass remains unspecified.

The spacer definition follows the user's confirmed outward translation of the
whole wheel/shin assembly and the original spaced model commits
`88f6a8888644c8072df36d4878d77684972c9231` (wheel) and
`27bc66823478758bd9dc701a5aec27cceaa0710a` (leg). Visuals, collision geometry,
foot sites and COM translate along each terminal hub body's positive local Z.
Joints, joint frames, mass and inertia about the translated COM stay unchanged.
For the QuadMorph meshes, mounting Z is 22.6 mm (13.6 + 9); wheel-center Z is
39.35 mm (30.35 + 9). The printed spacer solid is not modeled.

## Configurations

The backpack is present in these new local development checkouts. Original
training/review checkouts and their recorded results remain unchanged.

| Checkout / branch | Active model |
| --- | --- |
| `Pupper_backpack_wheel` / `codex/backpack-wheel` | `Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml` |
| `Pupper_backpack_leg` / `codex/backpack-leg` | Same relative path, using the spaced leg geometry |
| `Pupper_backpack_align` / `codex/backpack-align` | `training/wheel_align/model.xml`, used by deterministic lift/alignment |
| `Pupper_backpack_triangle` / `codex/backpack-triangle` | `motion/inverted_triangle/model.xml`, shared by point-up and subsequent flips |

The current `robot-code` checkout also has the backpack in all 18
`pupper_v3_complete*.xml` ROS simulation variants. Other example XMLs, source
exports, third-party models and archived experiment outputs are not active
configuration entrypoints and were not rewritten.
These ROS variants retain their legacy limb shapes, now translated outward by
9 mm; they are not replacements for the wheel/leg/triangle CAD in the dedicated
configuration branches. All active models listed above include the same gap.

The wheel/leg and ROS composition scripts insert the module after their normal
geometry processing. The triangle generator inserts it after the original CAD
floor-envelope processing and records its source hashes in `source_manifest.json`.
Triangle contact proxies use the existing floor-only collision convention;
the detailed backpack mesh is automatically included in its independent shin
self-clearance audit. Alignment's encoder clearance estimator still represents
the original body box in this model-only commit. Its Python and C++ wheel-center
offsets are now generated from the XML, including the 9 mm gap. The separate
paused alignment task's actuator/controller changes remain in the working tree,
outside this commit; no new controller behavior is certified here.

Each development checkout contains an identical copy of this module so it remains
self-contained when committed or copied to another computer. To change the design,
edit one specification and distribute it with `apply.py` and the STL, then reapply
the XML editor or rerun the triangle builder. Do not substitute changed model hashes
into old policies/candidates to make their provenance checks pass.

The laptop checkouts are sparse to fit available disk space; only relevant source
and model assets are materialized. On a machine with adequate storage, run
`git sparse-checkout disable` in the chosen checkout to restore its other tracked
files. No training, hardware deployment or robot motion was performed here.

## Validation

All 22 updated models load in MuJoCo 3.3.7 and gain exactly 0.60191707499 kg.
Checks compare existing body transforms/inertias, joint frames/limits, actuator
parameters and saved keyframes against the source models. Only terminal-limb
COMs/visuals/collision geoms/sites move where the 9 mm gap was previously absent;
joint frames, actuator parameters, masses and centroidal inertias stay unchanged.
The module's full tensor is checked after compilation and forward quantities are
finite. These checks do not establish locomotion or flip performance with the
additional weight. Existing motion results must be reevaluated with these models.

The portable records are [model validation](validation.json) and
[geometry validation](spacing-geometry-validation.json). Python and both C++
geometry implementations match MuJoCo wheel frames over 300 sampled poses;
this is a geometry check, not a dynamic acceptance test.

Run `python models/heating_module/check.py` with MuJoCo 3.3.7 and numpy to check
the active model(s) in a checkout. For a particular XML, pass its path explicitly.

## Review images

Each sheet shows an angled, side, top and end view of the actual XML geometry.
The camera presentation grounds the static pose without changing XML keyframes.

- [Wheel](review/wheel.png)
- [Leg](review/leg.png)
- [Alignment](review/align.png)
- [Point-up triangle / flip](review/triangle.png)
