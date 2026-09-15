# Heating backpack — placement review, revision 3

**Update: revision 3 is approved and propagated.** See the current
[shared module documentation](../../models/heating_module/README.md) and
[22-model validation record](propagation-validation.json). The design-history
notes below describe the review stage; their pending-propagation wording is historical.

One candidate XML has been created for visual review. No training model, robot
configuration, policy, original training checkout or hardware installation was
changed. Propagation is deliberately pending the user's placement review.

## Open the result

- [Four views](placement-review.png)
- [Candidate XML](wheel_gap9_backpack_preview.xml)
- [Editable module values and placement](module_spec.json)
- [Model checks](validation.json)

The orange geometry is the actual supplied STL. The wheel robot retains the
9 mm outward hub-assembly gap. Pictures are static model views, not a rollout.
The renderer translates the floating base down to place the wheels on the floor;
the XML's original home keyframe is unchanged.

## Current placement proposal

The user's side sketch reverses the first proposal: the long slotted plate is
above the battery/electronics. The subsequent top-view markup locates the lower
attachment on a transverse strip near base x=-29 mm, y=0. The lower bracket is
centered there, with its lowest surface at the nearby deck's 32.5 mm base-frame
height. The x coordinate is estimated from the markup; the height is inferred
from the actual adjacent CAD surface. The original STL is not resized to match
the hand-drawn silhouette. This is a provisional visual placement, not measured
mounting coordinates or a CAD-interference certification.

The latest user top-view drawing requests a 90 degree rotation, making the long
plate span across the robot. Revision 3 rotates about the existing lower-bracket
center, preserving the attachment point and height above. The CAD axes now map
as follows: +X to robot +Y, +Y to robot -Z, and +Z to robot -X. The fixed-body
quaternion (wxyz) is `[0.5, -0.5, -0.5, 0.5]`.
Changing `heating_module`'s `pos` and `quat` moves its visual geometry,
collision proxies, center of mass and inertia together. It adds no joint.

The specification is the editable source for this preview. Re-running
`build_preview.py` regenerates the XML from the pinned wheel model and that file.
Direct XML edits can instead be reviewed manually, but will be overwritten by
the generator unless incorporated into the specification.

## Mass, frame and shape provenance

User-provided sources:

- [Heating module STL](C:/Users/tundt/Downloads/Heating%20module.stl)
- [CAD mass-properties screenshot](C:/Users/tundt/AppData/Local/Temp/codex-clipboard-33290f0e-6ddb-42bf-b764-6944bb699341.png)

The screenshot supplies 1.327 lb = **0.60191707499 kg**, CAD center of mass
`[0.12, -1.72, 0.248] in` = `[0.003048, -0.043688, 0.0062992] m`, and this
centroidal tensor in lb in^2:

```text
1.446  0.016  0.008
0.016  3.789  0.097
0.008  0.097  4.279
```

The complete tensor, including cross terms, is converted by multiplying by
`0.45359237 * 0.0254^2`. The screenshot has no mate connector selected;
[Onshape's documentation](https://cad.onshape.com/help/Content/View/mass_properties_tool.htm)
defines the reported inertia about the selected parts' center of gravity in this
case. The inertia is entered at the supplied COM, not at the mesh origin. MuJoCo
then carries the fixed body's translated and rotated inertia into the robot.

The STL spans approximately 174.14 x 80.95 x 63.50 raw units. Treating these as mm
gives a mesh volume of about 513091 mm^3, consistent with the screenshot's
31.311 in^3. This is strong evidence for millimeters. Shared export/CAD axes and
origin are still awaiting operator confirmation. The mesh's volume centroid is
not substituted for the supplied density-weighted COM. The STL is not fully
watertight; explicit mass/inertia avoid relying on a uniform-density mesh estimate.

The original model's 3.218 kg remains in place; the candidate total is 3.819917 kg.
Visual and collision geoms have zero density so the module is counted exactly
once through its explicit inertial element. Three conservative box proxies cover
the mounting plate, electronics and end bracket for contact. These are simulation
approximations, distinct from the unmodified visual STL.

## Newer repository work inspected

- The 9 mm gap shifts all four shin/wheel visuals, collision shapes, foot sites
  and COMs while preserving hub frames. The leg and wheel source commits are
  `27bc66823478758bd9dc701a5aec27cceaa0710a` and
  `88f6a8888644c8072df36d4878d77684972c9231` respectively. This preview uses the
  latter's `Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml`.
- Alignment development now includes the deterministic coordinated keyframe
  controller, velocity/acceleration bounds, measured-state rotation gates and
  lowering before a changed request. See
  [KEYFRAME_ALIGNMENT.md](C:/Users/tundt/Desktop/Pupper_alignment_keyframes/KEYFRAME_ALIGNMENT.md)
  at `05299b3a61bf45bb5e89a371fa3313643be2014e`. The current robot code has further
  hub position-PD and bounded integral-torque changes; old v5 notes are historical.
- Startup home is now shared across the live encoder session. The repaired
  hanging reference is retained in
  [start_pose.json](../../hardware_testing/start_pose/start_pose.json), with the
  workflow in [STARTUP_CALIBRATION.md](../../STARTUP_CALIBRATION.md). This task
  does not start or recalibrate the robot.
- Inverted-triangle development now has dedicated contact geometry, motion
  search and continuous four-flip audit records. Preserved robustness failures
  remain in the later review branches; a previously successful sequence is not
  automatically validated with this additional backpack mass.

## Checks and later propagation

MuJoCo 3.3.7 loads the candidate, its inertia is positive and physically
consistent, its mass increment matches the screenshot, and it retains 19 qpos,
18 velocities and 12 actuators. Checks preserve existing body properties, joint
frames/limits, actuator properties and keyframes. Reconstructed centroidal
inertia differs from the entered tensor by less than 7.3e-11 kg m^2, consistent
with numerical eigensolver tolerance. Forward-dynamics quantities are finite.
There has been no policy or flip-sequence rollout with the backpack.

After placement approval, apply the same fixed module and asset to the active
leg, wheel, alignment and point-up/flip model builders. Preserve historical
training source snapshots and regenerate dependent model hashes/provenance in
new experiment versions. Triangle geometry generators must be updated too, so
regeneration does not discard the backpack. Each model must gain the same
0.601917 kg exactly once; meshes alone are insufficient.

Reproduce from the repository root:

```powershell
$backpackPython = Join-Path $env:TEMP 'quadmorph-backpack-20260912/Scripts/python.exe'
& $backpackPython output/heating-module-preview/build_preview.py
& $backpackPython output/heating-module-preview/render_preview.py
```

The isolated rendering environment contains MuJoCo 3.3.7, numpy and Pillow;
trimesh/scipy were used for source geometry inspection. It is not a training
environment and changes no robot dependencies.
