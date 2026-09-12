# Axial spacer clearance study

11 September 2026. Offline geometry only. No hardware motion or tracked model edits.

## Result

**The existing 16 mm spacer is sufficient in the supplied CAD for clearance between the original-length shin and its own upper motor assembly.** Approximately **5.7 mm additional outward offset** is the near-contact threshold across the checked hub rotations and proximal poses. That threshold has essentially no allowance for manufacturing error or material deflection. The 16 mm spacer leaves a measured minimum of **9.159 mm** including the sampled intermediate lift poses. The 20 mm spacer leaves approximately **12.961 mm** at the checked neutral/apex poses.

For the front-right neutral tip-up pose alone, near-contact occurs at about 3.315 mm added offset. This is insufficient to clear the entire turn. The closest part of the full turn occurs away from exactly point-up.

The eight neutral/apex configurations need approximately 9.25 mm spacer thickness for 3 mm CAD clearance, or 11.50 mm for 5 mm clearance. These margins are geometric design choices, not experimentally established physical requirements. Of the user's two existing spacers, 16 mm provides ample modeled clearance for the specified interference; the analysis gives no clearance-based reason to require 20 mm.

## Definition and sources

The user explicitly confirmed that the spacers translate the entire wheel/shin assembly outward along the hub rotation axis. No shortening is included. Added thickness means **additional to** the current mounting offset, not replacement of it. The actual printed spacer solid and fasteners were not provided or modeled.

- User-designated `leg` branch geometry, pinned at `6b55e30ff224193a2a21c8dc3dbb88e029a417c8`.
- [Original XML](../point-up-clearance/source/description/mujoco_xml/pupper_v3_complete.mjx.position.xml): asset scale at line 88; front-right upper motor visual placement at lines 99–106; hub body and shin visual placement at lines 115–135. The shin visual starts at `pos="0 0 0.0136"`; an added 16 mm spacer therefore places this visual at `pos="0 0 0.0296"`, retaining all original mesh dimensions. This describes the visual shift only, not a complete dynamic-model edit.
- [Original shin CAD](../point-up-clearance/source/description/meshes/stl/CustomLegFoot.stl): tip-direction extent 62.694 mm from the hub axis; original radial envelope, outer ring and three segments retained.
- [Front-right upper assembly CAD](../point-up-clearance/source/description/meshes/stl/LegAssemblyForFlangedv26_001.stl), plus the corresponding mirrored/rear assembly meshes from the same XML. The complete upper assembly visual surfaces were checked, including mounting details.
- Existing alignment apex coordinates from [wheel_align_reference_data.hpp](https://github.com/TundTT/Pupper_animation/blob/e3e1d9737c51f904ff7d0f4cae5692ac4703f659/ros2_ws/src/neural_controller/include/neural_controller/wheel_align_reference_data.hpp#L22), pinned at robot-code commit `e3e1d9737c51f904ff7d0f4cae5692ac4703f659`. These are implementation reference poses, not proof of a stable transformed-limb trajectory.

## Measurement method and scope

MuJoCo forward kinematics supplies the source joint frames and references. Python-FCL measures intersection and Euclidean separation between actual nonconvex CAD triangle meshes. Training contact capsules and convex hulls are not substituted for the visual CAD surfaces. Positive local Z of each hub body is the original mounting direction; an added axial translation is applied to the entire shin mesh while retaining the hub joint location.

All four limbs were swept through a complete hub revolution in both neutral and existing apex proximal configurations. This covers both point-up-to-point-down half-turn choices. Neutral uses proximal coordinates `[1,0]` for right limbs and `[-1,0]` for left limbs, respecting the source XML's reference angles. Point-up uses the source hub reference plus pi. At lifted proximal poses this follows the limb frame rather than enforcing a vertical world-space tip.

Rotation sweeps use 2-degree samples with scalar refinement of every sampled local minimum. Thickness thresholds target 0.01 mm positive separation as a numerical approximation to tangency; the results were rechecked at 0.5-degree spacing. Intermediate checks interpolate proximal coordinates at 10% increments between neutral and apex, sweeping the entire hub revolution at each increment, for 5.7 and 16 mm spacers. The 5.7 mm spacer cleared all sampled intermediate poses, with only about 0.02 mm at the closest one. Reusable FCL rigid transforms were independently cross-checked against directly transformed world-coordinate meshes for all four legs, agreeing within 1e-7 mm.

This is a numerical geometric search, not a formal continuous collision proof or hardware test. The result concerns the shin versus its own upper motor assembly. It does not establish full-body clearance, spacer strength, stable ground support, or the path of hot flexible polymer. The original model needs to match the physical motor, mounts and shin for these dimensions to apply.

## Artifacts

- [CAD comparison](spacer-comparison.png): original versus 16 mm outward offset; the printed spacer itself is not rendered.
- [Measurements and thresholds](measurements.json)
- [Intermediate-pose checks and transform validation](validation.json)
- [Rotation samples](rotation-clearance.csv)
- [Analysis](analyze.py), [validation](validate.py), [rendering](render.py)

The scripts reuse the pinned source loader and geometry utilities in `../point-up-clearance`. Run with the isolated Python environment at `%TEMP%/quadmorph-clearance-20260911/Scripts/python.exe`. Packages used: MuJoCo 3.3.7 and python-fcl 0.7.0.11. No controller changes or hardware startup were performed.
