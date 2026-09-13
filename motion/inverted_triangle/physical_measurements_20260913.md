# Cooled limb observation — September 13, 2026

Source: the operator's messages and seven attached photographs in the lab session.
These are physical observations, not simulation results or dimensions fitted from
perspective photographs.

- Operator explicitly confirms that the photographed limbs are cooled, deformed,
  rigid, with the formed point facing upward. Do not relabel these as unformed
  wheels because their outer tread remains rounded.
- Operator reports approximately 1/2 inch (12.7 mm) of clearance beneath the
  lower motor housing in the photographed supported-on-feet posture. Photos show
  positive clearance beneath pictured housings. This is not an independently
  calibrated measurement of every limb or a minimum over the rolling trajectory.
- The previously confirmed additional 9 mm axial mounting gap is unchanged.
- The later three photographs show the limb face and upward point with the torso
  supported on a stand; these demonstrate shape/orientation, not loaded floor
  clearance. Use the earlier tape-measure photographs for floor-clearance context.

The nominal source mesh remains `assets/CustomLegFoot.stl`, SHA256
`5d46e3d9b53c579b81c743f5a23f9e03d13582443e1cf94db4dc2bd06a2615ae`.
Its mesh coordinates are millimetres. For the point-up local profile, the nominal
mesh extends 62.694 mm toward the point and 25.447 mm toward the base from the
hub axis, with 97.608 mm total width. These are mesh-plane dimensions, not world
floor clearance. `model.xml` places this mesh with an axial translation only on
the right limbs; left limbs have the corresponding half-turn orientation.

The simulation's negative housing clearance applies to its specified mesh and
saved simulated joint posture. The physical photographs do not establish the
same physical joint posture or exact dimensions. They therefore establish that
the simulated failing start is not yet a validated representation of this lab
configuration. They do not prove that all modeled dimensions are wrong, that the
entire cold-start family is infeasible, or that the motion is hardware-ready.

The measurement session used a current-angle hold, not a marked-ring home
calibration. Startup encoder offsets were assigned in the operator-arranged
measurement pose. Do NOT map that session's numerical joint readings directly
into the nominal model or use them as walking-policy calibration.

Next identification measurements: hub-axis-to-lowest-tread and
hub-axis-to-highest-point distances, taken in the limb plane. Compare the physical
profile with the original CAD before fitting a new baseline. Retain the 10 mm
between-limb length-variation requirement around an appropriate measured family.
Do not simply raise root Z, remove motor contacts, or declare the old simulated
contact failures fixed. Unequal-tip support remains an independent open problem.

Attachment groups in the local task:
- `3449A885-85EC-4943-A926-0659124390A9`: four initial floor/tape photos.
- `9DA443C0-68F8-4FA4-977E-F248693C210C`: three confirming point-up face photos.
Images remain in the task attachments. No dimensions were inferred by treating
perspective pixel distances as an orthographic scale.
