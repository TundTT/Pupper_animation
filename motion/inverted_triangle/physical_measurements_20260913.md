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

## Subsequent operator ruler readings

| Measurement order (not anatomical leg identity) | Hub to top, mm | Hub to bottom, mm | Overall height, mm |
|---|---:|---:|---:|
| 1 | 55 | 37 | 92 |
| 2 | 57–58 | 34 | 91–92 |
| 3 | 60 | 32 (after verbal 31–32) | 92 |
| 4 | 56 | 37 | 93 |

The third bottom value was corrected from 28, with a final reading of 32. The
fourth repeated 56 was followed by a final bottom reading of 37. Preserve the
raw reported ranges in `measured_dimensions.json`; the readings are approximate.
Using 57.5 for sample 2, the means are 57.125 above and 35 below. A provisional
uniform profile uses 57/35. Preserve each top/bottom pair when varying limbs.
The observed top and base spans are both about 5 mm, while overall height stays
within roughly 2 mm. The previous 10 mm robustness requirement remains useful;
four samples do not define all possible future formations.

`measured_profile.py` smoothly fits the original mesh's Y coordinates to those
extents, keeping the attachment region, X width and Z mounting coordinates fixed.
This is a provisional geometry fit, not reconstruction of the full contour or
validation of inertia. It does not modify the historical nominal mesh/XML or the
walking-policy model. The same fitted mesh is used for rendering and detailed
CAD audits; floor contact still uses the explicitly configured convex parts.
