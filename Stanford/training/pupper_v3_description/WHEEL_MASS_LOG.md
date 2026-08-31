# Wheel mass measurement log

Tracks real scale measurements of the wheel assembly's parts against CAD-estimated
mass, so we can correct the URDF `<inertial>` block once enough samples are in.

## Decision context

- Final wheel is modeled as **one URDF link** (all sub-parts merged into a single rigid
  body before export — confirmed no relative motion between any of them once
  assembled).
- The assembly is **multi-material** (e.g. this "polymer wheel" piece vs. the rigid
  printed hub/spokes) — each part is measured on the real scale individually.
- **No CAD-estimated mass is used.** Workflow per part: measure N real samples on the
  scale, take the mean, and enter that mean directly as a **mass override** in CAD
  (not a density override) for that part's body. CAD still computes CoM and the
  inertia tensor from the real geometry, just scaled to the real measured mass instead
  of an assumed-density estimate.
- Real mass varies sample-to-sample (manufacturing/material process variability) —
  hence multiple measurements per part; std dev is tracked to gauge how tight/loose
  that variability is.
- Final assembly `<inertial>` = sum of all per-part CAD outputs (each already using
  its own measured-mass override).

## Parts in the wheel assembly

| Part name | Material | Qty per wheel | Total qty (4 wheels) | Status |
|---|---|---|---|---|
| Polymer wheel | smart polymer (TBD — confirm exact material/spec) | 3 | 12 | measured (n=10) |
| Mounting bracket — piece A (leg adapter/spider, 3 posts) | TBD (rigid printed) | 1 (assumed) | 4 (assumed) | measured (n=1) |
| Mounting bracket — piece B (hub plate, 4 holes) | TBD (rigid printed) | 1 (assumed) | 4 (assumed) | measured (n=1) |
| Outer ring (treaded) | TBD | 1 | 4 | measured (n=1) |
| *(others TBD as they come up)* | | | | |

**Flag:** qty-per-wheel for the mounting bracket pieces is assumed to be 1 each (unlike
the polymer wheel's 3-per-wheel) — confirm this is right before finalizing totals.

## Measured samples

| # | Part | Measured mass (g) | Date | Notes |
|---|---|---|---|---|
| 1 | Polymer wheel | 6.38 | 2026-08-30 | |
| 2 | Polymer wheel | 6.59 | 2026-08-30 | |
| 3 | Polymer wheel | 5.79 | 2026-08-30 | |
| 4 | Polymer wheel | 5.60 | 2026-08-30 | |
| 5 | Polymer wheel | 6.58 | 2026-08-30 | |
| 6 | Polymer wheel | 5.85 | 2026-08-30 | |
| 7 | Polymer wheel | 6.53 | 2026-08-30 | |
| 8 | Polymer wheel | 6.41 | 2026-08-30 | |
| 9 | Polymer wheel | 6.51 | 2026-08-30 | |
| 10 | Polymer wheel | 5.74 | 2026-08-30 | |

## Rollup per part

| Part | Sample count | Mean mass (g) | Std dev (g) | Min / Max (g) | Mass override to use in CAD |
|---|---|---|---|---|---|
| Polymer wheel | 10 | 6.198 | 0.400 | 5.60 / 6.59 | **6.20 g** |

## CAD mass properties (from Onshape, mass override = 6.2 g)

Quantity needed: 12 total (3 identical rings per wheel × 4 wheels). Numbers below are
**per single ring** — multiply/sum per instance when composing the full wheel-link
inertial block (Onshape sums this automatically if done at the assembly level with 3
part instances, each carrying this same override).

| Part | Mass (g) | CoM x/y/z (mm) | Lxx (g·mm²) | Lyy (g·mm²) | Lzz (g·mm²) | Lxy/Lxz/Lyz |
|---|---|---|---|---|---|---|
| Polymer wheel | 6.2 | 0, -0.208, 10 | 1464.652 | 1442.689 | 2494.007 | 0 / 0 / 0 |

Note: mass override used here (6.2 g) is the rounded mean of the 10 samples above
(6.198 g) — close enough for now; revisit if more samples shift the mean noticeably.

| Part | Mass (g) | CoM x/y/z (mm) | Lxx | Lyy | Lzz | Lxy | Lxz | Lyz |
|---|---|---|---|---|---|---|---|---|
| Mounting bracket — piece A | 10.62 | -0.009, 0.004, 17.348 | 1611.204 | 1610.992 | 1665.708 | -0.114 | -1.788 | 0.78 |
| Mounting bracket — piece B | 3.98 | 0.093, -0.376, 24.96 | 346.749 | 364.145 | 688.219 | -4.569 | -0.633 | 2.565 |
| Outer ring | 22.31 | 0, 0, 12 | 25081.749 | 25081.749 | 48030.806 | 0 | 0 | 0 |

(all inertia values g·mm², about each part's own CoM — current CAD state only, no
revision history kept; only 1 sample each so far, not yet averaged over multiple
prints like the polymer wheel — revisit once more units are printed and weighed.)

## Whole assembly (final, all parts combined)

Reference frame: mate connector on the wheel's bore centerline at the mounting face
(where it attaches to the leg's motor shaft) — matches the URDF joint-frame
convention. Radial coordinates (X, Z) are ≈0, confirming the wheel is radially
balanced; the Y offset is purely axial (CoM sits mid-depth in the wheel's thickness,
away from the mounting-face origin) — expected, not an imbalance.

| Mass (g) | CoM x/y/z (mm) | Lxx | Lyy | Lzz | Lxy | Lxz | Lyz |
|---|---|---|---|---|---|---|---|
| 54.56 (CAD sum, modeled parts only — no fasteners) | 0.016, -16.38, -0.019 | 37657.229 | 68462.575 | 37662.585 | -10.428 | -8.258 | 12.217 |

### Real assembled weight (scale, with screws/fasteners) — supersedes CAD sum

Full physical assembly weighed **62 g**, ~7.44 g heavier than the CAD sum (fasteners
aren't modeled as separate parts). Real measurement wins per this log's approach — use
62 g as the link mass. Inertia tensor is approximated by scaling the CAD tensor by
`62 / 54.56 ≈ 1.1364` (assumes fastener mass is distributed similarly to the rest of
the part — an approximation, since screws are likely concentrated near the bolt
circle at a larger radius than average, so the true spin-axis inertia could be
somewhat higher than this scaling gives).

| Mass (g) | CoM x/y/z (mm) | Lxx | Lyy | Lzz | Lxy | Lxz | Lyz |
|---|---|---|---|---|---|---|---|
| 62.0 | 0.016, -16.38, -0.019 | 42792.306 | 77798.381 | 42798.392 | -11.850 | -9.384 | 13.883 |

**This is the final number set to use for the wheel URDF link's `<inertial>` block:**

```xml
<inertial>
  <origin xyz="1.6e-05 -0.01638 -1.9e-05" rpy="0 0 0"/>
  <mass value="0.062"/>
  <inertia ixx="4.279231e-05" ixy="-1.185e-08" ixz="-9.384091e-09"
           iyy="7.779838e-05" iyz="1.388295e-08" izz="4.279839e-05"/>
</inertial>
```
