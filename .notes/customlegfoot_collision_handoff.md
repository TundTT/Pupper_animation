# CustomLegFoot collision geometry — handoff notes

## Repo / branch
`Pupper_animation` (github.com/TundTT/Pupper_animation), branch `leg`.
Model file: `Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml`

## Status: orientation fix is DONE and pushed, collision-shape work is NOT started (nothing landed)

Commit `6519657` on `origin/leg` ("Fix CustomLegFoot: l-side mount was showing the mesh's back face, plus hub
clocking") is the last real change. It fixed the l-side foot mount's rotation (was `quat="0 1 0 0"`, now
identity + an extra `quat="0 0 0 1"` for hub clocking) and copied/rotated the inertial values accordingly.
That work is confirmed correct by the user and already pushed — do not redo it.

**The task described below (cylinder collision geom) has NOT been added to the XML.** All experiments were
done as scratch edits and reverted; `git diff` against HEAD is currently clean. Nothing here should be
assumed to exist in the file — read the current XML before continuing.

## The task

User request: "would it be possible to make the collision a cylinder that looks something like the red
outline in my attached image" — an annotated screenshot circling one of the wheel-shaped `CustomLegFoot`
mesh's radiating spoke/paddle loops (the outer structure, roughly a "petal" shape spanning from near the
hub out toward the rim), asking for it to be approximated as a `type="cylinder"` collision primitive,
replacing (or supplementing) the current mesh-convex-hull collision geom:

```xml
<geom pos="0 0 0.0136" type="mesh" mesh="CustomLegFoot" class="collision" />
```

This exists identically (mod the l-side `quat="0 0 0 1"`) on all four legs: `leg_front_r_3`, `leg_front_l_3`,
`leg_back_r_3`, `leg_back_l_3`.

## Key background facts (established and trustworthy)

1. **MuJoCo bakes mesh recentering into compiled geom_pos/geom_quat.** `m.geom_pos[gid]` /
   `m.geom_quat[gid]` (post-compile) are NOT the raw XML-authored `pos`/`quat` — they're composed with the
   mesh asset's own intrinsic recentering transform (`m.mesh_pos[mesh_id]`, `m.mesh_quat[mesh_id]`), i.e.:
   - `geom_pos_compiled = xml_pos + R(xml_quat) @ mesh_pos`
   - `geom_quat_compiled = xml_quat ⊗ mesh_quat` (as rotation composition: `R_compiled = R(xml_quat) @ R(mesh_quat)`)
   This was the source of a lot of early confusion — don't overwrite `m.geom_quat`/`m.geom_pos` directly on a
   loaded model and expect it to behave like an XML edit; the mesh_quat component gets destroyed. Always
   test by editing the XML text and reloading, not by poking the compiled arrays (except for *read-only*
   analysis, e.g. finding out what `geom_pos`/`geom_quat` a given XML setting compiles to).

2. **`m.mesh_vert[vadr:vadr+vnum]` gives the mesh's raw vertex cloud in the mesh's own compiled/recentered
   native frame** (same frame that `mesh_pos`/`mesh_quat` relate to the geom). Verified working:
   ```python
   mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "CustomLegFoot")
   vadr, vnum = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
   verts = m.mesh_vert[vadr:vadr+vnum]   # (10670, 3) — already in meters (mesh scale 0.001 baked in)
   ```
   Overall mesh bounding box (native frame): min `[-0.0162, -0.0396, -0.0485]`, max `[0.0173, 0.0487, 0.0490]`
   — i.e. roughly a 50mm-radius wheel-shaped part centered near the origin.

3. **The "down direction" vertex-search approach finds the HUB TIP, not the outer loop — this was my
   mistake, not a dead end for the whole method, just needs a different target.** The method: take the
   physical "straight down in world" direction for a leg's body frame, transform it back through the
   compiled `geom_quat` into the mesh's native frame (`down_native = R_compiled.T @ down_body`), then select
   mesh vertices whose direction-from-origin is close to `down_native` (`cos_angle > 0.85`). This correctly
   and robustly finds the SAME feature we aligned in the orientation fix (the small conical hub boss's tip —
   confirmed by the tiny proj range found: ~9.5mm to ~18mm from origin, i.e. right next to the hub, not out
   near the ~50mm rim). Rendering a debug cylinder there produces almost no visible red sliver poking out
   from behind the hub — consistent with it being *inside* the hub boss, not marking the visible outer loop
   the user circled.
   **For the outer loop, this same origin-relative-direction method won't work directly**, because the loop
   is a discrete, repeating radial feature (looks like 3-fold symmetry matching the 3-bolt hub pattern) that
   curves out and back — its vertices don't all share one narrow "direction from origin," so a cosine-angle
   filter from the origin doesn't isolate it cleanly. An attempt to select by (azimuth wedge around the
   loop's own local axis, across the full radius) also didn't cleanly isolate one loop — it either grabbed
   the continuous outer rim ring (which connects all loops together at large radius) or nothing at small
   radius, i.e. the wedge selection crossed between loops instead of following a single one's curve.

4. **A "transform the r-side answer through the l-mount's known extra Rz(180°)" shortcut does NOT work** for
   this mesh. I verified this two ways (both gave inconsistent results with the direct per-leg derivation):
   the l-side mount's compiled orientation relative to r's is NOT simply "same as r, plus Rz180 in world or
   body terms" — the l/r leg kinematic chains are mirror images (reflections), not related to each other by
   a pure rotation, so a feature that's "the same mesh region" on both sides does not need to end up at
   mirrored positions in body-local coordinates. Concretely: independently deriving "down_native" separately
   for `leg_front_r_3` and `leg_front_l_3` (each using its own body's actual FK orientation) gives two
   *different* raw mesh regions/positions (not mirror images of each other) — yet each one individually
   satisfies "this direction really does point straight down in world" (verified to machine precision). Both
   are individually valid "a blade/feature pointing down," just not mirror-corresponding ones. Do NOT assume
   symmetry shortcuts here — always re-derive per-leg from that leg's own kinematics, or (better, see below)
   from direct visual/ray inspection.

5. **Ray-casting through rendered pixels to find exact 3D surface points was attempted and NOT gotten
   working reliably — this is the most promising unfinished thread, worth finishing rather than restarting.**
   Setup used `mujoco.Renderer` + a free `MjvCamera` (`lookat`, `distance`, `azimuth`, `elevation`), then tried
   to reconstruct the camera ray per-pixel two ways:
   - Using `m.vis.global_.fovy` + aspect ratio to build a symmetric perspective frustum (`half_h =
     tan(fovy/2)`, `half_w = half_h * aspect`, `dir_cam = forward + ndc_x*half_w*right + ndc_y*half_h*up`).
     This worked for the exact center pixel and one nearby offset (confirmed hitting the correct mesh geom
     at a sane distance via `mujoco.mj_ray`), but other nearby off-center pixels incorrectly hit distant,
     wrong geoms (e.g. the *opposite* foot on the other side of the robot). Root cause not fully diagnosed —
     candidates: (a) actually correct physically, because this is a spoked/perforated mesh with real gaps,
     and the sampled pixel just happened to be a gap not solid material (partially likely — some retries
     landed on background-colored pixels by mistake), or (b) a residual sign/scale error in the frustum
     reconstruction that only manifests off-axis.
   - Using the `mjvGLCamera` struct's own `frustum_bottom/top/width/center/near` fields (pulled from
     `renderer.scene.camera[0]` after `update_scene`) — this failed outright because `frustum_width` read
     back as exactly `0.0` for this mono (non-stereo) render, breaking the horizontal ray component
     entirely. Didn't get to the bottom of why width is 0 (may need `frustum_bottom`/`top` times aspect
     ratio instead of relying on `frustum_width`, or there may be a different intended API/field for mono
     rendering — worth checking MuJoCo's C `mjr_readPixels`/`mjv_*` docs or the `simulate` app's own picking
     code for the canonical formula).
   Recommended next step if resuming this: verify the ray reconstruction independently of the mesh by
   ray-casting against a known flat reference (e.g. the ground plane, whose equation is trivially known:
   world Z=0) across a grid of pixels, and compare the recovered world XY positions against where those
   pixels *should* be given the camera pose — this calibrates the per-pixel ray formula without any
   ambiguity about mesh gaps.
   `mujoco.mj_ray(m, d, pnt, vec, geomgroup, flg_static, bodyexclude, geomid)` signature notes: `geomgroup`
   can be `None` for "all groups"; it returns distance (`-1` if no hit) and writes the hit geom id into the
   `geomid` output array. It respects `group` visibility filtering (not `contype`/`conaffinity`), and for
   `type="mesh"` geoms it appears to intersect the actual mesh triangles (not just a hull) — confirmed by it
   correctly resolving geom 7 (a `CustomLegFoot` mesh geom) as the center-pixel hit.

## Recommended path forward (pick one)

- **(A) Finish the ray-casting approach** (most precise, ties directly to what's visible in a render/screenshot):
  fix the per-pixel ray reconstruction (calibrate against the floor plane first per above), then click/sample
  2-4 points along the visible loop in a render (e.g. near its hub attachment and near its outward tip) to get
  exact 3D points in the mesh's native or body-local frame, and fit a cylinder (axis = tip-to-attachment
  direction, radius = half the loop's visible width) directly from those measured points.
- **(B) Manual/visual iteration** (faster, lower precision, matches how the orientation fix was actually
  validated): add a debug cylinder geom (bright, semi-transparent `rgba`, `contype="0" conaffinity="0"`) as a
  sibling of the mesh geoms under `leg_front_r_3` (start with the r side only, since its mount is the simple
  identity case), render at azimuth=90 elevation=-10 distance=0.15 looking at that body's position (this
  framing reliably shows the hub+spoke structure clearly — see prior renders), and manually nudge
  pos/size/quat by eye across a few iterations until it visually matches the loop's extent. Once good on r,
  do NOT assume the l-side answer is a mirror (see point 4 above) — repeat the same visual process
  independently for `leg_front_l_3`, then copy whichever numbers work to the corresponding back-leg (back_r
  matches front_r exactly, back_l matches front_l exactly, since the mounts are literally identical XML
  between front/back per side).
- **(C) Ask the user to click on the loop directly in their live interactive viewer** and read off the
  clicked point's world coordinates (MuJoCo's `simulate`/viewer GUI shows this in the UI when you
  double-click a geom, or via the "Select" panel) — sidesteps all the rendering/raycasting reconstruction
  entirely, since they already have the model open and oriented correctly.

## Practical constants that will be needed either way

- Foot mount local `pos="0 0 0.0136"` (r side, identity quat) / `pos="0 0 0.0136" quat="0 0 0 1"` (l side) —
  any new geom should be defined as an XML-authored `pos`/`quat` in the **body's own local frame** (same
  convention as the existing mesh geoms), NOT copied from any of the "compiled" numbers above without
  converting back.
- Standing-pose joint targets used throughout testing (gives a nice symmetric, non-degenerate leg
  configuration for rendering/measuring — NOT necessarily the real standing pose, just useful for visualization):
  ```python
  qpos[jnt_qposadr[joint("leg_*_2")]] = 1.4   # for front_r/front_l/back_r/back_l
  qpos[jnt_qposadr[joint("leg_*_3")]] = -1.4
  qpos[2] = 0.16   # base height
  mujoco.mj_forward(m, d)   # do NOT also mj_step physics for measurement — an early attempt that
                            # ran 400 mj_step iterations let the (uncontrolled) robot tip over under
                            # gravity, corrupting several early measurements. Pure mj_forward at a
                            # hand-picked symmetric qpos is more reliable for this kind of analysis.
  ```
- Collision `class="collision"` default already sets `group="3" contype="0" conaffinity="1"
  solimp="0.015 1 0.015" friction="0.8 0.02 0.01"` — reuse that class on the new geom rather than repeating
  those attributes inline.
- The mesh is reused **unmirrored** across all four legs (see existing code comment at the `<mesh
  name="CustomLegFoot" .../>` declaration) — so whatever cylinder is found for the r-side loop is measuring
  the same physical geometry as would apply to the l-side; it's only the mount transform that differs
  between them (hence point 4's warning about not blindly transforming through that mount difference).
