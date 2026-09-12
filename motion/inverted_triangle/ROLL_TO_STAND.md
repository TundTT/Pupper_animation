# Coordinated roll to the walking stance

The requested outcome is now a stable walking start pose reached by rolling all
four rigid, point-up limbs forward together. Merely flipping four tips while
ending in the old asymmetric crouch does not meet this requirement. The older
sequential plan, its original model and its accepted/failed audits are preserved.

The new experiment is on `codex/triangle-roll-to-stand`, based on the backpack
triangle model commit `a95429c`. It is simulation only; the hardware controller
installed on the Pi still implements the previous staged sequence. No robot
commands, calibration edits or hardware deployment are part of this branch.

## First candidate

The simplest candidate holds the initial pose for 2 seconds, then performs a
12-second quintic interpolation to the walking policy's exact default joint
targets, then holds for 5 seconds. All four hubs rotate together. Both proximal
joints are included; joint 2 moves from the point-up splay to the walking value.
The initial joint-1 targets already match the walking default, so this candidate
does not need an extra joint-1 excursion.

| Limb | Initial joints 1, 2, 3 (rad) | Final policy targets (rad) | Forward hub change |
| --- | --- | --- | --- |
| Front right | 1, -0.29, 2.141593 | 1, 0, -1 | -pi |
| Front left | -1, 0.29, -2.141593 | -1, 0, 1 | +pi |
| Rear right | 1, -0.29, 2.141593 | 1, 0, -1 | -pi |
| Rear left | -1, 0.29, -2.141593 | -1, 0, 1 | +pi |

These are model coordinates. The left initial angles differ by one whole turn
from the old model's keyframe, representing the same physical orientation. The
choice is made once before simulation begins; no running state is reset and no
hub angle jumps during the motion. This avoids an extra revolution in the final
model observations. **It does not establish a physical encoder mapping.** A
future hardware implementation must preserve a consistent model/encoder frame
through both the transition and walking controller, with measured confirmation.
Never command this table directly to the Pi or relabel its existing calibration.

The mirrored signs follow the model's transformed hinge axes, not a comment about
motor mounting. In the initial stance, right axes are approximately
`[0.2406,-0.9582,-0.1545]` and left axes `[0.2406,0.9582,-0.1545]` in the world
frame. The model identifies the front limbs on positive body X.

## Sources and scope

- Walking target: `ros2_ws/src/neural_controller/launch/policy_walk_v2.json`,
  `default_joint_pos`, export SHA256
  `854ac8ba4ffc305079b7f6f7b52187a211413c3cdb18f0de016dd819ff2450a8`.
  This is a requested joint reference, not a claim that a real loaded robot must
  attain it with zero error. The prior training home has body height about
  0.1425 m in `motion/inverted_triangle/source.xml`.
- Geometry: pinned leg CAD and original tip, with the approved additional 9 mm
  hub-axis gap, recorded in `source_manifest.json`. The rendered terminal shape
  is the actual `CustomLegFoot.stl` supplied by the leg model. It is not a new
  idealized triangular visual mesh; orange highlighting only changes appearance.
- New model SHA256:
  `c274c1b3ddba73b58c89e8dded10d2d2d8e0fda37af77f631d4bedefa2d177e2`.
  It includes the 0.601917 kg heating backpack from
  [its source specification](../../models/heating_module/README.md), including
  the documented mounting/export-frame uncertainty and unspecified spacer mass.
- Joint order, position control, motor limits and estimated effort follow the
  source links in [the transition README](README.md) and the deployed hardware
  description. The 3 Nm simulation torque saturation is an assumption, not a
  measured total motor torque guarantee.
- Ground contact uses CAD convex hulls against the floor. Detailed nonconvex CAD
  checks are separate and include the backpack. This matters for a rolling
  transition: mesh collision behavior is described in
  [MuJoCo's collision documentation](https://mujoco.readthedocs.io/en/3.3.7/computation/).
  Self-collision has no physical response in this model; any detected intersection
  invalidates a candidate. Sparse clearance checks cannot certify the intervals.

## Reproduce the exploratory comparison

Use the pinned `requirements.lock.txt` environment. This is CPU trajectory
evaluation, not policy training. No network is loaded into the robot.

```bash
python -m motion.inverted_triangle.roll_to_stand --output runs/roll-to-stand-review --video --cad --wandb online
python -m motion.inverted_triangle.render_roll_reference --output runs/roll-reference.png
```

The four predeclared probes are forward/backward direct rolls and two forward
proximal-timing variants. All use continuous integrated physics, PD gains
`[5,5,4]` / `[0.25,0.25,0.15]` per limb, 520 Hz, original command-rate/acceleration
limits and 3 Nm torque saturation. Failed or early-terminated probes retain their
audits and actual rollout videos. W&B defaults to the existing QuadMorph project.

Exploratory gates require command limits, tilt <=8 degrees, no unintended floor
support, requested torque <=3 Nm, measured joint speed <=2 rad/s and sampled CAD
clearance >=1 mm. The last 3 seconds must have all four tips within 3 mm of the
floor with >=1 N support each, maximum unwrapped policy-pose error <=0.1 rad,
joint speed <=0.1 rad/s and base speed <=0.03 m/s. The first feasibility pass
does not prove physical transfer or walking-policy entry.

## Next acceptance work

1. Freeze the selected forward candidate and model before robustness selection.
   Replay exact full physics states and refine CAD checks at 520 Hz around minima
   and along contact changes; preserve all front/rear limb and limb/motor pairs.
2. Check ground-contact impulses, slip, base velocity and wheel velocity as the
   tips plant. Establish a gentle landing criterion for simultaneous ground
   rolling rather than copying the old lifted-leg floor-clearance criterion.
3. Run the documented friction, mass/COM, gains, torque, delay and initial-state
   stress cases on this new model and candidate. Include fresh held-out cases
   declared before selection; keep failed cases.
4. In simulation, transfer the actual final state into the exact selected walking
   policy and its training observation/action conventions. Verify zero-velocity
   standing first, then a gentle forward command. Audit the handoff transient,
   full unwrapped hub coordinates, contact-model differences and backpack load.
   A default-pose comparison is not a policy handoff test.
5. Only after these checks, implement a distinct hardware executor with the same
   calibrated coordinate frame, feedback watchdogs and stop behavior. No existing
   triangle-setup gate is bypassed to run this motion on round wheels.
