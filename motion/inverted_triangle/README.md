# Inverted triangles: offline lift, flip and plant

Start with **four rigid, already-transformed shins pointing up**. Shift the body,
lift one shin clear of its entire turning envelope, rotate its hub through 180
degrees, and lower it onto its tip. Keep the other three limbs supporting the
robot. Repeat from the resulting mixed stance, without resetting the robot.
Heating and material deformation are not part of this experiment.

The first approach is a physics search over a few smooth joint targets, followed
by an independent dynamic and CAD audit. This uses the existing position/PD
command interface. It does not require a new learned policy. The search currently
runs on CPU; an RTX GPU is not required. Reinforcement learning remains an option
if a robust sequence cannot be found with these trajectories.

## Sources and assumptions

| Item | Source and interpretation |
| --- | --- |
| Shin geometry, original tip length, additional **9 mm outward gap** | `source.xml` and original STL assets copied byte-for-byte from leg commit `27bc66823478758bd9dc701a5aec27cceaa0710a`, path `Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml`. This is the approved spaced leg model. |
| Rigid point-up initial state; no heating | User's explicit instructions in this task. This assumes the material holds its transformed shape under load. |
| Lower hub housings approximately 5 mm clear of the ground | User confirmation, September 12. The model uses joint-2 splay of -0.29 rad on the right and +0.29 rad on the left to approximate that clearance with the pinned CAD. **This is a candidate simulation stance, not a measured or calibrated robot pose.** |
| Proximal software joint limits and 3 Nm effort limit | `ros2_ws/src/pupper_v3_description/description/components.xacro`, joint parameters `position_min`, `position_max`, `effort_max`. These are implementation limits, not a measurement of the available motor torque. |
| Position and PD command interface | `ros2_ws/src/neural_controller/src/keyframe_controller.cpp`, `write_output`: position, velocity, effort, kp and kd commands. No new actuator mode is assumed. |
| Effort feedback is an estimate | `ros2_ws/src/control_board_hardware_interface/src/control_board_hardware_interface.cpp`, `read`, lines 853–857: reported effort is calculated from PD position/velocity error. It is not a measured foot contact force. |
| Smooth-motion limits and hub PD gains | `ros2_ws/src/neural_controller/launch/keyframe_config.json`: rates 0.45/0.65/0.5 rad/s; accelerations 2/2/1.2 rad/s²; hub kp=4, kd=0.15. Simulation proximal kp=5, kd=0.25 is an explicit experiment setting. |
| Mass, inertia and gap | Original source model values retained. The gap translates the existing assembly, COM and contact sites; it does not model additional printed spacer mass or measured polymer stiffness. |

The simulation never treats old anatomical comments as a joint-axis measurement.
Joint ordering is front-right, front-left, back-right, back-left, with joints 1,
2 and hub 3 for each. Model joint reference offsets are respected: the inverted
hub coordinate is its original reference plus pi, not an arbitrary absolute pi.

The candidate starting stance settles with approximately 4.5 mm of lower motor
housing clearance in the nominal model. The user-reported 5 mm is approximate;
this is not an additional 5 mm geometry translation. The axial spacer gap stays
exactly 9 mm. Existing saved wheel calibration is not overwritten or treated as
a measurement of the post-morph stance.

## Model and acceptance

The walking XML's tip capsules cannot support inverted triangles correctly.
`build_model.py` creates a separate transition model with the complete CAD
support envelope. Every shin and motor/body CAD hull collides **only with the
floor**. This preserves the minimum height against a flat plane, but is not an
exact deformable or concave contact simulation. The original detailed meshes are
independently checked using FCL for shin-to-motor, shin-to-body and shin-to-shin
interference. Never use a convex ring hull to certify self-clearance.

All motion is integrated in MuJoCo at 520 Hz with torque-saturated external PD.
Quintic targets have zero endpoint velocity/acceleration and durations bounded
by configured speed and acceleration limits. No pose teleportation is used to
claim motion success. Contact forces and base position are audit signals; the
fixed-time candidate trajectory does not consume them as controller feedback.

A single-flip audit requires at least 5 mm ground clearance while rotating,
three supporting shins carrying at least 1 N each, no active-shin ground contact
while rotating, no motor/body floor support, at least 1 mm sampled CAD separation,
base tilt below 8 degrees, requested torque within 3 Nm, speed limits, gentle
landing below 25 mm/s near contact, a loaded planted limb, settled motion and
hub-angle tracking. These are screening thresholds, **not hardware certification**.
CAD distance is sampled along replay; finite sampling, uncertain friction,
unmodeled compliance, actuator dynamics and calibration remain limitations.

The complete task additionally needs a continuous four-flip replay and a stable
final stance suitable for the existing leg policy. Four independent first-flip
passes do not meet that requirement. A deployable executor must add measured
joint/IMU tracking gates, timeout/abort handling and a separately validated
landing detector. The current scripts never connect to the robot.

## Run locally

Use Python 3.11 or 3.12 in a fresh virtual environment, from the repository root:

```sh
python -m pip install -r motion/inverted_triangle/requirements.txt
python -m pytest motion/inverted_triangle/test_model.py -q
python -m motion.inverted_triangle.core
```

The checked-in model is ready to load; rebuilding is unnecessary. To deliberately
rebuild it, fetch the pinned leg history and run
`python -m motion.inverted_triangle.build_model`, then inspect the diff and rerun
all affected audits. `source_manifest.json` verifies XML and mesh hashes.

Read [PC_HANDOFF.md](PC_HANDOFF.md) for the search, logging and continuation task.
Keep each run in a new output directory. Failed runs are evidence and must remain
available. Results from older model hashes cannot certify the current model.

`training/wandb_logging.py` is reused from alignment branch commit
`05299b3` without changing the shared implementation. The experiment wrapper
defaults to online logging under entity `QuadMorph`, project
`wheel-leg lift and align triangle base`. Videos are actual simulated trajectory
replays, labeled as trajectory candidates rather than trained policy checkpoints.
Offline or disabled logging must be explicitly selected and reported.

## PC follow-up

See [the committed PC review](results/pc_20260912/README.md) for the continuous four-flip plan, 20/20 friction replays, 7/16 additional stress results, all failed audits, and verified W&B rollout videos. [VALIDATION.md](VALIDATION.md) documents replay and the explicit uncertainty model. This is simulation evidence; direct policy/hardware handoff remains unvalidated.
