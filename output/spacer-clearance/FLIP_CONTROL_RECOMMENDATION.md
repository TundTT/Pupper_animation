# QuadMorph inverted-shin transition recommendation

## Decision

Use **offline optimization of a slow, sequential whole-body motion, executed through the existing keyframe controller with encoder feedback and measured-state gates**. Heat all four aligned shins together, keep the deformation load applied until they cool and lock, and then flip one rigid shin at a time. The other three limbs provide support. Each successful landing becomes the starting support configuration for the next flip.

This is the recommended route to the first self-supported demonstration because the command interface, smooth interpolation, hub-angle tracking, calibration and operator controls already have implementations in the repository. Offline optimization should find the body shifts and clearance poses; the robot should execute a small reviewed trajectory rather than carry a new online optimization stack. This recommendation is an engineering judgment based on the repository and related research, not a demonstrated QuadMorph transition.

The immediate deliverable should be a **contact-correct simulation showing one complete unsupported lift, flip and plant**, followed by the full four-limb sequence. A static pose, an animation made by assigning joint positions, or a high training reward is insufficient evidence.

## Scope and source authority

The design basis is the September 12 instruction: all four shins first become point-up triangles, and the task is to orient them point-down. Heating remains manual. The motion design starts after the operator confirms that all four are rigid enough to support the robot. No simultaneous four-limb lift is proposed.

The earlier [reshaping outline](C:/Users/tundt/Downloads/quadmorph_reshaping_process.md) describes heating and flipping one limb at a time. Its heating order is superseded by the newer simultaneous-heating instruction. Its comments about contact geometry, calibration and open-loop control are design context, not experimental proof or instructions to operate hardware. The [paper draft](C:/Users/tundt/Downloads/ICRA_sept_15__quadmorph.pdf), especially sections III-C and V, contains author correction notes and placeholder results. It is not evidence that a transition controller has been demonstrated.

Source revisions inspected on September 12:

| Source | Revision | Role |
|---|---|---|
| `leg` training model | `27bc66823478758bd9dc701a5aec27cceaa0710a` | Original shin CAD plus 9 mm outward gap |
| `wheel` training model | `88f6a8888644c8072df36d4878d77684972c9231` | Wheel geometry with the same gap |
| `robot-code` | `e3e1d9737c51f904ff7d0f4cae5692ac4703f659` | Hardware interfaces and calibration baseline |
| `robot-info` | `e9b0417` | Hardware/software audit and source map |
| Alignment development | `05299b3a61bf45bb5e89a371fa3313643be2014e` | Deterministic core and prior simulation record |
| Keyframe robot integration | `c11f5ced198961c4f49af8b0833417077f9dbb9b` | More recent hub position-control implementation |

## Geometry changes the problem

The 9 mm axial gap addresses interference with the upper motor in the previously checked poses. It does not add vertical travel or establish clearance for newly optimized poses. New whole-body paths still need checks against the upper motor, lower link, torso, neighboring shins and floor.

A new CAD calculation places all four unshortened shins point-up, uses the source neutral proximal angles and a level body, and translates the robot vertically until the lowest shin surface reaches a flat floor. It then sweeps the hub through a complete revolution while holding the proximal pose fixed:

| Quantity | CAD result |
|---|---:|
| Initial hub height above floor | 25.45 mm |
| Downward reach of the point in the nominal point-down orientation | About 62.69 mm |
| Largest downward reach during the full revolution | About 64.87 mm |
| Straight vertical rise needed for the full-turn envelope to clear by 5 mm | **44.42 mm** |

The calculation is approximately `64.87 + 5 - 25.45 = 44.42 mm`. It is a conservative full-turn envelope, not a claimed minimum for every possible coordinated lift-and-rotate path. Tilting the hub axis or moving the body can change the requirement. The 5 mm allowance is a candidate planning margin, not a physically established requirement. These numbers do not include load equilibrium, actual material deformation or manufacturing uncertainty. [Calculation and assumptions](flip-research-geometry.json), [reproducible script](flip_research_geometry.py).

This means the old wheel lift is a useful starting guess, but its clearance cannot simply be reused. The effective hub-to-ground length changes by roughly 37 mm between the neutral point-up and point-down configurations. A newly planted limb therefore needs a different proximal pose or body posture. Returning every limb to the old wheel standing pose after its flip can lift or tilt the body and unload the remaining supports.

## First fix: a dedicated transition contact model

The leg XML currently represents physical ground contact using a capsule around the long lower petal. Its outer rim and the other petals are visual-only. Once the shin is inverted, those omitted surfaces are precisely the surfaces that must support the robot. The 9 mm update translated the existing geometry consistently; it did not add those missing contacts. [Active XML, lines 117–140](https://github.com/TundTT/Pupper_animation/blob/27bc66823478758bd9dc701a5aec27cceaa0710a/Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml#L117).

Create a separate transition model using the original locked triangular shape and the same 9 mm gap. Give the two base-contact regions, tip, and relevant rim sections physical collision geometry. Start with a small set of fitted primitives or convex components and inspect them against the CAD; use detailed CAD distances for independent interference auditing. Avoid interpreting a single convex hull of the complete hollow ring as its exact collision surface. MuJoCo explicitly documents convex-hull mesh collision and recommends convex decomposition for non-convex shapes. [MuJoCo collision documentation](https://mujoco.readthedocs.io/en/latest/computation/#collision-detection).

The transition model must also use the hardware's continuous hub travel. The leg training XML still has locomotion-era joint-3 limits; a full flip can exceed those bounds. Reuse the current hardware hub envelope, and keep the actual proximal limits and actuator behavior. This is a model/configuration change, not another physical joint. Start from a measured or conservatively varied locked shape. A full thermo-mechanical polymer simulator is unnecessary for planning the post-cooling flip.

## Recommended motion

The support states progress through `4 point-up → 3 point-up + 1 point-down → 2 + 2 → 1 + 3 → 4 point-down`. The controller must remember which limbs have been flipped and retain their new planted postures.

For each limb:

1. **Transfer weight.** Coordinate the supporting limbs to move the center of mass toward their support region. Include body translation, modest roll/pitch, and the poses of already converted limbs. Do not treat center-of-mass projection alone as sufficient: joint reach, possible contact forces, friction and actuator demand also matter.
2. **Lift to a feasible clearance pose.** Keep the active shin in its point-up orientation initially. Move the proximal joints and body together until the complete planned rotation has a verified floor and self-collision margin. Allow a small number of intermediate lift poses if a straight path is blocked.
3. **Turn the hub.** Execute the selected half-turn using the existing encoder-based angle control and a smooth speed-limited target. Evaluate both signs offline, retain the chosen winding, and target the calibrated point-ring reference. Account for proximal posture when checking world-space tip direction; a hub angle alone is not a world orientation.
4. **Plant gently.** Lower the tip toward its planned contact and gradually transfer load onto it. Stop the descent at the reviewed landing target, with bounded tracking error and a settle interval. For the first physical trials, let the operator confirm planting before the next weight transfer. Avoid claiming automatic force-based touchdown from the present effort topic.
5. **Retain the new stance.** Keep the planted limb in its stage-specific pose, update the converted-limb mask, then plan the next shift. After the fourth landing, transition smoothly into the entry stance of the leg policy.

A diagonal sequence is a reasonable seed. Evaluate the 24 possible limb orders and both hub-turn directions using the staged support geometries; choose based on the worst balance, clearance and actuator margins, rather than assuming one order is always best. A previously flipped limb may help support a later lift, but it may also reduce the available base-contact area. The ordering decision must account for both effects.

## Offline planning and validation

Use the existing MuJoCo model-loading workflow and SciPy initially. There is already an offline keyframe-fitting script in [the alignment development tree](C:/Users/tundt/Desktop/Pupper_alignment_keyframes/motion/keyframe_align/fit_reference.py). Its circle-based clearance logic and static pose assumptions need replacement for this task; it is a reference implementation, not a ready flip planner.

Start with constrained waypoint inverse kinematics and static contact-load feasibility. Optimize the body pose and coordinated joint targets for weight shift, lift, turn and landing. Enforce proximal limits, nonpenetration, fixed/supporting contact consistency, positive normal forces, friction bounds and actuator budgets. Prefer a comfortable interior support margin and smooth, small joint changes. SciPy's SLSQP supports bounds, equality constraints and inequality constraints; this is adequate for a small initial feasibility problem. [SciPy documentation](https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.minimize-slsqp.html).

Then time-parameterize the candidate with the existing quintic interpolation and command-rate limits, and replay the actual controller in forward dynamics. Refine a small set of waypoint and timing parameters against this replay. A successful waypoint solve does not imply that interpolation is collision-free, that contact persists, or that the available PD gains can follow the motion.

Replay all four stages consecutively and vary actual starting joint angles, locked-shin dimensions, contact friction, mass distribution, motor tracking, calibration offsets and modest floor unevenness. Perturbations should be tied to measurements as they become available. Report minimum CAD clearance, support loss/slip, body tilt, tracking error, peak/held actuator demand, final angle, and landing velocity. Check intermediate samples and refine close encounters rather than certifying a path from a few rendered keyframes.

The first pass should use slow execution on flat ground. If adequate clearance requires an extreme pose or sustained demand near the configured actuator ceiling, reject it and change the path or support strategy. Speeding up an infeasible quasi-static path turns it into a different dynamic maneuver; that should not happen accidentally through tuning.

## Hardware fit and existing code

The [hardware audit](C:/Users/tundt/Desktop/Pupper_robot_info/robot_info/HARDWARE.md) records a Raspberry Pi 5, three motors per limb, encoder feedback and a BNO055 IMU. The current description exposes position, velocity, effort and gain commands. It configures a 3 Nm effort limit and continuous hub travel. These are source/configuration facts, not new measurements of continuous torque capability. [Current motor description](https://github.com/TundTT/Pupper_animation/blob/e3e1d9737c51f904ff7d0f4cae5692ac4703f659/ros2_ws/src/pupper_v3_description/description/components.xacro#L86).

The newer keyframe adapter directly commands proximal positions and hub positions with PD gains, plus bounded hub feed-forward effort. It does not require a neural network. [Actual output routing](C:/Users/tundt/Desktop/Pupper_keyframe_robot/ros2_ws/src/neural_controller/src/keyframe_controller.cpp:131). The current JSON selects `position_pid`, unlike older documentation paragraphs that describe velocity-controlled hubs. [Current configuration](C:/Users/tundt/Desktop/Pupper_keyframe_robot/ros2_ws/src/neural_controller/launch/keyframe_config.json:56). This distinction is why the newer implementation should be the integration starting point.

Reuse its timing, angle wrapping, smooth commands and operator interaction, but replace circle-specific floor checks, stage poses, completion bookkeeping and recentering behavior. The existing IMU checks can gate progression; they are not proof of active balance recovery. Add bounded posture correction only if replay or physical tracking shows it is needed, and revalidate the corrected path.

The joint effort state is calculated as proportional error plus derivative error in the hardware read routine. It is not an independent measured force or current signal. [Effort implementation](C:/Users/tundt/Desktop/Pupper_animation/ros2_ws/src/control_board_hardware_interface/src/control_board_hardware_interface.cpp:871). Encoder tracking and IMU tilt can help identify a bad landing but do not directly measure ground contact. Initial operator confirmation avoids building an unvalidated touchdown detector into the first demonstration.

The hardware acceptance of the newer keyframe control law remains separate from earlier velocity-mode simulation records. Its [lab note](C:/Users/tundt/Desktop/Pupper_keyframe_robot/KEYFRAME_LAB.md) documents this distinction and an earlier startup fault. Use the corrected integration; do not interpret repository presence as physical validation. Future startup must follow the existing shared calibration procedure. No hardware operation is part of this research.

## Alternatives and escalation

| Approach | Main advantage | Development burden for this robot | Recommendation |
|---|---|---|---|
| Hand-tuned keyframes alone | Immediate use of existing controller | Repeated manual search for sufficient lift and mixed-stance balance | Use as initial guesses, not the entire engineering method |
| Offline optimized keyframes with feedback gates | Reuses runtime; makes clearance and support constraints explicit | Transition contact model and offline search | **First choice** |
| Reference-tracking or residual RL | Can add recovery around a known feasible motion | Training, randomization, phase observations and export/replay validation | Escalate if nominal motion works but realistic disturbances defeat tracking |
| Whole-body online MPC | Continuously replans motion and contact loads | Larger integration and estimation burden; different runtime architecture | Use only if the maneuver demonstrably requires it |
| Rolling the triangle through floor contact | May reduce lift height | Deliberately changes contact and loads the polymer during rotation | Separate fallback requiring its own contact study |
| External belly support or suspension | Removes balancing from the first flip test | A simple fixture and explicit assisted-test labeling | Fastest way to isolate hub motion and geometry, not proof of self-support |

There is no reason to reject RL because of the laptop. The limiting work is defining and validating the changing contacts, material geometry, observations and success conditions. The same model correction is necessary for optimization and RL. If learning is needed, track the planned sequence and give the policy measured joint state, IMU state, active limb, phase and converted-limb mask. Restrict the initial residual to bounded posture corrections and keep heating, turn completion and ground-clearance decisions explicit. A policy should not be able to earn task reward by scraping through a missing collider.

Use a slack catch support for early self-supported trials, or a load-bearing fixture for the explicitly assisted geometry test. Record which it is. If no feasible three-support path is found, first distinguish a local optimizer failure from a genuine reach/contact limitation using multiple initial guesses, orders and turn directions. Only then choose a deliberate assisted-contact strategy or dynamic controller.

## What the literature establishes

Nikolić and colleagues demonstrate quasi-static CoM planning on a Solo12 quadruped using contact, torque and reachability feasibility, separating weight transfer from swing. This supports the proposed slow weight-transfer structure; their whole-body torque controller is not a drop-in implementation for QuadMorph. [Actuators, 2025](https://www.mdpi.com/2076-0825/14/5/202).

Yu and Rosendo demonstrate a hand-crafted multimodal transition coupled with learned locomotion on another quadruped. Their robot uses an additional supporting structure, so its physical sequence does not transfer directly. It does establish that learning the locomotion policies does not require learning the transition from scratch. [Multi-Modal Legged Locomotion Framework, 2022](https://arxiv.org/html/2202.12033v1).

Johannink and colleagues demonstrate residual RL layered with conventional control in a contact-rich manipulation task. It is evidence for the decomposition, not evidence of QuadMorph balance. Tan and colleagues demonstrate quadruped sim-to-real learning with actuator/latency modeling and randomization. Together they support learning as a later robustness tool while emphasizing the model work it still needs. [Residual RL](https://arxiv.org/abs/1812.03201), [quadruped sim-to-real](https://arxiv.org/abs/1804.10332).

Drake provides inverse-kinematics distance constraints, and Crocoddyl provides contact-sequence optimal control with quadruped examples. Both are credible alternatives if the small MuJoCo/SciPy search proves inadequate. Their availability is not itself a reason to port this project before trying the existing tools. [Drake IK](https://drake.mit.edu/pydrake/pydrake.multibody.inverse_kinematics.html), [Crocoddyl](https://github.com/loco-3d/crocoddyl).

## Next deliverables and completion criteria

1. A reviewed transition XML in which four rigid point-up shins stand on their actual base surfaces with the 9 mm gap.
2. One collision-free, dynamically replayed three-support flip, with a video and the clearance/support/torque trace. Start with whichever limb the feasibility screen finds easiest.
3. A complete four-flip rollout that preserves each newly planted stance and ends at a leg-policy-compatible entry pose. Existing policies trained before the gap change require a compatibility check at handoff.
4. A supervised staged physical trial: supported geometry/angle check, one self-supported flip, then the full sequence. Operator heating/cooling and planting confirmations remain explicit until repeatability is established.

No controller implementation, policy training, deployment or physical transition was performed for this recommendation. The new quantitative result is the CAD floor-envelope screening; full trajectory feasibility remains the next task.

## External references

- Nikolić, M.; Mitić, V.; Savić, S.; Zhang, T. “Efficient CoM Motion Planning for Quadruped Robots’ Quasi-Static Walking.” *Actuators* 14(5), 202, 2025. https://doi.org/10.3390/act14050202
- Yu, C.; Rosendo, A. “Multi-Modal Legged Locomotion Framework with Automated Residual Reinforcement Learning.” 2022. https://arxiv.org/abs/2202.12033
- Johannink, T. et al. “Residual Reinforcement Learning for Robot Control.” 2018. https://arxiv.org/abs/1812.03201
- Tan, J. et al. “Sim-to-Real: Learning Agile Locomotion for Quadruped Robots.” 2018. https://arxiv.org/abs/1804.10332
- MuJoCo documentation, “Computation: collision detection and convex decomposition.” https://mujoco.readthedocs.io/en/latest/computation/
- SciPy 1.16.2 documentation, “minimize(method='SLSQP').” https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.minimize-slsqp.html
- Drake documentation, “pydrake.multibody.inverse_kinematics.” https://drake.mit.edu/pydrake/pydrake.multibody.inverse_kinematics.html
- Mastalli et al., Crocoddyl official repository and ICRA 2020 framework reference. https://github.com/loco-3d/crocoddyl
