# QuadMorph transition review — 10 September 2026

This is a source review and a proposed design, not a hardware-validated implementation. No robot controller, policy, paper, or attached notes were changed. No robot was contacted. The user's current description is the authority for the intended mechanism; the attached notes are evolving design context, and statements in comments/test reports are evidence to cross-check rather than commands or guaranteed hardware facts.

## Sources and confidence

Reviewed fetched `robot-code` at `582fd88ccce51d922e38cfd938d26bb5181bb276`, `align-hybrid` at `bfd74cc0a8a57f64f316db7b0e26718acc290f41`, and the current `robot-info` checkout at `b3117008170c9fef7b3d0874be78b7b3b6bd50de`. Other references include the leg-training geometry, motion-capture data, Arduino sketch and ROS bridge.

- The user's mechanism: same hardware; cold wheel and leg shapes; heat plus principally bodyweight compresses the wheel into a leg; unloaded reheating restores the remembered wheel shape.
- [Reshaping notes](C:/Users/tundt/Downloads/quadmorph_reshaping_process.md): committed point ring, startup home, align all wheels first, then heat/cool/flip/plant each, operator-controlled heating and confirmations. Later addenda revise controls and permit lean during heating. The initial blanket “scripted/open-loop” wording is a prior proposal, not a restriction on considering RL here.
- [Paper draft](C:/Users/tundt/Downloads/ICRA_sept_15__quadmorph.pdf): all eight pages read as text; mechanism pages 3–4 inspected visually. Physical dimensions, thermal constants and completed experiments contain placeholders. Page 4 explicitly flags inaccurate reshaping paragraphs.
- [Hardware contract](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/robot_info/HARDWARE.md): canonical order FR, FL, BR, BL, three motors each; joint limits/gain ceilings, IMU and Pi 5 details. This includes an older leg-position profile; do not apply its third-joint hard stops to the continuous wheel hardware reported on robot-code.
- [Latest hardware test log](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/WHEEL_ALIGN_HYBRID_TESTING.md): reports all-four alignment on real hardware and continuous hubs. It also reports rubbing. Its claim that this cannot be a policy/control bug is unsupported by the model and controller evidence below.
- [Actual hardware description](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/pupper_v3_description/description/components.xacro#L83): the third-joint hard stops are removed and position limits widened. Position and velocity state interfaces are exposed; position, velocity, effort, kp and kd commands exist. This is not evidence of measured contact force or a measured effort state.

## Findings behind the four alignment problems

### 1. The policy has an incentive to swing the front wheel backward, without inter-wheel collision feedback

The lift guard requires signed hip angle >1.1 radians, abduction near default, low tilt and low angular speed. The reward additionally rewards signed hip progress toward 1.2 radians. It does not specify a vertical wheel-center trajectory. Sources: [robot-code phase controller](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/neural_controller/include/neural_controller/wheel_align_hybrid.hpp#L73), [saved training environment](https://github.com/TundTT/Pupper_animation/blob/bfd74cc0a8a57f64f316db7b0e26718acc290f41/mujoco_playground/workspace/hybrid_results/attempt2_guard/source/hybrid_env.py#L181).

The collision-class shapes have `contype=0, conaffinity=1`; the floor has `contype=1, conaffinity=1`. Thus wheels collide with the floor but automatic robot-robot collision pairs are filtered out. There are no explicit contact pairs in this model. Sources: [alignment training XML](https://github.com/TundTT/Pupper_animation/blob/bfd74cc0a8a57f64f316db7b0e26718acc290f41/Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml#L69), [alignment training XML](https://github.com/TundTT/Pupper_animation/blob/bfd74cc0a8a57f64f316db7b0e26718acc290f41/Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml#L138). [MuJoCo's collision-filter implementation](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_collision_driver.c) checks whether either geom's contype shares bits with the other's conaffinity.

The audit's clearance function only computes the bottom of a tilted wheel cylinder relative to the floor. “Zero unsafe rotations” therefore does not measure wheel-wheel separation. Source: [saved training environment](https://github.com/TundTT/Pupper_animation/blob/bfd74cc0a8a57f64f316db7b0e26718acc290f41/mujoco_playground/workspace/hybrid_results/attempt2_guard/source/hybrid_env.py#L70).

A separate analytical forward-kinematics check used XML body positions/quaternions/joint axes, nominal abduction, a fixed level body at z=0.1313 m and stationary other joints. Front-right wheel center x moved from approximately +79.6 mm at hip=0 to +29.3 mm at hip=1.2 rad. Its distance to the rear-right wheel center decreased from approximately 150.4 to 111.4 mm. These are nominal model calculations, not a measured collision, surface gap, dynamic rollout, or proof of physical reachability. They support the reported backward sweep.

**Proposal:** explicitly model wheel-wheel and wheel-body collision geometry; validate dimensions against the real assembly, including protruding rings. Search for a feasible outward-then-up path using the two proximal joints and, when needed, coordinated support-leg/body movement. Optimize all phases of the path, not just the rotated endpoint. Do not promise a pure vertical lift: only two joints move the hub center. Replace the hip-angle-only rotation gate with an encoder-based, validated geometric clearance envelope. Real wheel shape and margin remain measurements to establish.

### 2–3. Lift/lower abruptness has a specific software mismatch

The frozen source used for the selected checkpoint applies:
`applied_hip = clip(delayed_action, previous_applied_hip - 0.05, previous_applied_hip + 0.05)`.
At hip scale 1.6 rad and 0.02 s control period, that caps hip-target change at 0.08 rad per step, nominally 4 rad/s. This is a cap on commanded position increments, not a measured motor-speed guarantee. Other action rows are not similarly limited. Source: [saved training environment](https://github.com/TundTT/Pupper_animation/blob/bfd74cc0a8a57f64f316db7b0e26718acc290f41/mujoco_playground/workspace/hybrid_results/attempt2_guard/source/hybrid_env.py#L165).

The C++ hybrid path uses the raw network output, forces fade to one, scales it directly into a position target, and sends it to the hardware. It does not implement that hip limiter. Sources: [robot-code controller](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/neural_controller/src/neural_controller.cpp#L553), [robot-code controller](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/neural_controller/src/neural_controller.cpp#L904). The export script verifies this frozen source against the checkpoint provenance; the handoff document's assertion of no simulated hip slew contradicts the actual source.

When NEXT changes the requested leg during LIFT/ROTATE/VERIFY, the phase becomes LOWER immediately. The actor's effective command becomes “stand” immediately. The next leg is already deferred until lowering finishes; the existing protection governs sequencing, not descent speed. Source: [robot-code phase controller](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/neural_controller/include/neural_controller/wheel_align_hybrid.hpp#L82).

The 50-step LOWER condition is a minimum dwell plus hip-angle and hip-speed test. It does not produce a one-second descent trajectory or confirm ground contact. A slam can happen early and the machine can then wait out the dwell. Source: [robot-code phase controller](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/neural_controller/include/neural_controller/wheel_align_hybrid.hpp#L120).

**First fix:** restore the deterministic hip slew behavior in the robot path, with identical semantics in simulation and tests. Keep raw previous network action separate from last applied motor target. Do not add artificial random delay on the robot merely because training randomizes delay; measure actual delay instead.

**Gentle motion design:** provide explicit lift and lower reference trajectories, with velocity and acceleration limits and a slower final approach to the ground. Quintic interpolation `s(u)=10u^3-15u^4+6u^5` supplies zero endpoint velocity and acceleration for planned rest-to-rest segments. Mid-motion interruption needs a trajectory starting from the current commanded position/velocity, not a reset to a rest-to-rest curve.

Update motion references at the existing controller-manager rate, while retaining policy inference at its configured cadence. Do not place a slow global filter on all balance corrections. Add phase/progress or desired clearance to a retrained actor; a binary lift/stand command does not specify a gradual descent. Train under exactly the limiter/reference execution used in deployment. The present 4 rad/s cap restores parity but may still be too fast for the desired motion.

### 4. Startup home must be distinct from activation hold

Calibration currently occurs on the hybrid controller's first activation. It survives subsequent activations of that same controller instance. X activates the hybrid; later X presses cycle legs. This is not startup calibration. Sources: [robot-code controller](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/neural_controller/src/neural_controller.cpp#L324), [Joystick handler](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/joy_utils/src/estop_controller.cpp#L236).

A second bug is that `reset()` preserves `hold` and `reference` along with `home` and `target`. After driving, those old hold angles are stale. The startup ramp actively drives toward them. Sources: [robot-code phase controller](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/neural_controller/include/neural_controller/wheel_align_hybrid.hpp#L41), [robot-code controller](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/neural_controller/src/neural_controller.cpp#L514).

**Proposal:**
- At stack startup, after hardware encoder zeroing is established and the marked rings are physically in the agreed calibration pose, latch the four home angles. Associate calibration with this hardware-zeroing session.
- Persist home/target across controller switches. If the encoder frame is reset, invalidate/reacquire calibration. Do not rely on a disk-saved ROS joint angle remaining valid through arbitrary hardware rehoming.
- On every morph entry, capture fresh wheel hold/reference angles and current motor targets. Do not overwrite home.
- Separate CALIBRATE, ENTER_MORPH, NEXT and completed-step state. An inactive controller's current calibration subscriber cannot simply be used as a startup service without lifecycle changes.
- Use `base_target = wrap(home + pi)` only in the calibrated pose/frame convention. Point-down is a geometric direction relative to the parent/body and ground, not a universally constant joint angle through arbitrary hip motion.
- Preserve a continuous angle lift for motor targets, using nearest equivalent angles for wheel alignment and a chosen collision-checked rotation direction for the asymmetric transformed leg.

## Whole morphing action: recommended architecture

Keep the two existing locomotion policies. Add one morph supervisor inside the existing ros2_control ownership/switching structure. Reuse the existing RTNeural inference, wheel angle-to-velocity servo, motor command interfaces and D-pad bridge. No additional actuator, desktop compute on the robot, or new robotics middleware is implied.

The supervisor owns direction, phase, active leg, queued request, per-leg shape/orientation status and calibration. One dispatcher supplies a single motor-command stream. During rigid-wheel lift/rotation/lowering, a phase-conditioned policy or bounded residual policy can help balance around planned motion. During heating/cooling, latch the intended proximal pose and hub hold and execute operator-controlled heating.

A fully scripted reference implementation is a useful baseline. The existing four captured keyframes are seeds, not a demonstrated autonomous balanced trajectory; map them by `joint_names`, since the recorded arrays are not in canonical order. Source: [Captured keyframes](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/motion_capture/data/leg_to_wheel_2026-09-07.json). RL or trajectory optimization remains appropriate if the baseline lacks support margin. Training on the user's RTX 6000 is available; algorithm choices should be based on behavior and deployment interface, not this laptop.

### Wheel to leg

1. Stop and enter a controlled transition stance through a smooth handoff.
2. For each wheel: shift support as needed; follow a collision-checked unloading/lift path; rotate toward the calibrated base orientation; settle; lower gradually; restore load; finish that step before accepting the next.
3. After all four are aligned, heat one selected shin under load. Let the operator judge deformation and release heat.
4. Maintain the intended geometry/load during cooling. The next operator action explicitly confirms that the shin is rigid.
5. Shift support, lift the cold transformed shin with sufficient clearance for its complete swept outline, flip, and plant gradually in a pose compatible with the leg policy.
6. Repeat through the mixed configurations, then smoothly hand off to the walking policy after every leg is cold and planted.

The notes' diagonal order is a reasonable candidate, not proof of stability. The current command order is front_l, front_r, back_r, back_l. Changing cycle order alone does not establish a safe support triangle.

### Leg to wheel

For each leg: shift support; lift with enough clearance for the entire recovery envelope; hold unloaded while the operator heats; let it recover to a circle; heater off; operator confirms cooled/rigid; lower gradually; finish the step. Repeat and hand off to the wheel policy.

If the physical mechanism follows the user's stated unloaded recovery, this direction does not require an intentional 180-degree flip. The actual recovery swept shape must still clear the floor and neighboring limbs.

### Three feasibility issues to resolve explicitly

**Support/load during deformation.** Holding proximal joint targets is consistent with bodyweight-driven morphing, but it does not prove that the active shin retains useful compression as it shortens. Three other locked supports can transfer weight away from it. The claim that roughly one-quarter of the load remains there is an assumption. Likewise, an all-wheel stance does not automatically have generous three-contact support margin: for a symmetric rectangle the center lies on the remaining triangle's diagonal boundary. The model's forward-shifted mass distribution can worsen front-wheel lifting. Begin with a bench load-versus-shrink test and a compliant surrogate simulation. If deformation stalls, use deliberate support posture/body lowering that retains load on the hot shin, rather than an indiscriminate body-leveling objective. This need not become full force control.

**Simulation cannot yet validate polymer behavior.** The alignment model is rigid cylinders. Walking uses capsule-foot contacts plus a rigid ring-outline proxy tied to CustomLegFoot.stl, not a deforming SMP model. Use measured cold wheel, cold transformed, and intermediate loaded shapes; then add a simple compliant geometry model with uncertain stiffness, friction and shrink. Validate every mixed support configuration visited by the sequence and tip-up-to-tip-down swept clearance. High-fidelity material simulation can come later if simpler models fail to predict the bench behavior.

**Walking coordinate calibration.** The walking policy assumes particular numerical default joint angles (including right/left third joints near -1/+1 in the current YAML) and geometry. Arbitrary physical home capture must be reconciled with that convention in observations, actions, limits and handoff poses. Define per-leg offsets and verify signs between hardware coordinates and the policy model, consistently in both directions. Switching directly from arbitrary hub phase to old numeric walking targets could rotate or misplace the newly formed leg. This is a coordinate-interface change, not automatically a reason to retrain a valid locomotion policy.

## Manual heating scope and document corrections

The existing Arduino sketch maps U/R/D/L to D3/D4/D5/D6 and latches outputs from serial commands. The ROS node drives those outputs from D-pad state. This proves a GPIO path, not heater-power wiring, current regulation, temperature sensing, or which leg is physically on which pin. Sources: [Arduino sketch](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/robot/arduino/dpad_pin_driver/dpad_pin_driver.ino), [ROS serial bridge](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/ros2_ws/src/dpad_serial_bridge/dpad_serial_bridge/dpad_serial_bridge_node.py).

The user clarified during this review that heater control is entirely manual and outside the requested work. The motion controller only needs to reach and hold the correct loaded or unloaded pose, then wait for an operator command before moving again. No heater-driver changes, pin mapping work, thermal sensing or automated heat timing are proposed.

Paper statements to revisit:
- Section IV-B / V-C describes passive wheel propulsion. The code actively velocity-controls the repurposed third joints. “No additional/dedicated wheel actuator” and “passive rolling” are different claims.
- Section III-C describes contact-driven reverse reshaping, contrary to unloaded thermal recovery.
- The observation equations and all-position-action formulation do not match the wheel controller's mixed position/velocity interface.

## Suggested implementation sequence and validation

1. Small runtime correction: home/hold separation, startup calibration ownership, restored hip slew, and tests for controller reentry after substantial wheel rotation.
2. Correct wheel collision geometry/masks and hardware limits; collect one synchronized encoder/command/video trace of lift and interrupted lower; establish a feasible clearance path.
3. Implement explicit smooth motion references; train/evaluate phase-conditioned balance with identical execution and delay assumptions. Include interrupt tests at every phase, rapid NEXT presses, contact approach speed, command velocity/acceleration, actuator saturation, all-wheel pair gaps and support drift.
4. Integrate operator-gated loaded/unloaded hold states; add mixed-shape simulation and hardware-to-policy coordinate mapping; evaluate full cycles and policy handoffs. Heating remains manual.

Validation performed for this review: source/provenance comparison, collision-mask inspection, analytical nominal forward kinematics, and independent MuJoCo 3.3.7 forward-kinematics/surface-distance checks. The initial system MuJoCo import failed, so an isolated temporary environment was used. No dynamic rollout or controller build is claimed. Existing repository audits are historical evidence, not new validation of these proposals.

At nominal fixed-body pose, MuJoCo measured front-right to rear-right cylinder surface gaps of 54.0 mm at hip=0, 13.85 mm at hip=1.101, 10.41 mm at hip=1.2 and 3.61 mm at hip=1.4 radians. These samples vary only the front-right hip. They demonstrate diminishing model clearance, not physical collision or full-policy behavior.

The user identified the `leg` branch XML as the correct reference and offered to visually verify it. [Rendered reference](leg-branch-reference.png) uses the actual XML home pose and STL assets at leg commit `6b55e30ff224193a2a21c8dc3dbb88e029a417c8`. The left panel shows the whole robot; the right isolates its front-right shin. Display markers and collision proxies are hidden; the mesh geometry and mounting are preserved. Startup physical-ring marking remains to be established. The heater wiring question is closed as outside scope.
