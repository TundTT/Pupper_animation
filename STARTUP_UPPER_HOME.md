# Required startup: automatic upper home, manual wheel hubs

This is the operator's September 14 standing instruction for every agent asked
to run this robot stack. It supersedes historical hanging-pose and manual
upper-stop procedures. Reuse valid running sessions; do not restart to switch
policies. If starting hardware, run automatic stop homing for motors 1 and 2,
then return to the saved upper home. Do not substitute Stanford zero or a policy
stance for the saved home.

The real-hardware xacro enables this in the driver for every launch using
`pupper_v3.urdf.xacro`. Simulation is unchanged. Motors 2 seek first, then motors
1. The stop-derived offsets are recomputed each activation. All four motor-3
hubs remain unpowered and receive no automatic position commands.

Runtime targets are `ros2_ws/src/pupper_v3_description/description/upper_home.yaml`.
The approved measurement and provenance are `hardware_testing/start_pose/upper_home.json`;
the same eight positions are in `start_pose.json`. Prior hub entries in that
12-joint file are historical targets, not authorization to move hubs at startup.

## Agent procedure

1. Check the selected checkout and active hardware processes. This robot uses
   `/home/pi/robot-code-leglift`. Never start a second hardware owner.
2. Confirm the robot is supported and joints are clear. Existing explicit
   confirmation in the current setup persists; do not repeatedly ask. The upper
   legs do not need to be posed manually at stops or at home.
3. Use `bash scripts/run_robot_stack.sh --supported` for the normal combined
   stack. Use `bash scripts/run_robot_stack.sh --supported startup_home.launch.py`
   for home plus state broadcasters only. The flag records actual operator
   confirmation, not permission an agent may invent. The wrapper selects the
   tested overlay and sets a boot-specific confirmation checked by the driver
   before enabling motors. An unconfirmed startup fails before motion.
4. Wait for `Saved upper home reached` and successful activation. The driver
   holds upper joints at low gains; it does not disable them after normal startup.
   A timeout/fault disables the motors and leaves calibration invalid. Do not
   bypass a fault or apply a previous boot's raw offsets.
5. With upper joints at home, have the operator align the four marked hub rings
   and confirm. Then run `python3 scripts/calibrate_robot.py capture --replace
   --operator-confirmed --pose-note 'Upper home reached; operator aligned hub rings'`.
   Do not supply that confirmation until it is actually given. Run
   `python3 scripts/calibrate_robot.py status` and report the result. Policies
   stay gated until the current-session hub capture succeeds.
6. Enable only the requested policy. Heating remains manual. An existing valid
   calibration is reused when switching policies within the session.

The normal combined launcher spawns policies inactive. The driver's upper hold
is startup behavior, separate from those policy controllers. Its home acceptance
is within 0.06 rad at low velocity for 20 cycles; low-gain gravity loading can
produce small steady-state error. Wheel capture checks fresh position stability
and finite feedback, with velocity magnitude ignored as the operator requested.

## Installation

After changing code or the target YAML, rebuild the selected checkout:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/local_setup.bash
source ros2_ws/install-roll/local_setup.bash
source ros2_ws/install-combined/local_setup.bash
cd ros2_ws
colcon build --packages-select control_board_hardware_interface pupper_v3_description robot_calibration --build-base build-upper-home --install-base install-upper-home --cmake-args -DBUILD_TESTING=OFF
```

The wrapper refuses the wrong hardware/description/calibration overlay or an
installed target YAML that differs from source. Saving a replacement home must
update both the evidence/profile and the runtime YAML, followed by rebuild.

## Validation and known limitations

The saved-home implementation builds successfully under local ROS Jazzy. The C++
homing checks, all 19 calibration tests, expanded-URDF target/limit checks, and
wrapper confirmation check passed. The robot became unreachable during the new
installation: its new `install-upper-home` build and live saved-home acceptance
check remain pending. Rebuild on the Pi using the command above before first use;
the wrapper will reject a missing or incorrect overlay. Earlier live tests covered
stop homing and return to Stanford zero, not this final saved-home startup path.

Stanford homing retains kp 5.5, kd 0.2, 1.5 rad/s and filtered estimated torque
threshold 2. Added guards bound estimated PD command to 2.5, phase time to eight
seconds, return time to five seconds, and check finite feedback, travel and speed.
These are command estimates, not direct torque measurement. Mechanical resistance
can be mistaken for a stop: an earlier full-sequence FR1 discrepancy remains in
the logs. Subsequent three-run groups across a power cycle had maximum within-boot
spread 0.044 degrees and maximum median change 0.022 degrees. This supports the
operator's decision to use automatic homing; it is not proof against obstruction.

`stanford_upper_homing_test.launch.py` remains an explicitly selected diagnostic
that returns toward zero and disables afterward. It is not normal startup.
Old manual-reference launchers are historical diagnostics, not the default.
