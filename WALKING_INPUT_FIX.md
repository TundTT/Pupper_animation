# Walking input correction

The selected `reference_2026-09-14_06-51-00` (seq_d) export was trained
with absolute model joint angles. Its JSON observation description incorrectly
says to subtract the home pose. The controller previously followed that description.

Source evidence is on the original repository's `leg` branch at
`a5df97c5149245282769a254b0d7ca4a693499aa`:

- `mujoco_playground/workspace/train_reference.py`, `PupperReferenceEnv.__init__`,
  sets `default_pose` to twelve zeros to accommodate actuator bias.
- `Stanford/training/pupperv3-mjx/pupperv3_mjx/environment.py`, `_get_obs`,
  subtracts that zero pose from joint angles.

`launch/config.yaml` now selects the read-only `joint_position_observation:
absolute_model` override for this walking controller. The override rejects other
source runs/behaviors; other controllers keep their existing observation convention.
The policy bytes, motor home, action scale, limits and gains are unchanged.
Policy SHA256: `aa4cd244eb3fd9e0f44a0f0c6f6c48169afe021d0cd26e55a447d92065d5f3e3`.

## Long appendage reference

The operator confirmed that the hub can be installed in three orientations.
Calibration identifies the long appendage by physically pointing it down.
The software cannot recognize appendage geometry from encoder feedback alone.

The existing driver and `walking_frame.hpp` implement two translations:

1. Driver: `calibrated = raw - gravity_offset`, where the offset is measured
   with the long appendage down and the wheel rings at their reference marks.
2. Walking: `policy_angle = calibrated - walking_offset`, where the fixed
   offset retains the captured hub home and any complete hub revolutions.
3. Position command: `raw_target = policy_target + walking_offset + gravity_offset`.

The fix removes only the extra subtraction of motor home from `policy_angle`.
It does not remove either calibration translation. A full revolution preserves
the long appendage; 120 degrees does not. Walking entry rejects a hub displaced
120 degrees from its captured reference. Initial capture still requires correct
operator positioning. Startup and capture prompts now state this explicitly.

## Validation scope

The local native MuJoCo comparison used the exact export and source-model hash
`1ad0794fd9d5c1b517f15b36c74184c4932561e5841089bb0a87b65a9c1cbd7e`.
At a 0.20 m/s forward command, the old input failed after 0.66 seconds.
Changing only the joint observation produced 10 seconds without falling and
0.212 m/s mean forward speed after warmup. This deterministic simulation omits
hardware latency/noise; it has lateral drift and is not physical validation.
Local artifacts are in `output/walk-policy-review/` (not versioned).

Regression tests cover all 81 combinations of three mounting orientations across
four hubs, complete encoder revolutions, calibration-to-command round trips,
120-degree entry rejection, and the actual walking plugin's absolute observations
with gravity-only and roll-handoff references. Software changes do not establish
physical calibration accuracy. Hardware deployment and supervised testing remain
separate from this local correction.

Local verification on September 15: ROS Jazzy Release build passed; all 14
neural-controller CTest groups passed, including actual-plugin walking handoff
and mock controller-manager tests. All 20 calibration Python tests passed.
Import/hash verification and `git diff --check` passed. Build/test logs are in
WSL `/var/tmp/quadmorph-gravity-20260915/walk-fix-{build,tests}.log`.
