# Wheel-blind lift and slow position alignment

The approved `leg-lift-wheel-position-v1` policy is now selected by
`combined_motion.launch.py` as `neural_controller_wheel_lift`.
It replaces the notebook controller in this launch. Other dedicated notebook
launches remain available.

## Controls

**R2 (right trigger, digital button 7)** follows the current robot-code binding.
First pull activates the policy into **pose**. Release between pulls:

**pose → FR lift → FR align → FR lower → FL lift → FL align → FL lower →
BR lift → BR align → BR lower → BL lift → BL align → BL lower → done**.

A pull before measured readiness is rejected immediately, with no queued step.
An align timeout or lost verified angle holds the lift command and reports a
sequence fault; it never automatically lowers. Reaching done holds the supported
pose. Switching away requires fresh supported pose/lower/done status. X remains
roll, Triangle walking, Circle wheels, and PS emergency stop.

The controller starts inactive with the other combined policies. Use the existing
startup wrapper and current-session calibration procedure in
[STARTUP_UPPER_HOME.md](STARTUP_UPPER_HOME.md). This port has not activated or
installed software on the physical robot.

## Policy and position control

The unchanged approved export is `launch/policy_leg_lift_wheel.json`, SHA256
`20868146985fb5d8dae3e80339f35a5e727c133a1bb851d1cd2e268f8dc33636`.
The included model has the wheeled heating backpack and 9 mm spacer, SHA256
`6efdbaab466e3a0f761db9d643516c159603f4f8019dae38de3dd64d6f98422e`.

The actor receives 27 values: body angular velocity, projected gravity, five-way
stand/leg command, eight proximal position offsets, and its previous eight
normalized outputs. There are no wheel measurements, rotation goals or sequence
phases in its observation. Lift and align use the identical actor command.

Eight outputs control proximal joints. Four independently generated hub position
references approach the nearest encoder winding of calibrated startup home + π.
Reference speed is limited to 0.15 rad/s and acceleration to 0.3 rad/s². Every motor
uses position PD: proximal kp=5/kd=.25, hubs kp=4/kd=.15; commanded velocity and
feedforward effort are zero. Position targets are bounded to reproduce the
training model's ±3 Nm PD saturation. If physical joint limits make that bound
impossible, the controller stops. This is a sampled command estimate, not a
measured torque guarantee.

The actor uses elapsed time to average 50 Hz on the 520 Hz controller manager,
with at most one manager-tick scheduling jitter. Hub references update every tick.
Rotation requires 10 mm conservative floor clearance, 10 mm wheel separation,
5 mm conservative wheel-to-base margin, tilt below .12 rad and angular speed
below .3 rad/s. The gate must stay qualified for .2 seconds. Alignment requires
measured angle/speed settling for .5 seconds. Geometry comes from the exact model;
physical contacts and terrain are not directly sensed.

## Validation

- Local ROS Jazzy build and installed plugin loading passed.
- Real controller-manager test loaded/configured **all five combined policies
  inactive**, activated the new lift controller against GenericSystem mock
  interfaces, checked eight outputs/position gains, and exercised emergency stop.
- Core and ROS lifecycle tests passed: calibrated winding, rejected presses,
  wheel-blind observations, gate loss, timeout, stale sensors and clock faults.
- Eigen RTNeural inference agrees with 128 independent Python export evaluations
  within 9.81e-7 normalized action units.
- The actual C++ core and Eigen actor completed native MuJoCo at **520 Hz**:
  no self-collisions, peak tilt .06698 rad (3.84°), peak drift .07536 m,
  final hub error below .00254 rad (.146°). This is simulation evidence.
- Ten button-dispatch tests passed. Walking-frame/handoff and both roll lifecycle
  regressions passed.

Results and the original training release evidence are under
`policies/leg_lift_wheel`. `upstream_manifest.json` describes the original training
release, including artifacts not duplicated here. `integration_manifest.json`
records the ported source and test evidence. The historical `controller.json`
contains the original release's pre-integration status.

### Existing preparation failures

The broader `scripts/prepare_combined_motion.sh` still rejects the branch's
walking/wheel regression records. Commit `63820b0` replaced walking with seq_d
without updating `policies/latest.json`, the checker assumptions or reference
fixtures. The narrower wheel policy also disagrees with its old inference CSV.
These fail independently of this port; the current weights were preserved.
The script includes the new plugin/tests but cannot report a clean full-stack
preparation until those separate provenance/fixture updates are reconciled.

The local verified overlay is `/tmp/wheel-lift-ros-install` (workstation only).
After resolving those existing records, `scripts/prepare_combined_motion.sh`
builds the new plugin into the Pi's normal `install-combined` overlay. Then run
`python3 scripts/check_wheel_lift.py --package-share
ros2_ws/install-combined/neural_controller/share/neural_controller` to verify it.
