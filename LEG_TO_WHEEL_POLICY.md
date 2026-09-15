# Trained leg-to-wheel policy

The approved simulation policy and gentler lowering are available on `robot-code`
as an exported RTNeural network and a tested, ROS-independent C++ runtime.
This is a **source release**. A ROS controller/joystick adapter, validated
clearance/contact inputs, and physical validation are still required before
hardware activation. The existing startup/calibration requirements in
`STARTUP_UPPER_HOME.md` apply to that future integration.

## Files

- [Policy export](ros2_ws/src/neural_controller/launch/policy_leg_to_wheel.json)
- [C++ policy runtime and sequencer](ros2_ws/src/neural_controller/include/neural_controller/leg_to_wheel/policy.hpp)
- [Release manifest](policies/leg_to_wheel/manifest.json)
- [Demo: FL → FR → BR → BL](policies/leg_to_wheel/demo.mp4)
- [Simulation results and known failures](policies/leg_to_wheel/SIMULATION_RESULTS.md)

The bundled checkpoint, saved configuration, training-source snapshot and Python
inference references are under `policies/leg_to_wheel/`. Export and runtime
fixtures are retained, so checking the release does not require the original
training checkout or contact with the robot.

## Check the release on a development machine

After fetching the branch, run `git lfs pull` to retrieve the JSON files tracked
through the repository's existing Git LFS rules.

```bash
python3 scripts/leg_to_wheel/check_release.py
bash scripts/test_leg_to_wheel_policy.sh
```

The C++ test requires a C++17 compiler and uses the repository's RTNeural Eigen
backend. On this workstation:

```bash
micromamba run -n ros_jazzy bash scripts/test_leg_to_wheel_policy.sh
```

It checks 128 network fixtures against the trained JAX actor, 196 sequential
observation/action/target fixtures against Python, reset history, command mapping,
clearance gating, a >60-second operator hold, explicit heating confirmation,
settled touchdown and the final converted bitmask. The test is also registered in
the ROS package's CMake tests. A standalone CMake project is available under
`scripts/leg_to_wheel/`.

## Runtime interface

`leg_to_wheel::Policy` loads the export. Call `reset()` before a new activation,
then call `step(gyro_body, gravity_body, q_model, command, capsule_clearance, dt)`
at **50 Hz** (`dt=0.02`). It returns the raw neural action, eased actuator action,
and absolute joint-position targets in the trained model frame. Gains, scales,
home and joint limits are exposed from the export. The existing calibrated
hardware coordinate conversion remains the responsibility of the future adapter.

The observation is four newest-first frames of 35 values: angular velocity (3),
projected gravity (3), one-hot command (5), joint position minus home (12), and
last raw policy action (12). Reset repeats the first measured frame four times.
Normalization is folded into the export; do not normalize or clip observations
again. All 12 outputs are position actions.

Command indices are `0=stand, 1=FL, 2=FR, 3=BR, 4=BL`. Actuator rows and converted
bits are in `FR, FL, BR, BL` order. The neural policy controls every stance leg,
including previously converted legs; no fixed converted-leg pose is imposed.

`leg_to_wheel::Sequencer` owns operator requests and bookkeeping. `request(1..4)`
starts a lift. Supply the active foot's actual capsule-bottom world-floor
clearance and contact state to `update()`. At least 15 mm clearance with no contact
must persist for 0.3 seconds before heating can be confirmed. Heating is manual
and the hold has no timeout. Explicit confirmation commands stand/lower; contact
must then remain stable for 0.3 seconds before the leg is marked converted and
another request is accepted. Sensor feedback must be refreshed independently of
the network outputs; a predicted joint target is not a contact measurement.

The hardware has no direct foot-contact input in this release. Do not substitute
the old script's foot-relative height estimate for the simulator's world-floor
clearance. The adapter must establish trustworthy clearance/contact feedback and
handle missing/stale data through the robot's existing authority controls.
The export uses a distinct `leg_to_wheel` behavior tag; it is not a drop-in
replacement for the legacy `leg_lift` controller's policy JSON.

## Gentler lowering and measured limits

The inference adapter eases only downward motion of the returning leg's lifting
hip (`_2`). It starts with an 8 rad/s target-change limit, smoothly releases over
0.2 seconds, and also fades between 40 and 20 mm capsule clearance. Below 20 mm,
the policy fully owns touchdown. Support joints, abduction, knee/hub and upward
hip corrections remain responsive. Raw neural actions stay in observation
history; previous actuator targets are tracked separately.

The demo's peak downward foot speed decreased **26%**, from 0.676 to 0.498 m/s.
Touchdown remains 0.038 m/s, front-lift tilt 6.43°, and sag 8.23 mm. All **195/195
nominal** cases pass; **179/195 randomized** cases pass. Both protocols include
three seeds, 60-second holds and every subset of 0–3 shortened support limbs.
All 24 long-hold cases pass and all 390 cases have zero falls. Randomized
acceptance remains incomplete: 11 touchdown, 2 tilt, 2 stance-contact and 1 sag
failures. Before easing, randomized acceptance was 182/195.

These are validation samples used during tuning, not independent held-out or
hardware results. Shortening upstream limb reach by 5–10 mm is a simulation
proxy for conversion; thermal deformation is not simulated. The evidence files
retain every failure and the original thresholds.

## Re-export

With the existing JAX/Brax training environment:

```bash
JAX_PLATFORMS=cpu python scripts/leg_to_wheel/export_policy.py \
  --params policies/leg_to_wheel/checkpoint/params_000093716480 \
  --output /tmp/policy_leg_to_wheel.json \
  --fixtures /tmp/leg_to_wheel_reference.csv
```

The saved checkpoint's config is authoritative. The original training checkout's
later experimental defaults are not used for this export. The manifest preserves
model/ring hashes and the checkpoint/source provenance; no model asset shipment
or physical robot deployment is implied by this release.
