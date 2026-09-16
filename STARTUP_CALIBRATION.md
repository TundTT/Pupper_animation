# Gravity-pose startup

## Physical reference

The operator supports the robot, lets the legs hang freely, and puts the marked
wheel rings at the agreed calibration references. Keep the robot stationary.
This positioning must be confirmed for **each fresh hardware activation**.
A request to test/start is not physical confirmation. Heating remains manual.

The editable nominal angles are in [config/gravity_pose.yaml](config/gravity_pose.yaml).
They come from `robot-code` commit `6b07745`, `components.xacro`'s existing
`homed_position` values. They are model-frame reference angles, not persistent
raw encoder readings. The previously saved September 12 hanging-pose snapshot
was a measured pose in its old session; it is not used as a new encoder offset.

## Startup on the robot

1. Inspect existing processes and the sourced overlay. Do not start a second
   hardware manager or another robot's checkout. If this hardware session is
   already running, inspect calibration status and reuse a valid capture.
2. Source ROS Jazzy and this branch's built `install/setup.bash`.
3. From this checkout, run `python3 scripts/start_robot.py`. It prints the selected
   package paths, checks bundle markers/ownership, and prompts for the supported
   gravity pose and wheel marks. Type `READY` only after the setup is complete.
   Agents must ask the operator and wait for actual confirmation before starting
   the hardware stack; they must never invent or pre-answer this confirmation.
4. Leave the supported reference pose unchanged. After startup, in another sourced
   terminal run `python3 scripts/calibrate_robot.py capture`. This prompts again
   before saving the shared record. An agent may supply `--operator-confirmed`
   only after the operator actually confirms this startup's physical setup.
5. Run `python3 scripts/calibrate_robot.py status`. Report the calibration ID,
   wheel homes and storage path before enabling requested policy motion.

All six ROS motion-controller instances start inactive. The physical calibration
gate remains enabled, including when controller-manager services are used directly.
Calibration does not authorize deployment, training, heating or additional motion.

## What the driver does

Stanford's torque-threshold homing and post-homing movement are replaced by:

1. Consume this process's one-activation confirmation and create an unready session.
2. Clear command position/velocity/effort/gains and sample while motor enable flags
   are zero. No search for a mechanical stop or pose-restoration movement occurs.
3. Require successful responses from both SPI boards, correct checksums, finite
   readings and nonempty feedback. Require at least 50 samples over one second
   with at most 0.002 rad position excursion on every joint and no gap over 0.2 s.
   Timeout after 15 seconds. Reported velocity is not the stillness criterion,
   consistent with the existing operator-confirmed position-stability capture.
4. Set `offset = measured_raw - gravity_reference` for each joint, preserving hub
   winding. Measurements use `model = raw - offset`; commands use `raw = model + offset`.
5. Enable motor communication with all gains/velocity/effort still zero, check
   feedback, save the actual offsets and mark the session ready. Policy commands
   remain gated until the separate operator-confirmed calibration capture.

The SPI protocol has no per-motor sample counter in this driver. A valid transfer
and checksum do not prove that each motor's firmware refreshed its measurement.
The fake-transport tests establish software behavior, not physical communication
or reference accuracy. Validate disabled-motor feedback and the real pose on the
robot during the later supervised hardware test.

## Storage and invalidation

The shared directory is `$XDG_STATE_HOME/quadmorph`, otherwise
`~/.local/state/quadmorph`; `QUADMORPH_CALIBRATION_DIR` can select an absolute path.
Use the same directory for the driver, controllers and capture command.

- `encoder-session.json`: boot/process/activation identity and readiness.
- `gravity-offsets.json`: actual offsets and the session that produced them; audit only.
- `calibration.json`: current operator-confirmed joint references and wheel homes.
- `history/`: preserved captures and archived mode mappings.

Deactivation, hardware error or a new activation invalidates the prior session.
Old files cannot authorize a new session. No persistent cross-power-loss mapping
or automatic physical homing is claimed. Never disable the calibration gate on
the real hardware or replace measured values with guesses.

## Roll preparation is separate

Gravity calibration alone does not establish a rigid tips-up triangle pose.
With the robot supported, motion controllers inactive, and the operator-confirmed
rigid tips-up configuration prepared, run:

```sh
python3 scripts/inverted_triangle_reference.py capture --roll
```

This retained read-only helper checks stationary feedback, the calibrated proximal
frame and the selected roll plan, then saves a session-bound mapping. It never
moves the robot or overwrites gravity calibration. X remains unavailable until
this preparation is valid. Walking, wheels and lift/align retain their existing
entry-pose checks and fault behavior.

## Source distinctions

- User-confirmed procedure: supported, freely hanging legs and manually positioned
  wheel rings, confirmed in this task.
- Numerical reference/mapping/limits: `robot-code`'s committed component parameters,
  pinned in `config/gravity_pose.yaml` and `STANFORD_IMPORT.json`.
- Infrastructure baseline: Stanford `main` at `6f96c5e79faa05492992c19918f8cd90b9243281`.
- Implementation: `control_board_hardware_interface.cpp`, `gravity_calibration.hpp`,
  `rt_spi.cpp`, the shared calibration package and `gravity_runtime.launch.py`.
- Physical installation and validation: pending. No robot was contacted by this task.
