# Stanford/gravity runtime validation — 2026-09-15

## Result

ROS 2 Jazzy Release build passed for all five packages in WSL Ubuntu:
`robot_calibration`, `control_board_hardware_interface`, `pupper_v3_description`,
`cmd_vel_mux`, and `neural_controller`.

Final `colcon test-result` report: **61 tests, 0 errors, 0 failures, 0 skipped**.
This aggregate includes CTest wrappers and their nested pytest cases; it is not
61 independent hardware trials. The configured suites comprise 18 CTest groups.

## What was exercised

- Actual hardware lifecycle linked to in-memory SPI, with IMU device access disabled:
  refusal without confirmation, zero gains/velocity/effort during startup, fresh
  gravity offsets including multi-turn hubs, new activation identities, one-use
  confirmation, no automatic shared capture, and invalidation on lost feedback.
- Real Stanford transport code with wrapped ioctl: failed/short transfers, empty
  packets, bad checksums and nonfinite data are rejected; fresh valid data recovers
  the transport validity indicator. No spidev device was opened by this test.
- Gravity sample window: stationary acceptance, offset calculation, moving-joint
  rejection, bad feedback, gaps and nonfinite inputs.
- Shared calibration storage and ROS capture against a fake manager/encoder publisher.
- Installed xacro expansion: all 12 gravity references, unique motor addresses,
  required command interfaces, continuous hub limits, resolved meshes and package
  markers. Direct startup without this boot's confirmation and noninteractive
  wrapper use are refused before creating hardware nodes.
- Original policy inference, roll core/lifecycle, walking frames/handoff,
  lift core/lifecycle and button dispatch regressions.
- A real ROS controller manager using only `mock_components/GenericSystem`:
  all five selected motion controllers load/configure inactive; lift/align produces
  eight policy outputs and the expected proximal/hub PD gains; e-stop zeros gains
  and effort. This is interface validation, not simulated or physical balance.

## Fixture corrections

The inherited calibration movement test changed reported velocity while keeping
positions fixed, conflicting with the existing position-stability capture policy.
The fixture now actually moves a joint; a separate case checks fixed positions
with velocity noise. Runtime capture behavior was not changed.

The first mock-manager run failed lift entry because GenericSystem immediately
echoed the activation's zero position commands into sensors despite zero gains.
That is not position-PD physics. The fixture now uses `disable_commands=true` to
hold its nominal sensor pose and checks command values on the actual controller's
motor-command telemetry. The initial failure is retained in the local validation
logs. The focused rerun and final complete controller suite passed. Production
controller code and policy exports were not changed to make the fixture pass.

## Reproduction and evidence

Use the build/test commands in README.md. This run used:

- Build: `/var/tmp/quadmorph-gravity-20260915/build`
- Install: `/var/tmp/quadmorph-gravity-20260915/install`
- Build log: `/var/tmp/quadmorph-gravity-20260915/build.log`
- Original test log: `/var/tmp/quadmorph-gravity-20260915/tests.log`
- Final controller log: `/var/tmp/quadmorph-gravity-20260915/final-neural-tests.log`
- Aggregate result: `/var/tmp/quadmorph-gravity-20260915/results.txt`

These paths are local WSL artifacts, not robot installation paths. Build output
retains upstream deprecation/vendor warnings; this is not a warning-free audit.
`scripts/verify_import.py` verifies original import records, documented adaptations,
Stanford file hashes and unchanged selected policy hashes.

## Remaining work

No physical robot was contacted, started, calibrated or moved. Pi installation,
actual disabled-motor feedback, IMU timing, physical reference accuracy and motion
validation require the later supervised hardware procedure. SPI transaction validity
does not prove per-motor firmware sample freshness. The Stanford visual model is
not a validated custom-geometry simulation model.

The leg-to-wheel ROS adapter and its live clearance/contact feedback remain the
next task, explicitly deferred by the user. Its policy and core runtime are kept
but are not exposed as a runnable controller by this launch.
