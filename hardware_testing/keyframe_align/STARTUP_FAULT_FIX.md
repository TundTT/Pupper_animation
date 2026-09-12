# Correction after the September 11 oscillation incident

The incident release e6192ab remains withdrawn. This candidate corrects a
reproduced software fault; it does not establish the physical cause of the later
oscillation or prove motor stability. No hardware startup or motion is part of
the software verification described here.

## Changes

- The first update with period zero establishes the clock and requests zero
  torque without advancing the trajectory. Only that first zero is accepted.
  Later zero/negative/over-40-ms periods, reversed/repeated timestamps and clock
  gaps over 40 ms still latch a fault. Sensor and calibration checks remain.
- Keyframe fault, activation failure, bootstrap and deactivation request zero
  position, velocity, feed-forward effort, kp and kd. The inherited deactivation
  path uses keyframe-specific estop_kd=0. Other policy stop settings are unchanged.
  This is a zero-torque request, **not motor power isolation**. The robot loses
  active support on stop and must be physically supported for initial testing.
- The hardware writer and homing logic are unchanged. The selected wheel-profile
  xacro has no finite hard-limit overrides on any joint; the source/installed
  preflight now explicitly checks this, because those overrides can synthesize
  gains. Source: `copy_actuator_commands` in the hardware interface and
  `components.xacro`. These are code/configuration facts, not proof of motor
  current accuracy or electrical disable.
- Fault reasons are latched and logged by an executor timer outside the motor
  update callback. Faults do not silently clear with another leg request.
- Status retains its original first 25 entries, then adds fault code [25], supplied
  period in seconds [26], and IMU age in seconds [27] (-1 when unavailable).
  Codes: 0 none, 1 clock, 2 period, 3 encoders, 4 IMU, 5 IMU age, 6 tilt,
  7 invalid command/core fault, 8 operator stop, 9 activation failure.
- `/neural_controller_keyframe_align/motor_commands` records 60 requested values
  at 20 Hz: canonical FR/FL/BR/BL, three joints per leg, five values per joint in
  position/velocity/effort/kp/kd order. This records controller interface requests,
  not measured torque or post-clamp motor-board commands.

## Verification

The revised actual-plugin fixture exercises first period zero, all four wheel
targets, timing failures after startup, stale sensors, invalid commands, failed
activation and deactivation. It verifies zero gains on fault paths.

`keyframe_manager_test.py` runs the actual installed ROS controller manager with
only `mock_components/GenericSystem`, in ROS domain 181 with private synthetic
calibration. It cannot load the physical board plugin from its generated URDF.
It verifies successful entry through the manager, invalid-command fault and
zero outputs, ignored requests while faulted, deliberate reactivation, and
operator stop. It checks both mock hardware state and requested command telemetry.
The preparation script includes this test; a fixture-only pass is insufficient.

Final runtime source: **b9e2214**. All **13 selected CTests passed** on the native
Pi in both `/home/pi/keyframe-software-test` and the selected physical-trial
checkout `/home/pi/robot-code-leglift`. Each includes the actual manager test,
actual plugin fixture, calibration, joystick and legacy regressions. All six
packages built and the installed overlay/source/configuration checks passed.
The selected checkout retained its three local hardware/launch edits exactly
(before/after patch SHA256 b48b4fae72fb3d56899fc24ed0870a9b605fce6690d5d6b24c5b371d3490434c).

Selected Pi installed plugin SHA256:
`ef6d5d298c4ecbe8154166fe9fe4bffdba82d24c1623ee1e9972d7bcf9631243`.
The separate Pi test checkout's library SHA256 is
`8ee48a0f998d686ca9d00404d17374729f317734e5b2e5944f5e37a76f490209`;
separate installation/build paths produce distinct binaries, both tested against
their own installed overlay. Pi versions remain controller-manager 4.25.0 and
realtime-tools 3.3.0 on Debian 12 ARM64.

The two new/revised tests also passed locally on native x86-64 after rebuilding
the final source. An earlier local run used an outdated installed library during
ongoing edits and failed (fixture crash and missing new telemetry); its output
is preserved and is not counted as a pass. The frozen native Pi builds and final
local rebuild resolved that build-state mismatch. No production threshold was
relaxed to obtain the passing results.

Evidence: `/home/pi/keyframe-incident-evidence/final-fix-prepare.log`,
`robot-final-fix-prepare.log`, `final-plugin-hashes.txt`; copied locally under
`C:/Users/tundt/Desktop/quadmorph-lab-recordings/keyframe-incident-20260911/`.
The motor stack is stopped and the automatic D-pad launch service is disabled.
This revision is prepared for the supported check below; physical stop behavior
remains unverified. Passing mock tests is not clearance for an unsupported
all-joint trial.

## First physical check

1. Keep automatic D-pad startup disabled. Select the reviewed checkout/overlay;
   confirm its installed plugin and config match the tested release. Keep the
   robot fully supported so it cannot fall when holding torque is removed. Do
   not twist powered wheels to probe resistance.
2. Follow STARTUP_CALIBRATION.md: get the operator's explicit current homing/ring
   pose confirmation, then start with motion inactive and capture/status. A prior
   confirmation or file from before power-off does not apply.
3. Start recording joint states, IMU, joystick, status, requested motor commands
   and rosout. Press X once only for the supported entry check. Verify ENTRY then
   HOLD, authority true, fault code zero and bounded commands. Do not send a leg
   request if startup does not behave as expected.
4. While fully supported, request PS stop and verify requested gains/effort are
   zero and the controller is inactive. Confirm the robot's physical response
   is stable without touching powered joints. Cut motor power if instability
   occurs. Telemetry alone does not establish the stop's physical behavior.
5. Only after this check is reviewed should a separately agreed single-wheel
   sequence proceed. Heating remains manual. Record any issue before retrying;
   do not immediately escalate to a full sequence or change gains in the lab.

These steps require operator observation. They have not been completed by the
software tests, and this document does not authorize an unattended hardware run.
