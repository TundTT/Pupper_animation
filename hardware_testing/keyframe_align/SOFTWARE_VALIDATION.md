# Keyframe robot integration validation

Motion source: alignment branch a110b09. Adapter base: robot-code e3e1d97.
The copied core/configuration is checked by source_manifest.json (C++ header
hashes normalize CRLF to LF; JSON uses raw bytes for cross-platform replay).
The final integration commit is recorded by Git history for this file.

## Local x86-64 ROS check

An isolated six-package build passed for robot_calibration, neural_controller,
joy_utils, control_board_hardware_interface, pupper_v3_description and
animation_controller_py. The build/install directories are `/tmp/keyframe-robot`
inside WSL. A subsequent incremental build/install and the installed preflight
passed. Existing vendor RTNeural/hardware warnings remain; they are not new
keyframe behavior failures.

All 12 selected CTests passed: eight controller/launch/regression tests, three
calibration tests, and the joystick test. The latter is parameterized for both
legacy hybrid and keyframe controller/topic names. The new actual plugin test
covers all four wheel mappings, calibrated target retention, startup holds,
reentry after many revolutions, missing/stale calibration, stop, stale IMU and
invalid commands. It asserts that no neural model is loaded. Launch tests exercise
the installed spawner parser with motion inactive and calibration required.

The first new-plugin test exposed missing action_scales metadata inherited from
the old network loader. Explicit identity scales were added to the new YAML. The
final plugin test and installed preflight pass. A cached CMake configuration had
initially omitted the new test; force configuration corrected this, and the new
test was run explicitly. Neither earlier result was counted as an adapter pass.

Local package versions: controller-manager/controller-interface/hardware-interface
4.44.0; realtime-tools 3.11.0; ROS Jazzy on Ubuntu Noble x86-64.
Installed plugin SHA256:
`37bc8b684efdc5cec95f9d9aec340a6a804e1b7b0e9cd7298da6b09259561369`.
This hash identifies this architecture/build, not a binary to copy onto the Pi.

## ARM64 architecture check

The standalone core compiled statically with GCC 13 for AArch64 and passed its
full-sequence/final-target/stop test under qemu-aarch64. This does not measure Pi
realtime timing.

The fuller ROS image recipe is [Dockerfile.arm64](Dockerfile.arm64), with the
software-only build/test command in [arm64_check.sh](arm64_check.sh). It uses an
ARM64 ROS Jazzy Noble container under QEMU, with source mounted read-only and a
private source/build copy. The base image digest is
`sha256:386d06ec6d4188f731bae5678e07b4cb64a4e4d4152090c0bd1f881dcf7706f5`.
The full four-package ARM64 build passed for robot_calibration, neural_controller,
joy_utils and pupper_v3_description (22 minutes under emulation). Seven selected
controller/regression CTests passed, including the actual keyframe controller
fixture. The launch-parser CTest initially timed out at 60 seconds; a direct run
with pytest plugin autoload disabled passed all three identical assertions in
22.18 seconds. Calibration storage and Python tests also passed.

**The ARM64 suite is not fully passing.** The calibration ROS test hit its
60-second harness timeout and the joystick integration test hit 45 seconds. A
direct calibration rerun completed in 40.24 seconds but failed all three cases:
the child CLI rejected the fixture's synthetic encoder session as not live/homed,
before it could exercise the expected capture/controller/stationarity behaviors.
The cause of this emulated cross-process session rejection is unresolved; it is
not evidence that physical encoders or wheel feedback failed. Both integration
test groups pass on native x86-64. They must pass on native ARM64 before this
candidate is cleared for a physical trial. No production session checks,
freshness thresholds or safety gates were relaxed to obtain a pass.

ARM64 package versions: controller-manager/controller-interface 4.48.0,
realtime-tools 3.12.0, ROS Jazzy on Ubuntu Noble. These differ from the local x86
versions and have not been matched to the Pi. Do not deploy these container
binaries to a different target distribution. The preserved built image is
`quadmorph-keyframe-arm64:build-aa62561`, image ID
`sha256:6c050b006e2e4ad273dd0ff260b94f89ed5e1a3cf22f4683c739550673f4a98f`.
Runtime controller sources match integration aa62561. The additional pluginlib
discovery assertion was added to the x86 fixture after the ARM snapshot; that
assertion passed against the explicitly selected installed x86 library.

Local evidence is retained in WSL `/tmp/keyframe-arm-check` (build, install and
test logs), `/tmp/keyframe-arm-check.log`, `/tmp/keyframe-arm-final-tests.log`,
and `/tmp/keyframe-arm-capture-direct.log`. Copies of the summary logs are saved
under `C:/Users/tundt/Desktop/quadmorph-lab-recordings/keyframes-a110b09/robot-integration/`.
The ARM script now gathers build metadata before testing and runs every test
group, returning failure if any group fails; it does not silently skip the
unresolved graph tests. No further neural training is needed for this controller.

## Native Pi verification — September 11, 2026

The Pi subsequently became reachable at its recorded address. Integration
`14948015f23c0e474867cb13f4ecdf026185c3e6` was checked out separately at
`/home/pi/keyframe-software-test` and built natively with its motor stack stopped.
Target: host `pupper`, AArch64, Debian 12 Bookworm, ROS Jazzy; controller-manager,
controller-interface and hardware-interface 4.25.0, realtime-tools 3.3.0,
generate-parameter-library 0.4.0. This directly tests the target's older packages.

**All 12 selected CTests passed natively:** three calibration tests (including
the ROS capture fixture), the joystick test (both legacy and keyframe cases),
and eight controller/launch/regression tests. The actual plugin discovery and
lifecycle fixture passed. The emulated calibration/joystick failures did not
reproduce; no production gate or test assertion was relaxed. These tests use
synthetic calibration/encoder fixtures, not physical motor data.

All six preparation packages built in 3 minutes 21 seconds. A subsequent
incremental build/install passed in 3.99 seconds, and the installed overlay,
geometry/configuration and source-manifest checks passed. Installed plugin SHA256
was unchanged across both builds:
`baec07d3872e0c328763385c6715995c6a12176d7330924fa0ccbcaf42363ede`.
Existing compiler/vendor warnings remain.

Evidence is saved on the Pi in `/home/pi/keyframe-validation-1494801` and copied
locally to `C:/Users/tundt/Desktop/quadmorph-lab-recordings/keyframes-a110b09/robot-integration/pi-1494801/`.
It includes inventory, complete preparation/test output, incremental build output,
plugin hashes, and the preserved original checkout patch. The original
`/home/pi/robot-code-leglift` remains at e3e1d97 with its three local modifications
unchanged (before/after patch SHA256 matched). The isolated candidate does not
incorporate those local homing/optional-node changes; review them before selecting
the eventual physical-trial checkout. No hardware stack, homing, motor command,
calibration capture, service or scheduling change was performed.

## Still requires physical-session preparation

Inventory found no gamepad device and realtime-priority limit 0 for the SSH user;
the legacy D-pad launch service was active. Resolve device access and scheduling,
select the physical-trial checkout/overlay, and preserve the existing robot's
local edits before deployment. Follow KEYFRAME_LAB.md and STARTUP_CALIBRATION.md
for explicit physical-pose confirmation, fresh startup calibration and a
supervised single-wheel trial. Physical geometry, ring marks, encoder zeros,
load response, transport freshness and realtime timing are not established by
the passing software tests.
