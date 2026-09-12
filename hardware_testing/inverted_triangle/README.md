# Inverted-triangle controller preparation

Prepared on `robot-code`, starting from `896b2879ed4365478960fce759ff952ca9a60f77`.
Hardware compatibility review used `origin/robot-info` at
`e9b04173b034d595a85e6147d73a45cdfe9393e3` and the actual current robot-code writer,
description and calibration implementation. No robot was connected, started,
calibrated, heated or moved during this preparation.

Use [INVERTED_TRIANGLE_LAB.md](../../INVERTED_TRIANGLE_LAB.md) for the physical
sequence, commands, stops and source distinctions.

## Motion and controller

- Immutable parent simulation: `8904c2a836951250317300b2722e36376a810a50`.
- Original plan: `204c0a54172874749b6702e2802872b265489bffc18ce9b20bb08b2bde41e369`.
- Original rigid-triangle model: `768cbeb0bfbaab0c898e6078d0b778ee2968718d98bdb2857e13538daeb607a8`.
- Original tip and **9 mm additional axial gap** retained.
- Export: 38 quintic segments, 67,278 steps at 520 Hz, about 129.381 seconds of
  trajectory, excluding supported arming and operator pauses.
- Order and hub winding: rear-right -pi, rear-left +pi, front-right -pi,
  front-left -pi. Rear-left's extra final revolution is preserved.
- Source plan/candidate bytes and exported-plan hash are in [source/](source/).

The new plugin uses the same targets and PD gains, with explicit one-leg commands,
planted pauses, sensor/tracking/timing checks, a supported gain ramp and latched
torque-release faults. A separate current-session triangle reference prevents
using model hub angles directly against arbitrary encoder homes. It preserves
the shared calibration and does not infer proximal offsets.

No neural network, contact estimator, MuJoCo or GPU runs on the Pi. The existing
position/velocity/effort/kp/kd interfaces are sufficient. Robot-info's older leg
hub hard stops conflict with this motion; the current robot-code continuous-hub
profile is the applicable implementation. No legacy limit was silently removed
as part of this change.

## Verification

Local x86 Ubuntu/ROS Jazzy clean build and incremental preparation completed for
six packages. The installed controller library was selected explicitly for tests.
Eight selected controller/launch tests, three calibration test groups, one joystick
test group and four triangle-reference Python cases passed. The source and
installed-asset compatibility checks passed. These retain the existing keyframe,
wheel, walking and hybrid regression checks relevant to the shared library.

The C++ motion core was additionally connected to actual MuJoCo encoder feedback
through the test-only bridge in `inverted_triangle_test.cpp`. Every commanded
target was compared with the original Python trajectory. Two full four-leg
replays, with **five-second and one-minute planted pauses**, passed all original
single-flip physics/CAD gates at every stage. All 67,278 trajectory targets in
each replay agreed within **3.491e-13 rad**.
The added pauses retained four floor supports, no motor/body floor support and
positive CAD clearance. See the saved runtime audit under [evidence/](evidence/).

The original 60/60 robustness result belongs to the parent continuous simulation.
It is not presented as 60/60 hardware-controller trials. The added arming ramp is
performed under external support and is not covered by the free-standing replay.
Feedback settling extensions beyond nominal and fault recovery are not accepted
trajectories; recovery is deliberately not implemented.

Preparation caught and corrected a too-strict *new* settling guard that treated
all supporting hubs like the active unloaded hub. Parent traces contain about
0.103 rad peak loaded hub deflection; the tight 0.035-rad check now applies to the
active limb, with separate 0.15-rad supporting-hub tracking protection. The parent
geometry, targets, gains and acceptance gates were unchanged. The failed guard
audit is retained. An initial local test invocation also lost the ROS library path
through shell argument expansion; the file-based check script fixes the invocation
and the passing run uses the installed library. That failed log is retained.

The ARM64 clean build completed for all six packages. All seven selected C++
controller tests and all five launch/parser tests passed. C++ calibration storage
passed. The Python storage/sampler suite exceeded CTest's 60-second allowance;
an unchanged rerun with a 180-second **test-process** allowance passed all 18
cases in 91.14 seconds. No runtime timeout or calibration gate was changed.

**The full ARM64 shared integration suite is not green.** The three ROS capture
tests reject their fake sessions, and the two legacy joystick dispatch tests
cannot discover the other process within their eight-second test deadline.
Increasing only the outer joystick test-process allowance preserved that failure.
These shared tests pass in the local native x86 environment; they still need a
native Pi check. The new triangle trial does not bind those legacy activation
buttons, but actual gamepad stop/discovery must still be verified on the Pi.

The standalone [process-identity probe](arm_environment_probe.py) reproduces an
emulation discrepancy: a process reads its start tick as `16829728`, while a child
reading that same PID sees `16829727`. The shared calibration's exact process
identity check consequently rejects such a fake cross-process session. We did
not weaken that production check or fabricate a valid session to pass the tests.
The joystick discovery failure is retained separately; it is not claimed to have
the same diagnosed cause. See the original and diagnostic logs in [evidence/](evidence/).

The QEMU image is `quadmorph-keyframe-arm64:build-aa62561`, immutable image ID
`sha256:6c050b006e2e4ad273dd0ff260b94f89ed5e1a3cf22f4683c739550673f4a98f`.
Package inventory and installed plugin hash are saved with the evidence. This is
an architecture check; Ubuntu 24.04/GCC 13/Python 3.12 does not reproduce the Pi's
Debian 12/GCC 12/Python 3.11 image exactly or prove its scheduler and SPI behavior.

## Reproduce

On a ROS Jazzy host, without hardware startup:

```bash
bash scripts/prepare_inverted_triangle.sh
```

To regenerate the immutable export, use the source checkout at the exact parent
commit and its pinned simulation environment:

```bash
python scripts/export_inverted_triangle.py --source /path/to/reviewed-8904c2a-checkout
```

For the physics comparison, compile the test-only bridge and run from the
robot-code repository, using the parent's pinned environment:

```bash
g++ -std=c++17 -O2 -shared -fPIC -DTRIANGLE_BRIDGE \
  -Iros2_ws/src/neural_controller/include \
  -Iros2_ws/src/neural_controller/modules/RTNeural/modules/json \
  ros2_ws/src/neural_controller/test/inverted_triangle_test.cpp \
  -o /tmp/inverted_triangle_core.so
python hardware_testing/inverted_triangle/audit_runtime.py \
  --source /path/to/reviewed-8904c2a-checkout \
  --library /tmp/inverted_triangle_core.so \
  --output /tmp/new-inverted-runtime-audit --pause-seconds 5
```

This runs no policy training and does not upload to W&B. The existing reviewed
video is linked in the parent report. Use a real Linux checkout for Linux Git
provenance; a Windows worktree's `.git` pointer is not directly portable to WSL.

## Remaining target and physical checks

1. Preserve Pi-local changes, prepare the selected checkout while its stack is
   stopped, and run native build/parser/installed-overlay checks. Source work here
   is not deployment to `/home/pi/robot-code-leglift`.
2. Verify scheduling, live loop timing, the actual joystick/stop mapping and sensor
   freshness on the intended Pi. ARM emulation does not establish these.
3. Follow the existing startup confirmation/capture procedure, then physically
   confirm and capture the separate inverted-triangle starting reference.
4. Inspect unpowered clearances and supported motion, then the first loaded
   rear-right flip with a collapse-catching support. Proceed one limb at a time
   after reviewing the preceding result.

The roughly 2.72 mm limiting simulated shin/motor gap still needs physical
verification. No automatic walking-policy handoff, heating control, interrupted
flip recovery or automatic approach from the saved hanging pose is included.
