# September 12 locomotion preparation

## Next trial: wheel command increased to 0.40 m/s

The operator reported both new policies were "not bad", with the wheel policy
feeling slow. The tested joystick cap was 0.20 m/s; no actual speed-tracking
measurement was obtained. After the robot was turned off, the operator requested
0.40 m/s for the next trial. The focused launch now supplies a dedicated
`/wheel_cmd_vel` stream at twice the original forward/back command; walking
continues to consume `/cmd_vel` at 0.20 m/s. Both share the existing timeout.
Yaw remains 0.40 rad/s, and weights, motor gains and the saved pose are unchanged.
This update requires a native rebuild and has not been physically tested.

Local WSL Ubuntu / ROS Jazzy verification passed after the speed change: all seven
packages built, all ten selected CTest suites passed, and source/installed policy
hash checks matched. The joystick fixture verifies both command streams in both
directions and timeout-to-zero; the real controller manager with GenericSystem
verifies each policy observes its own command topic across switches and responds
to emergency stop. Evidence: `wheel-speed-preparation.log`. This is PC software
verification, not a new Pi or physical trial.


## Confirmed startup and replacement home

After minor repairs, the operator explicitly confirmed the robot was hanging on
its stand in the desired starting pose. The focused hardware launch started at
07:55:50 UTC; all twelve joints completed the installed homing procedure. Both
policies configured inactive. The control-loop thread was FIFO priority 50.
Startup evidence: `native_pi/startup-025550.log`.

The first 15-second and subsequent 45-second captures failed the stationary-data
gate. The operator reconfirmed hanging, still and hands off. Read-only diagnostics
found stable positions but intermittent velocity outliers consuming the existing
burst/total budgets; some later windows passed. A final capture with a 90-second
timeout succeeded without changing any acceptance threshold or calibration gate.
The operator subsequently described the spikes as an encoder glitch; their exact
cause has not been established by this capture.

Current calibration is `535afed263c24d7fa65a3c44dc7083d0`, captured at 08:01:11 UTC,
session `247fdb119d9cac915000259c9339e7a0`. FR/FL/BR/BL wheel homes are
`[-1.044378060546875, 0.951598939453125, -1.0447595302734376, 0.960754212890625]`
radians. The live file is `/home/pi/.local/state/quadmorph/calibration.json`.
The operator-requested replacement preset was saved on the Pi and mirrored locally
at `hardware_testing/start_pose/start_pose.json`, preserving the earlier preset in
history. Both copies compare equal. Automatic return after power loss remains
unimplemented. Neither locomotion policy was activated by the agent.

At the last check the gamepad was disconnected; joy_linux remains part of the
running focused launch. The earlier native-preparation checks below remain valid,
but their statement that hardware had not started is historical.

## Native Pi preparation completed

The Pi at `10.140.55.163` was prepared after the operator powered it on. The
selected checkout is `/home/pi/robot-code-leglift`, at detached `c11f5ce` with
a reviewed 27-file source/configuration/weight overlay. Its existing hardware
source and components.xacro matched the local baseline after newline normalization
and were preserved. Other robot checkouts and saved calibration were untouched.
Previous files and the installed seven packages are backed up at
`/home/pi/locomotion-prep-20260912/backup-023122`.

The native seven-package build and all 10 selected CTest entries passed. Installed
package prefixes, binary linkage, export hashes and hardware URDF resolution passed.
`native_pi/native-manifest.json` records installed binary and source hashes;
all 27 deployed file hashes match the current PC working tree. The initial local
`source_manifest.json` below remains the historical pre-deployment snapshot.

Live `/dev/input/js0` is the DualSense gamepad; js1 is its motion sensor. The kernel
button map confirms triangle=2, circle=1, **PS=10**, Options=9. Index 12 is R3,
so the earlier PS=12 configuration was corrected to PS=10 during native preparation.
The installed launch config and joystick integration test use the corrected index.
A standalone joy_linux node on isolated domain 96 delivered fresh neutral input;
it was stopped after inspection. See `native_pi/gamepad-map.json` and
`native_pi/joy-readonly.json`.

No physical hardware stack has been started. Homing, current-session calibration,
actual policy stability and physical stop response await the operator's explicit
supported-pose confirmation and supervised trial. The remaining-target list below
describes the earlier PC-only preparation, before these native checks.

## Earlier PC-only preparation

Local software preparation passed on WSL Ubuntu, x86-64, ROS Jazzy. No Pi
connection, physical hardware activation, calibration capture, or motion occurred.
Changes are in the working tree based on `bb393e4`; `source_manifest.json` records
the tested files and policy hashes.

Triangle (2) selects the selected gap-trained walking policy; circle (1) selects
the gap-trained wheel policy. The old circle leg-lift action is disabled. Duplicate
button bindings now reject joystick-node startup. The focused trial leaves X,
square and L2 unbound and starts both policies inactive with calibration required.

## Verification

- Seven packages built: robot_calibration, control_board_hardware_interface,
  neural_controller, joy_utils, pupper_v3_description, cmd_vel_mux,
  animation_controller_py. Existing hardware/vendor compiler warnings remain.
- All **10 selected CTest entries passed**: three calibration checks, alignment
  joystick, locomotion joystick, startup launch, locomotion manager, hybrid
  lifecycle, walking inference, and wheel inference.
- The locomotion joystick tests use the actual installed trial parameters and
  real joystick/teleop/mux executables with a fake manager. They cover triangle
  and circle dispatch, mutual deactivation requests, disabled old controls,
  held-button behavior, emergency-stop precedence, Options reentry, missing
  calibration rejection, conflicting circle binding rejection, stick scaling,
  and zero velocity after input timeout. The positive dispatch case disables
  calibration only in its isolated fake process; the hardware launch requires it.
- A real ROS controller manager with `mock_components/GenericSystem` loads both
  policies, runs inference, switches walking to wheels and back, and checks finite
  outputs, actuator gains, and emergency-stop outputs. Its calibration record is
  a synthetic temporary fixture, never a physical robot calibration.
- Both RTNeural exports match 128 independent reference actions each. Source
  and installed policy/configuration preflight passes for both recorded hashes.
- The installed joystick, neural plugin and hardware plugin have no missing
  shared libraries after sourcing the selected ROS overlay (`linkage.log`).
- The exact `scripts/prepare_locomotion.sh` workflow passed end to end; see
  `preparation.log`. The initial full build is retained in `build.log`.

## Remaining target checks

The Pi needs native ARM64 installation/build, package-prefix and service/process
inspection, live `/joy` mapping, scheduling checks, and physical trials. The local
WSL environment has no `joy_linux` device driver package or robot gamepad; tests
inject synthetic `/joy` messages. They do not validate that hardware input path.

Follow [LOCOMOTION_LAB.md](../../LOCOMOTION_LAB.md) and
[STARTUP_CALIBRATION.md](../../STARTUP_CALIBRATION.md). The operator must confirm
this startup's physical reference before homing. Both new policies' stability,
wheel direction, stop response and actual timing remain unmeasured on hardware.
