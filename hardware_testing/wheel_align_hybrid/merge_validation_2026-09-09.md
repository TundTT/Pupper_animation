# Robot-code merge validation — 2026-09-09

Merged `origin/robot-code` (`9ad0301`) into local `robot-code` (`76eeced`)
with `git merge --no-ff --no-commit origin/robot-code`, then validated before
committing. Git resolved the merge cleanly; no conflict choices were needed.
No push or hardware activation was performed.

## Policy and calibration reconciliation

Local `342adf3` and remote `9ad0301` contain byte-identical `selection.json`
and `training_run.json` in `hardware_testing/walk_2026-09-07_long_stride`.
Both select `walk_2026-09-07_16-48-35/params_000021626880`, step 21,626,880,
W&B `QuadMorph/pupper-leg/zw331stu`:

- Checkpoint SHA256: `f9b30170c1d927c039204a92b4015fff59ebbdc9ede2137aca4878812290ea5e`.
- Export SHA256: `814d095421ebfbf7d2b8a1cdb4f1fa9d2ccd7c8fb85406e4162ce60bcfd42b2a`.
- Both training records name the same warm-start checkpoint,
  `walk_2026-09-07_15-29-41/params_000043253760`, with SHA256
  `a3dc1e16c6f2680af34130000048075e54728d7008d9e3269b0ad5e5bd6ba450`.

The walking policy LFS object, inference CSV, checker, entire long-stride
evidence directory (including both flat-friction reports), and
`WALK_V2_TESTING.md` are identical on both sides. The existing consistent copy
is retained. There is no walking-policy selection decision outstanding.

The only merged change relative to `76eeced`, apart from this report, is
`components.xacro`. It exactly matches the remote file, preserving its XML/YAML
comment fixes and dated knee raw references: front_r `0.072700`, front_l
`0.110100`, back_r `-0.252300`. Local commits did not modify that file.
Joint position limits are unchanged. Every file from `76eeced`, including the
hybrid implementation, launch/config, policy, exporter, tests and handoff,
remains unchanged.

## Build and focused tests

Ran the full-workspace build command in `WHEEL_ALIGN_HYBRID_TESTING.md` using
the existing `ros_jazzy` micromamba environment, build/install directories,
build-local setuptools 79.0.1, `-g0`, and CMake policy minimum 3.5.
All **16 packages passed** (9.34 seconds, incremental build), with existing
setuptools/CMake/dependency warnings; zero failed or aborted packages.
Full build log: `ros2_ws/log/build_2026-09-09_18-55-42/`.

Rebuilt `neural_controller` with the same flags and `BUILD_TESTING=ON`:
one package passed. Other packages retain `BUILD_TESTING=OFF`, following the
documented missing simulator `ros_testing` dependency workaround.
Controller build log: `ros2_ws/log/build_2026-09-09_18-56-43/`.

Ran `ctest --test-dir ros2_ws/build/neural_controller -R
'hybrid_|walk_policy_contract' --output-on-failure -V` in `ros_jazzy`, with
`ROS_DOMAIN_ID=197` and `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`:

- `hybrid_controller_lifecycle`: passed actual YAML/controller lifecycle,
  simulated loaned actuator mapping, observations, wheel holding, interruption,
  emergency stop during startup/between policy ticks, and reactivation.
- `hybrid_policy_contract_and_inference`: passed gate/PD/phase/reset cases and
  128 Brax/RTNeural fixtures; maximum action error `3.09944e-6`.
- `walk_policy_contract_and_inference`: passed command/history regression and
  128 JAX/RTNeural fixtures; maximum action error `1.3113e-6`.

Result: **3/3 passed**, zero failures. Detailed local test log:
`ros2_ws/build/neural_controller/Testing/Temporary/LastTest.log`.

## Manual activation verification

No correction to `WHEEL_ALIGN_HYBRID_TESTING.md` was needed. Checked the
installed launch with `ros2 launch neural_controller launch.py --show-args`,
controller-manager executable discovery, and help for `ros2 control
list_controllers`, `switch_controllers`, and `ros2 topic pub`. Inspected the
inactive spawner, plugin XML, runtime subscriptions, command order and telemetry
publishers. Resolved the installed YAML substitutions and verified the model
path, controller registration, and inclusion in stop/switch handling with no
joystick activation binding.

Source and installed hybrid/walking weights, config and launch files are
byte-identical. Hybrid export SHA256 remains
`4804edb8fa2381f9b48aca449354e13af7ab3b56731a65db0d17fbd1d6a8c8cb`.
Installed hardware and simulator xacros both expand successfully and pass ROS
launch's normalized parameter evaluation, including YAML interpretation of
`robot_description`. Walking preflight passes against source and installed files.

The documented strict switch must deactivate the actual active actuator
controller(s); publish on `/wheel_align_hybrid_command_index` after successful
activation and the two-second startup ramp. Commands remain 1=FL, 2=FR, 3=BR,
4=BL, 0=stand/interrupt. `/emergency_stop` remains `std_msgs/msg/Empty`.

Hardware testing remains pending. Session home is still captured on each
activation and targets are `wrap(home + pi)`, so this tests relative half-turns.
Physical home calibration/persistence needs a lab decision for absolute
alignment. Button binding remains a separate TODO; manual activation is available.
