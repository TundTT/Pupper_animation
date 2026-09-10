# Hybrid wheel alignment: lab handoff

**Current implementation:** see [ALIGN_MOTION_V2.md](ALIGN_MOTION_V2.md) for startup calibration, restored legacy slew, and the new training workflow. The dated entries below are historical; their old calibration/button instructions do not describe the current branch.

`neural_controller_wheel_align_hybrid` is registered and spawned **inactive** by
`neural_controller/launch.py`. The controller and full ROS workspace compile;
robot testing is still pending. No joystick button activates this behavior yet.

## Selected policy and verified contract

The guard run's **own `audit64.json` names `attempt2_guard/mjx_params`**. It is
byte-identical to `params_50790400` (final step **50,790,400**):

- Checkpoint SHA256: `3c514877e07c28fed37525ff712cf8b22ca513f17de9853bbb0c64a0d77e0c2a`.
- Export: `ros2_ws/src/neural_controller/launch/policy_wheel_align_hybrid.json`.
- Export SHA256: `4804edb8fa2381f9b48aca449354e13af7ab3b56731a65db0d17fbd1d6a8c8cb`.
- Actor: **51 → 128 ELU → 128 ELU → 128 ELU → 8 tanh**. One frame, no history.
- Normalization is folded into layer zero. No observation clipping, sign correction,
  action delay, simulated hip action lag/slew, or policy action fade is applied.
- Audit: **50/64** full sequences, **0 falls**, **0 unsafe rotations**. A second
  seed audit was 49/64 with 0 falls. These are simulation results; incomplete
  sequences and drift failures remain possible.

The exporter reads the reference worktree without writing it, verifies source/XML
hashes against the run records, derives position limits from the training XML,
checks checkpoint layer dimensions, and checks serialized inference against Brax.
The JSON embeds `observation_layout`, joint/action order, and audit provenance.

| Offset | Size | Observation |
|---|---:|---|
| 0 | 3 | Angular velocity, body-aligned IMU frame |
| 3 | 3 | World gravity `[0,0,-1]` projected into that frame |
| 6 | 5 | Effective command one-hot: stand, FL, FR, BR, BL |
| 11 | 12 | Position joints minus default; wheels absolute wrapped angle |
| 23 | 12 | All encoder velocities × 0.1 |
| 35 | 8 | Previous raw network action, zero on activation |
| 43 | 4 | Sine of calibrated target error for every wheel |
| 47 | 4 | Cosine of calibrated target error for every wheel |

Joint rows are **FR, FL, BR, BL**, each **abduction, hip, wheel**. Eight actions map
to rows `[0,1,3,4,6,7,9,10]`, using scales `[0.5,1.6]` per leg. Position gains are
5.0/0.25. Wheel velocity servo gain is 0.35 with **zero position gain**; the outer
PD is `clip(2*wrap(goal-angle)-0.35*velocity,-2,2)` for all four wheels.

The position joints have the existing two-second startup pose ramp. Wheels hold
the activation snapshots during that ramp. Thereafter, the controller follows the
source's LIFT/ROTATE/VERIFY/LOWER/HOLD sequence. Phase changes are logged.
The manager remains 520 Hz with `repeat_action: 10` (52 Hz nominal; measure actual
rate in the lab). Wheel-reference slew uses accumulated elapsed control time,
0.25 rad/s. Gate/settle/lower thresholds remain **10/25/50 control steps**, as in
the source, rather than being converted to time thresholds.

Source details that differ from the task recap:

- `hybrid_env.py` latches `hold` on **VERIFY → LOWER** (and on interruption), so
  the settled wheel snapshot is held through lowering and HOLD. It does **not**
  re-latch at LOWER → HOLD. This port follows that source exactly.
- Settling counts consecutive gated rotation samples, including ROTATE as well
  as VERIFY. Losing the gate resets settling and freezes the reference **once**
  at the encoder angle; later drift is actively corrected against that snapshot.
- A leg is marked completed only after verified lowering. Completed legs are
  skipped for the rest of the activation; repeated commands do not rotate them
  another half-turn. Interrupted legs remain eligible for retry. The most recent
  command waits until lowering finishes. Stand interrupts and lowers too.

## Two explicit lab TODOs

1. **Calibration:** `WheelAlignHybrid::reset()` in
   `include/neural_controller/wheel_align_hybrid.hpp` contains `TODO LAB CALIBRATION`.
   It currently captures four encoder angles once on each activation as session
   `home`, then computes `target=wrap(home+pi)`. Thus this provisional version
   requests a half-turn relative to activation, not a known physical home mark.
   Tomorrow, decide what physical home means and how it is acquired/persisted;
   replace that acquisition with the agreed four values. Keep target computation
   and the independent current-angle hold snapshots intact. Deactivate/reactivate
   currently clears completion state **and recalibrates**; there is no persistence.
2. **Button binding:** search `TODO LAB BUTTON BINDING` in `launch/config.yaml`.
   The commented template has `wheel_align_hybrid_button_index: -1`, controller,
   command/cycle states, and topic. These are explicitly **not consumed yet**.
   Tomorrow, choose the actual button and add a separate press-edge cycle and
   publisher in `joy_utils/src/estop_controller.cpp`, using those names/states.
   Switch to this controller on the first press; publish the chosen `Int32` only
   **after activation succeeds and the subscriber is available**, then advance
   on subsequent presses. Its topic is volatile to avoid replaying commands from
   an earlier calibration session. Preserve the completed-leg semantics above.
   The instance is already in `controller_names` so existing stop/switch paths
   deactivate it, but it has no entry in `switch_button_indices` and no dispatch.

## Manual bring-up before binding

Build on the robot with its standard Jazzy environment, obtain the LFS weights,
and use the existing robot launch/service procedure (do not start a second
controller manager). With the robot supported, confirm the installed controller
is inactive and the joint/IMU conventions agree with the contract above.

```sh
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DCMAKE_CXX_FLAGS=-g0
source install/local_setup.bash
ros2 control list_controllers
```

For manual activation, first identify the currently active actuator controller(s).
Include their actual names in `--deactivate`; for example, when `neural_controller`
is the active one:

```sh
ros2 control switch_controllers --strict \
  --deactivate neural_controller --activate neural_controller_wheel_align_hybrid
# Wait for activation and the two-second pose ramp. Then select FL:
ros2 topic pub --once /wheel_align_hybrid_command_index std_msgs/msg/Int32 '{data: 1}'
# Subsequent commands: 2=FR, 3=BR, 4=BL. Stand/interrupt:
ros2 topic pub --once /wheel_align_hybrid_command_index std_msgs/msg/Int32 '{data: 0}'
# Existing emergency stop (also checked during startup and between policy ticks):
ros2 topic pub --once /emergency_stop std_msgs/msg/Empty '{}'
```

Record `/neural_controller_wheel_align_hybrid/observation` (51 values),
`/neural_controller_wheel_align_hybrid/policy_output` (8),
`/neural_controller_wheel_align_hybrid/position_command` (12, mixed position/velocity),
`/wheel_align_hybrid_command_index`, joint states, IMU, and phase logs. Verify wheel
encoder continuity/signs, physical lift clearance, support-wheel holding, gate loss,
interruption, lowering, stop response, and actual policy cadence on hardware.
The IMU must be aligned to the body convention used by the existing controller;
no additional sensor mounting transform is introduced here.

## Build and automated validation completed here

The machine has no `/opt/ros`; its existing `ros_jazzy` micromamba environment was
used. **All 16 packages built successfully**, with existing dependency/compiler
warnings. Setuptools 83 in this environment broke editable installs and repeated
normal installs (`--editable`/`--uninstall` unsupported). The final full build used
a build-local setuptools 79.0.1 via `PYTHONPATH`; the shared environment was not
changed. `CMAKE_POLICY_VERSION_MINIMUM=3.5` allowed the older
vendored RTNeural CMake project to configure under CMake 4. No dependency files or
other packages were changed. Full-workspace testing was disabled because the
local simulator test dependency `ros_testing` is absent; the relevant controller
was separately rebuilt with testing enabled.

Commands from repository root:

```sh
uv pip install --python /home/theerawit/micromamba/envs/ros_jazzy/bin/python \
  --target ros2_ws/build/python_build_compat 'setuptools==79.0.1'
micromamba run -n ros_jazzy env \
  PYTHONPATH="$PWD/ros2_ws/build/python_build_compat" \
  colcon --log-base ros2_ws/log build \
  --executor sequential --base-paths ros2_ws/src \
  --build-base ros2_ws/build --install-base ros2_ws/install \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS=-g0 \
  -DPython3_EXECUTABLE=/home/theerawit/micromamba/envs/ros_jazzy/bin/python \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_TESTING=OFF
# Re-enable tests for neural_controller, using the same build flags/paths:
# add --packages-select neural_controller and change BUILD_TESTING to ON.
micromamba run -n ros_jazzy ctest --test-dir ros2_ws/build/neural_controller \
  -R 'hybrid_|walk_policy_contract' --output-on-failure
```

All three focused tests passed: hybrid PD/gate/phase/inference, actual ROS plugin
lifecycle with the real YAML and loaned simulated state/command interfaces, and
existing walk-policy regression. Hybrid inference matched Brax on 128 fixtures
with maximum action error **3.10e-6** in RTNeural. Source versus installed policy
bytes and unbound launch/config wiring were also checked.

To reproduce the export with the existing training Python environment:

```sh
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  /home/theerawit/Pupper_animation-combine/mujoco_playground/.venv/bin/python \
  scripts/export_wheel_align_hybrid_policy.py \
  --reference-root /home/theerawit/Pupper_animation-align-hybrid \
  --out ros2_ws/src/neural_controller/launch/policy_wheel_align_hybrid.json \
  --reference-out ros2_ws/src/neural_controller/test/hybrid_policy_reference.csv
```

No motors were activated and no new physical simulation rollout was performed in
this worktree. Controller tests validate software behavior, not tomorrow's physical
calibration, IMU mounting, encoder signs, traction, or policy success on the robot.

## Changed files

Paths below are relative to the repository root:

- New exporter: `scripts/export_wheel_align_hybrid_policy.py`.
- New policy: `ros2_ws/src/neural_controller/launch/policy_wheel_align_hybrid.json` (Git LFS).
- Runtime: modified `ros2_ws/src/neural_controller/src/neural_controller.cpp` and
  `include/neural_controller/neural_controller.hpp` under that package; added
  `include/neural_controller/wheel_align_hybrid.hpp` for the small phase/PD state.
- Wiring: modified that package's `launch/config.yaml` and `launch/launch.py`.
  `joy_utils/src/estop_controller.cpp` was not changed; its binding is the lab TODO.
- Tests: added `test/hybrid_policy_test.cpp`, `test/hybrid_controller_test.cpp`,
  and `test/hybrid_policy_reference.csv` under that package; updated `CMakeLists.txt`.
- Handoff: this file and the link in `README.md`.
- Evidence: `hardware_testing/wheel_align_hybrid/validation.json`,
  `build_summary.txt`, and `test_results.txt`.

## Hardware test log

### 2026-09-09: first real-hardware activation, all four legs -- with two open issues

Continuous wheel joints were confirmed physically mounted on `leg_*_3` (no mechanical
stop), so `components.xacro`'s leg-era `hard_limit_min/max` clamp on all four of those
joints was removed and `position_min/max` widened to +-1000 to match, and the button
was bound live: **X now activates `neural_controller_wheel_align_hybrid`** (replacing
its prior job of switching back to plain locomotion -- there is currently no button
bound to locomotion; PS+Options reactivates whatever was last active, not locomotion
specifically).

The first X press after activation is calibration-only (captures home from wherever the
wheels are, commands nothing); each press after that advances the
front_l/front_r/back_r/back_l cycle. That calibration now also persists across
switching to another controller and back within the same process (fixed same day --
see the `wheel_align_hybrid.hpp` commit adding the `calibrated` flag -- because the
operator regularly bounces between wheel/leg/transition policies and was losing the
ground-truth home every time this controller reactivated).

**Result: on a full run through all four legs (front_l, front_r, back_r, back_l), each
one lifted, rotated to its target, and lowered successfully** -- the ported policy,
contract, PD gains, and phase machine all check out end to end on real hardware, not
just in sim.

**Open issue 1 -- mechanical interference:** when a leg lifts and then rotates back, it
collides/rubs against the back wheel. This is a geometry/clearance problem, not a policy
or control bug -- needs physical measurement of the collision and either a trajectory
adjustment (e.g. widen the lift or rotation path) or a hardware clearance fix. Not yet
investigated further.

**Open issue 2 -- one uncommanded shutdown, not reproduced:** during an earlier attempt
the same session, the robot went completely dead (full reboot, not just an estop) right
as the operator pressed X to advance from a just-completed `front_l` cycle to the next
leg -- `front_l` itself had lifted, rotated, and lowered cleanly beforehand. This matches
the signature of a previously-documented, still-open issue from the wheel-driving
policy's own hardware testing (`6f91d4d`, `WHEEL_TESTING.md`): a physical bump causing a
momentary power-connector disconnect. The operator subsequently checked and confirmed
the power connector was secure, then reproduced the full four-leg sequence above
cleanly with no further shutdown. Cause is therefore **undetermined** -- possibly the
same intermittent connector/mounting issue (which can reseat itself and look secure at
rest even though the fault is momentary, under jolt), possibly something else. Flag this
if it recurs; nothing in this session's code changes obviously explains a full system
reboot (as opposed to a node crash or estop), which points away from a pure software
cause.
