# Startup calibration: every robot session

This workflow applies whenever an agent starts the robot stack: ordinary use, a fresh boot, a restart, a new branch, or a new policy test. It is the user's standing requirement, not just a testing checklist. Heating remains manual and is outside this workflow.

For the current v5 alignment trial, follow [ALIGN_V5_LAB.md](ALIGN_V5_LAB.md) for the selected checkpoint, preparation checks and minimal launch. The physical-confirmation and live-session capture requirements below still apply.

## Agent procedure

1. Inspect the selected checkout, installed overlay and existing processes without activating anything. If the stack is already running, run `status` below. A valid calibration in that same live encoder session should be reused; do not restart or recalibrate merely to switch policies.
2. **Before a fresh hardware-stack launch, ask the user to prepare the physical reference and wait for their reply.** Suggested wording: “Please support the robot in its documented encoder-homing pose and position the marked wheel rings for the agreed calibration reference. Keep the robot stationary. Tell me when it is ready before I start the stack.” If the physical ring/proximal pose convention has not been established, clarify it; do not invent it from an old comment. An instruction to start the stack authorizes preparation, but is not confirmation that the robot is physically positioned.
3. Start the selected, built stack only after that confirmation. The existing hardware homing still runs; **starting the stack is not motor-free**. All neural motion controllers are spawned inactive. The hardware publishes its ready encoder session to a local file only after homing finishes. Do not press a policy/animation button during capture.
4. Once joint states and controller-manager services are ready, run the capture command with `--operator-confirmed` only if the user has actually confirmed this startup's physical setup. If setup changed during homing, or capture fails because the robot moved, obtain a fresh physical confirmation before trying again. Otherwise use the interactive command, which asks directly.
5. Check `status`, report the calibration ID, four home angles and file path, then enable only the policy/motion already requested. Capturing calibration does not itself authorize a new test, training run or deployment.

Do not synthesize a confirmation, pre-answer an interactive prompt, fill in guessed wheel values, or disable the hardware gate to make startup succeed. A stopped/missing calibration command is not a reason to start locomotion with provisional values.

## Commands on the robot

Run as the same user as the stack, normally `pi`. Source the overlay from the selected checkout; the recorded target checkout is `/home/pi/robot-code-leglift`, not the other robot's `/home/pi/pupperv3-monorepo`.

The legacy `robot/utils/robot.sh`, `robot.service` and service installer still target `/home/pi/pupperv3-monorepo`. Do not install or invoke them for this robot without adapting and reviewing the selected checkout. The D-pad launcher also starts hardware homing immediately; an agent must obtain physical confirmation before using it. Unattended services cannot ask a chat question: with the updated packages, they log the calibration requirement and keep neural policies gated until an operator completes capture.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/pi/robot-code-leglift
source ros2_ws/install/local_setup.bash

# Interactive capture: prompts before reading and saving.
ros2 run robot_calibration calibrate capture

# Agent equivalent, ONLY after the user's actual confirmation for this startup.
python3 scripts/calibrate_robot.py capture --operator-confirmed \
  --pose-note "Describe the physical mark and proximal reference pose confirmed by the operator"

# Validate live session + saved calibration; nonzero exit means not ready.
python3 scripts/calibrate_robot.py status
python3 scripts/calibrate_robot.py status --json

# Optional manual entry: actual current encoder readings, radians, FR FL BR BL.
# Replace placeholders with measured values, not desired zeros or old-session values.
python3 scripts/calibrate_robot.py capture --operator-confirmed \
  --wheel-home FR_RADIANS FL_RADIANS BR_RADIANS BL_RADIANS
```

`capture` never commands joints, switches controllers or starts the stack. It checks the live hardware session, checks that no active controller claims command interfaces, and collects at least one second of stationary readings for all 12 named joints. Data must have fresh, increasing ROS timestamps; missing values, nonfinite readings, moving joints, duplicate names and gaps restart/reject capture. Following operator-confirmed stationary recordings on September 11, the Python sampler permits brief reported-speed outliers above 0.02 rad/s: at most 20 ms consecutively and 50 ms total in the one-second window, charging both edges of each outlier. Any speed above 0.2 rad/s resets capture. Peak-to-peak position excursion is tightened to 0.002 rad on every joint, and the latest speed must be below 0.02 rad/s. Maximum sample age/gap remains 0.2 s. The saved record identifies `bounded_velocity_outliers_v1`. These are capture checks, not proof of physical stillness; user-confirmed support and reference positioning remain required.

Manual input undergoes the same live checks and must agree with the measured wheel angles modulo a full turn within 0.03 rad. It is useful for transferring reviewed current-session readings; it does not redefine the hardware encoder zeros. Plain capture is preferred.

If an operator explicitly wants to redo calibration within the same session, deactivate motion controllers and use `capture --replace` with a new physical confirmation. Normal startup, X and policy-switch actions never replace it automatically. A process-shared lock prevents neural activation/hardware zeroing from racing capture. Low-level direct forward-controller commands and old/unmodified controller plugins are not covered by the neural lifecycle gate; do not issue those during calibration.

## Saved data and validity

Default folder: `$XDG_STATE_HOME/quadmorph`, or `~/.local/state/quadmorph` when XDG_STATE_HOME is unset. On the normal Pi account this is `/home/pi/.local/state/quadmorph`.

| File | Purpose |
| --- | --- |
| `encoder-session.json` | Written by the hardware interface for each activation; marks homing pending/ready and identifies boot, owning process, process start time and a unique session ID |
| `calibration.json` | Current operator-confirmed calibration in that encoder session |
| `history/<calibration_id>.json` | Preserved successful captures for traceability; not automatically restored |
| `capture.lock` | Synchronizes calibration, hardware startup and neural-controller activation |

`QUADMORPH_CALIBRATION_DIR` can select an absolute folder. Set it consistently for the entire stack and capture command; do not use per-checkout copies of calibration. A common folder lets different policy branches use the same live hardware frame. Keep these machine-specific records out of git.

Schema version 1 records: calibration ID, encoder-session ID, operator confirmation, UTC time, canonical joint names, all 12 measured reference joint positions, four `wheel_home` angles, four `wheel_base_target` angles, radians, reference convention, capture method, optional pose note and the command checkout's source commit. Capture uses atomic replacement and preserves history. A write failure does not turn missing/stale data into a valid calibration.

Validity is tied to the actual hardware process and activation, not to “the file exists.” A reboot, dead controller-manager process, different process start time, hardware deactivate/reactivate or unfinished homing invalidates the old record. Saved values remain useful for inspection but are not automatically trusted after encoder zeroing. Restarting just the joystick node, switching policies, or stopping/restarting a policy does not re-home the wheels.

## Policy integration

Every `NeuralController` instance validates and loads the shared record on activation. Missing/stale calibration rejects activation even when requested directly through controller-manager services. No file I/O is added to the motor update loop. The joystick reports the missing calibration before requesting activation; the animation entry path also checks it. Stops remain available.

Hybrid alignment receives **startup home** from that record and captures **fresh hold/reference angles** from current encoders on every entry. X enters alignment without capturing calibration; later presses retain the existing leg selection cycle. The old `/wheel_align_hybrid_calibrate` topic is removed.

Python consumers:

```python
from robot_calibration import load_current
calibration = load_current()  # raises if missing or stale
home_fr_fl_br_bl = calibration["wheel_home"]
base_fr_fl_br_bl = calibration["wheel_base_target"]
```

C++ consumers add a dependency on `robot_calibration` and use:

```cpp
#include "robot_calibration/calibration.hpp"
robot_calibration::CaptureLock lock;  // lifecycle/capture boundary only
auto calibration = robot_calibration::load_current();
// calibration.wheel_home, calibration.wheel_base_target,
// calibration.reference_joint_positions, calibration.calibration_id
```

Within `NeuralController`, the protected `startup_calibration_` member is available to all behaviors after activation. Existing locomotion observation/action coordinates are deliberately preserved. **Do not subtract wheel home from every policy's observations or replace its default pose blindly.** A policy using another model-zero convention needs an explicitly derived per-leg mapping in both observations and commands. The shared record supplies the measured reference and provenance; it does not prove a model-to-hardware offset.

The marked-point-ring convention comes from the user's reshaping notes: `base_target = wrap(home + pi)`. It is a joint-angle convention. Physical setup must account for the proximal reference pose; “ring facing the ground while hanging” does not automatically mean “ring facing the ground in the alignment stance.” The script records all proximal angles, but does not infer ring identity or transform an arbitrary pose into the correct one. Verify this physical convention before the first hardware use.

`sim:=True` explicitly disables the physical-calibration gate for simulated policy/joystick/animation instances. Hybrid simulation uses `simulation_wheel_home` (four zeros by default), never a real robot file or an angle silently sampled at the action button. Never set `calibration_required:=false` on hardware.

## Build and local checks

This adds a small shared ROS package with standard Python/rclpy and Boost header dependencies; no heater controller, GPU dependency or policy retraining is required.

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --packages-select robot_calibration control_board_hardware_interface \
  neural_controller joy_utils animation_controller_py \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
source install/local_setup.bash
ctest --test-dir build/robot_calibration --output-on-failure
ctest --test-dir build/neural_controller \
  -R 'startup_launch|hybrid_|walk_policy_contract' --output-on-failure
```

Build all consumers together before a deployment. Older installed hardware/controller packages do not implement this workflow. Check `ros2 pkg prefix robot_calibration` and the selected hardware/neural package prefixes before startup. This document does not authorize deployment or physical motion on its own.

Local verification on September 10, 2026 used WSL Ubuntu with ROS Jazzy and isolated build/install folders. The five affected packages built. Checks cover storage/session invalidation and atomic writes, operator confirmation, stationary capture against a fake ROS encoder publisher/controller manager, actual neural-controller activation/reentry, hybrid/walking inference contracts, both launch calibration modes, and animation rejection. These are software checks; no robot was connected, calibrated or moved. The existing vendor RTNeural and hardware sources emit compiler warnings; this change does not claim a clean audit of those components.

## Source distinctions

- User-confirmed requirement: prompt and wait at every fresh stack startup, share saved calibration across policies, keep heating manual.
- Reference convention: user-supplied `quadmorph_reshaping_process.md`, “Startup calibration” (lines 18–32) and Phase 1 (lines 63–68), provided at `C:/Users/tundt/Downloads/quadmorph_reshaping_process.md`. It specifies the committed point ring touching the ground at `home_i`, then `home_i + 180°` for alignment. This is the design source; physical confirmation is still needed for the chosen proximal pose.
- Canonical FR/FL/BR/BL order and hardware limits/interfaces: [robot-info contract](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/robot_info/HARDWARE.md).
- Existing gravity-droop homing setup: [walking hardware instructions](WALK_V2_TESTING.md); zero-offset assignment and homing run are implemented in [hardware interface](ros2_ws/src/control_board_hardware_interface/src/control_board_hardware_interface.cpp).
- Continuous third-joint hubs: [robot-code hardware description](ros2_ws/src/pupper_v3_description/description/components.xacro) and September 9 entry in [hardware log](WHEEL_ALIGN_HYBRID_TESTING.md). The old robot-info third-joint hard stops describe its older leg profile.
- Physical marked-ring orientation remains an operator-verified convention; an agent comment or a passing software test is not hardware confirmation.

Historical documents describing X-time capture or publishing `/wheel_align_hybrid_calibrate` are superseded by this procedure.
