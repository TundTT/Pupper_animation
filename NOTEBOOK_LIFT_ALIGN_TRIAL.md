# Notebook lift + position-PD alignment candidate

Adds **lift → rotate to startup hub position +180° → lower → next leg** to the existing `neural_controller_notebook_lift` controller, selected by the new, read-only `alignment_mode=quadmorph-notebook-lift-align-v1` contract. The original lift-only launch remains available. This is the same trained eight-output actor (`params_2129920`, checkpoint hash `bf27a972ef20f4187e0210a5416e775772d69ef71b5a27601616a087642a6ad1`), not a newly trained alignment policy. Walking, wheel and normal startup defaults are unchanged.

**Historical diagnostic (`2357f8e`, 10 mm gate):** the four nominal native-MuJoCo diagnostics did not achieve alignment. Maximum conservative floor clearance was about 5–7 mm, below the then-current 10 mm gate, so no rotation was enabled. FR/FL/BR completed supported timeout recovery; BL failed to settle within the bounded recovery window. This implementation is a reviewable controller candidate, not demonstrated successful alignment. Do not bypass the gate to turn those failures into a claimed pass. ROS/ARM build/lifecycle and physical tests remain pending; no physical hardware was accessed for this change.

## Combined launch update

The latest combined integration (`514db35`) changed the floor gate to **3 mm**;
that upstream change is retained. The recorded four-case failures above used the
old 10 mm gate and are historical, not evaluations of the new threshold.
`combined_motion.launch.py` now selects this mode via **R2/right trigger (button7)**:
first pull activates stand, subsequent pulls advance lift/rotate/lower per leg.
Square is unbound; the dedicated `--align` trial below still uses X. Build the
combined workflow with `scripts/prepare_combined_motion.sh`; see
COMBINED_MOTION_LAB.md. No physical R2 test or successful revised-gate alignment
is claimed by this button-binding change.

## Target definition

Each hub's orientation goal is **its live-session startup calibration position + π radians**. The live calibration is captured after the operator aligns the marked hub rings, following `STARTUP_UPPER_HOME.md`. It is not the position when the rotation button is pressed, the proximal nominal pose, or a saved simulation home.

On controller activation, choose the equivalent continuous encoder target nearest the current hub measurement; an exact half-turn tie chooses **positive π**. For example, startup 0.3 rad gives 3.441592653589793 rad when entering at 0.3 rad. If that wheel is already there after several whole revolutions, choose the equivalent nearby target instead of commanding another half-turn or unwinding multiple revolutions. The chosen goal stays fixed throughout that activation. Repeated rotate/leg requests never add another 180°. Reentry uses the same valid session home and fresh encoder winding.

Supporting hubs hold their captured or already aligned targets. Once alignment settles, the selected hub holds the final goal through lowering and later legs. Those are fixed **targets**; actual tracking can still drift. Actual drift invalidates the completed mask and remains visible in telemetry.

## Build and select

After pulling the new robot-code commit into the existing `/home/pi/robot-code-leglift` checkout, preserve any running hardware session and build/check without starting hardware:

```bash
cd /home/pi/robot-code-leglift
bash scripts/prepare_notebook_lift.sh --build
```

The existing isolated overlay, installed-library receipt, calibration and startup checks apply. Rebuild is required: a previous lift-only receipt cannot authorize the changed binaries. Both lift-only and alignment fake-interface lifecycle tests are now included in the robot-side test command. They have not been run on this PC because ROS is unavailable.

For a subsequently authorized supervised trial, follow `STARTUP_UPPER_HOME.md` and `STARTUP_CALIBRATION.md`. Only after actual support/clear-joint confirmation, and only with no other stack owning hardware:

```bash
bash scripts/run_notebook_lift.sh --supported --align
```

This selects `notebook_lift_align_trial.launch.py`. It spawns the candidate inactive; automatic upper homing and subsequent operator-confirmed marked-hub calibration remain mandatory. This task did not perform those actions. A valid running session should be preserved; the wrapper refuses duplicate hardware ownership rather than restarting it.

Verify selected mode with `ros2 param get /neural_controller_notebook_lift alignment_mode` before activation. Without `--align`, the existing lift-only trial is selected.

## Buttons and requests

First **X** activates into stand; allow at least 0.5 seconds of supported settling. Subsequent presses cycle:

**FR lift → FR rotate → FR lower → FL lift → FL rotate → FL lower → BR lift → BR rotate → BR lower → BL lift → BL rotate → BL lower.**

Wait for the lift to stabilize before rotate, and for `verified` before lower. An early rotate press queues permission until the lift/clearance checks qualify; it does not spin a loaded wheel. A lower or next-leg request during an unverified rotation is cancellation: stop the hub, lower first, and leave that leg incomplete. Completed legs cannot be automatically spun again by cycling X. The existing PS zero-command emergency stop is unchanged.

Request topic remains `/notebook_lift_command_index` (Int32): stand/lower=0, FL=1, FR=2, BR=3, BL=4; **rotate active leg=5 in alignment mode only**. Requests are volatile and reset at activation. Never publish candidate requests during startup calibration. The old lift-only mode rejects command 5.

## Controller equations and state

The 288-input/four-frame actor runs continuously at nominal 52 Hz. All eight proximal absolute targets retain the trained scaling, clipping, 0.1 rad/s and 2 rad/s² execution at nominal 520 Hz. There is no scripted proximal lift, freeze during verification or authority fade. Requested clearance remains 8 mm with four-second lift/lower ramps; it has not been increased solely for deployment.

Before actor inference, the controller computes the upcoming hub reference velocity and rotation permission. The existing actor inputs at offsets 51 and 52–55 carry them; offsets 56–63 encode current hub reference errors. Supporting/aligned reference velocities are zero. The learned weights are unchanged, so robustness to rotation reaction forces remains unproven.

For the active unverified hub, normal reference updates use:

- Desired velocity `clip(2*(goal-reference), -0.5, +0.5)` rad/s.
- Velocity change bounded by `1.2 * actor_dt` rad/s, then integrated every command substep.
- Position PD `8*(reference-measured_position) - 1*measured_velocity`, matching this notebook checkpoint's hub gains. No outer velocity controller, velocity feedforward or integral is added.
- The existing sampled 3 Nm estimated-PD stop and command/sensor guards remain. Hardware's total local-PD saturation remains unverified; estimated torque is not force sensing.

Rotation requires phase HOLD, an explicit rotate request, and continuously qualified gates for 0.2 s: conservative floor clearance >3 mm, wheel surface sphere gap >10 mm, base-body sphere/box gap >5 mm, tilt <0.12 rad, body angular speed <0.3 rad/s. Geometry uses the exact backpack/9 mm model; floor estimation uses the corrected **lowest support upper bound** with wheel-radius interval 45.5–50.5 mm. The heating backpack is excluded from wheel-body checks as requested; the base-body collision box is retained.

Gate loss is rechecked at each 520 Hz update. It captures the measured hub position **once**, sets reference velocity to zero and resets qualification/settling. This explicit safety reset follows the notebook servo's hold behavior and is an exception to normal reference-acceleration continuity; residual physical deceleration must still be audited. It does not repeatedly move the hold with encoder drift. The next actor frame reports the stopped reference. No future measurements are provided to the actor.

Verification requires actual goal error <0.025 rad, actual hub speed <0.08 rad/s and reference error <0.001 rad, settled for 0.5 s under the gate. The goal then becomes the fixed hold. Lower/NEXT waits for actual hub speed <0.08 rad/s for 0.2 s before descending; failure to stop within 2 s latches the existing stop. Cancellation keeps current clearance/rate and proximal filter history. There is no automatic descent simply because alignment was verified: the operator presses lower.

Completion is marked only after supported lowering and final angle <0.035 rad / speed <0.08 rad/s. Actual drift clears completion, without changing the commanded aligned target; a verified active hub error >0.1 rad faults. The 48 s attempt timeout requests stop/lower and does not count as success or automatically retry. Lower recovery remains bounded by the four-second ramp plus ten-second settling budget. These are sensor-derived geometry/tracking checks, not physical proof of wheel load.

## Checks and evidence

- All 1,028 independent native geometry fixtures match C++ within 1.11022e-16 m. Four statically reachable lifted poses exercise gates; they are **not learned balance results**.
- Pure controller tests exercise all four +180° goals, positive half-turn ties, both winding directions and multi-revolution reentry; normal reference velocity/acceleration; all-leg completion/retained holds; repeated commands; substep gate loss and requalification; cancellation and timeout recovery. Their ideal encoder-following harness is software evidence only.
- Existing trained-network inference, observation/history, target-execution, walking-frame, keyframe-position and align-motion regression fixtures remain in the release and are rerun by the portable check script.
- Native diagnostic uses the **actual RTNeural Eigen trained actor and new C++ runtime core**, with real MuJoCo hub position actuators in the pinned model. Four single-leg nominal cases request lift at 1 s and rotate at 6 s, lower after verification, or exercise timeout/recovery. Failures are preserved in `hardware_testing/notebook_lift/align_native_result.json`; full traces remain in the PC checkout's ignored `hardware_testing/notebook_lift/native_align_diagnostic/` directory with hashes recorded alongside the result. This is not MJX parity or the final randomized/sequence acceptance suite.

Reproduce source/native unit checks:

```bash
CXX=c++ bash hardware_testing/notebook_lift/run_native_checks.sh
python3 scripts/notebook_lift/test_source.py
python3 scripts/check_notebook_lift.py
```

Portable rerun logs/binaries go to the ignored `hardware_testing/notebook_lift/local_checks/` directory, preserving the committed evidence hashes.

`hardware_testing/notebook_lift/run_native_align_diagnostic.sh` compiles the RTNeural bridge and runs the native diagnostic when given the pinned model path and a Python environment containing MuJoCo 3.6.0/numpy; use `--help` for its exact arguments. It launches no ROS or hardware and explicitly disables GPU visibility. The model is not copied into robot-code; its hash is checked before simulation.

## Telemetry and remaining work

Existing `lift_status`, actor inputs/outputs, applied targets and motor commands remain unchanged. New `~/alignment_status` indices:

| Indices | Signal |
| --- | --- |
| 0–5 | rotation requested, enabled, verified, completed mask, failed mask, stopping |
| 6–8 | conservative floor, wheel gap, body gap in meters |
| 9–10 | gate qualification and alignment settling seconds |
| 11–14 | fixed final goals, canonical FR/FL/BR/BL |
| 15–18 | upcoming hub reference velocities |
| 19–22 | current hub position references/holds |
| 23–26 | live-session startup homes |

Before a physical rotation trial: validate the revised 3 mm gate with documented geometry evidence and test the actor under actual rotation reaction forces and gate-loss deceleration, complete the ROS/ARM and installed-overlay checks, then follow the separate supported hardware calibration/confirmation procedure. This commit implements the requested rotation controller but does not claim that this lift checkpoint now passes alignment.
