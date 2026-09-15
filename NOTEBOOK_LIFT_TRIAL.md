# Notebook lift policy: explicit robot-code trial

For the optional button-driven startup-home+180° rotation extension, see [NOTEBOOK_LIFT_ALIGN_TRIAL.md](NOTEBOOK_LIFT_ALIGN_TRIAL.md). The commands below continue to select lift-only mode.

This candidate uses the **exact lift-only actor shown in the continuous FR → FL → BR → BL video**. The policy commands all eight proximal positions throughout stand, lift, hold and lower. Each motor 3 holds its fresh activation angle using position PD. **This launch does not rotate or align wheels.** Existing walking/wheel policies, startup homing targets, normal launch selections and old policy ABIs are unchanged.

Status: experimental simulation candidate, **not physically validated**. PC integration did not start, calibrate, install on, or move the robot. The PC has no ROS/colcon/ARM build environment; actual plugin compilation and fake-interface lifecycle tests must pass in the robot's build environment before launch. A failed check is a blocker, not permission to bypass the wrapper.

## Install and check (no hardware startup)

Use the existing robot-code checkout `/home/pi/robot-code-leglift`. Preserve local changes and any active hardware session. After bringing that checkout to the robot-code commit containing this document:

```bash
cd /home/pi/robot-code-leglift
bash scripts/prepare_notebook_lift.sh
bash scripts/prepare_notebook_lift.sh --build
```

The first command checks source hashes. The second builds only the neural-controller package into separate `ros2_ws/install-notebook-lift` / `build-notebook-lift` directories and runs trained-network, geometry, observation, target, request-state, fake-interface lifecycle and applicable existing controller checks. It uses an isolated ROS domain for tests; it starts no hardware node. It records a receipt tied to the source release and installed shared-library hashes. The trial wrapper requires that receipt.

Prerequisite: current `install`, `install-roll`, `install-combined`, and `install-upper-home` overlays. If the upper-home overlay is missing or stale, follow **STARTUP_UPPER_HOME.md / Installation**, which is a build operation. Do not replace it with an old raw encoder or manual upper-pose procedure. Default hardware/description/calibration packages remain selected from `install-upper-home`.

## Operator trial

Follow **STARTUP_UPPER_HOME.md** and **STARTUP_CALIBRATION.md**, with the current upper-home instructions taking precedence over historical manual upper posing. Never start a second hardware owner. If a valid stack is already running, preserve it: first installation of this candidate may require a planned supported stop/relaunch because an existing manager may not have the new plugin/parameters. This wrapper refuses occupied SPI devices; it does not restart anything automatically.

Only after the operator actually confirms support and clear joints, start the explicit candidate stack:

```bash
cd /home/pi/robot-code-leglift
bash scripts/run_notebook_lift.sh --supported
```

Hardware activation automatically homes motors 1/2 against stops and returns to the saved upper home. Hubs remain unpowered during that startup procedure. Wait for `Saved upper home reached`. The operator then aligns the marked hub rings and confirms; only then perform the current-session capture described in STARTUP_UPPER_HOME.md. Reuse a valid calibration in the same encoder session. Do not reuse saved simulation homes. The candidate is spawned **inactive** and refuses activation without valid calibration and stationary, near-nominal, level support geometry.

For a floor trial, the operator must have the robot in a stable supported stand, clear of obstructions, before activation. Geometry agreement is not a physical contact sensor and can also occur while the robot is suspended.

- First **X** press: activate candidate into stand. Allow at least 0.5 seconds of settled stand.
- Subsequent **X** presses: **FR lift → lower → FL lift → lower → BR lift → lower → BL lift → lower**, then repeat.
- Wait for supported recovery before advancing. An early next-leg request queues that leg and lowers the current leg first.
- Existing **PS emergency stop** remains a latched zero-command/zero-gain stop. Releasing it does not resume the maneuver.

`/notebook_lift_command_index` is an Int32 request topic for this candidate only: stand/lower=0, FL=1, FR=2, BR=3, BL=4. It is volatile; stale requests are not replayed on activation. Do not publish requests during startup calibration. Changing a request does not change the captured motor-3 targets.

## Policy and execution contract

- Checkpoint: `params_2129920`, 2,129,920 training steps, SHA-256 `bf27a972ef20f4187e0210a5416e775772d69ef71b5a27601616a087642a6ad1`.
- Learned ABI: `quadmorph-align-notebook-v4`; runtime behavior `notebook_lift_v4`, contract `quadmorph-notebook-lift-runtime-v1`. This is the user's subsequent notebook-based design, not the earlier proposed 91-observation direct-position ABI.
- 288 inputs = four newest-first 72-value frames; eight tanh outputs; twelve position commands. ELU network and folded normalization are unchanged from the trained export. No contact force, contact flags, base height or base velocity enter the actor.
- Canonical motor order FR, FL, BR, BL. Eight proximal targets are `[1,0,-1,0,1,0,-1,0] + 0.75 * action`, clipped 0.06 rad inside the trained physical limits. Requested, applied and measured targets are distinct.
- Proximal target execution: 0.1 rad/s, 2 rad/s² acceleration with bounded braking. Actor inference every ten command updates; nominal command rate 520 Hz / actor rate 52 Hz. Execution and request timing use measured dt; command gaps exceeding 10 ms stop the candidate. There is no authority fade or hold-phase target freeze.
- Desired clearance ramps to 8 mm in four seconds; manual hold; four-second descent preserving current height/rate when interrupted. The actor supplies the joint motion. A 48-second attempt timeout requests descent and prevents an automatic retry. Supported recovery is bounded to four-second descent plus ten-second settling allowance.
- Gains: proximal kp=5, kd=0.25; hubs kp=8, kd=1, exactly as this lift checkpoint's simulation. This differs from the original kp=4/kd=0.15 alignment proposal. No hub integral, no outer velocity PD and no commanded alignment motion.
- Startup homes define angle observations; fresh measured hub positions define the fixed continuous motor commands. No wheel-forward signs or wrapping are applied to those commands. Reentry captures new winding. Proximal entry targets initialize at measured positions with zero target velocity; four history frames seed current measurements. Entry requires a stationary stand within 0.25 rad of nominal.
- Hardware effort clamp is not proof of a total local-PD torque clamp. The candidate stops if sampled `kp*(target-position)-kd*velocity` exceeds 3 Nm. This is an **estimate**, not measured torque. Simulation saturates torque; the candidate stops on excess instead. Its dynamics after such an event are not claimed equivalent. Per-encoder packet age is unavailable from this hardware interface; finite feedback and IMU age <=0.1 s cannot prove individual encoder freshness.
- Supported recovery uses all-wheel relative cylinder-bottom spread <6 mm, tilt <0.12 rad, body angular speed <0.3 rad/s, measured joint speeds <0.1 rad/s, proximal tracking <0.12 rad, hub hold error <0.035 rad, settled 0.5 seconds. It is a rigid-ground geometry check, not proof of load-bearing contact. Runtime holds can still have tracking error; fixed targets do not guarantee physically motionless hubs.

Model: cold WHEEL, heating backpack mass 0.60191707499 kg and the existing 9 mm spacing applied once. Total model mass 3.81991707499 kg. Model hash `4275f813ad1b6b1763ec42f70f23e871114724b3a51baa238c0b4efaafa588cb`. Runtime wheel geometry is generated from this model, not the shared Stanford XML. Runtime integration base is robot-code `63820b0`, preserving the newer wheel export, saved upper homing, and seq_d walking export relative to the original handoff's `085a3f0`.

## Evidence and limitations

The continuous simulation video used one physics reset and retained controller/history/filter state across all legs. All four recovered to supported stand in 87.75 s. FR/FL/BR achieved eight seconds of qualified hold; BL achieved 4.48 s and triggered `grounded_hub_motion`. Overall diagnostic pass is **false**. The checkpoint's separate single-leg audit was 47/64: FR 16, FL 16, BR 9, BL 6. These are diagnostic results, not the original task's complete alignment acceptance suite.

Video run: https://wandb.ai/QuadMorph/Align%20triangle/runs/26b97c51476743e2 . Full-speed rollout Media was verified remotely. On the PC the video and synchronized traces remain in `/home/theerawit/Pupper_animation/align_notebook_button_sequence/runs/four-lift-video-01/`. No new training is represented by this integration.

PC checks passed:

- 350 real trained-network fixtures, including continuous rollout phase/leg edges: max normalized-action error 2.80176e-6 with actual RTNeural Eigen backend; 5.30435e-6 with STL, both below 1e-5.
- 256 native-MuJoCo wheel geometry fixtures: max error 6.93889e-17 m.
- 512 observation/history fixtures: max error 5.96046e-8; 20,000 command filter steps: max error 2.47359e-6.
- Request cancellation, queued legs, timeout/recovery, no automatic retry, fresh hub winding, live actor authority and bounds; existing walking-frame, keyframe-position and align-motion core regressions.
- Four source/launch tests with stub launch constructors, explicit inactive selection and help/startup guards. These do not substitute for real ROS launch tests.

Pending: ROS Jazzy build, plugin discovery, fake-interface lifecycle tests on the target architecture, measured inference timing at 520 Hz, actual installed overlay verification, physical calibration/stand entry acceptance, supervised supported then floor trial, and physical hub drift/clearance evaluation. Full PD alignment behavior and acceptance remain separate unfinished work.

Reproduce portable native checks (no ROS/hardware):

```bash
CXX=c++ bash hardware_testing/notebook_lift/run_native_checks.sh
python3 scripts/notebook_lift/test_source.py
python3 scripts/check_notebook_lift.py
```

PC used `/tmp/pupper-hardware-toolchain/cxx` (Zig 0.14.1 wrapper) as CXX with private `.cache/zig-global` and `.cache/zig-local`. Reproduction on the Pi uses the system C++ toolchain. Source/import provenance and file/checkpoint hashes are in `hardware_testing/notebook_lift/release.json`; full fixtures and logs accompany it.

## Telemetry

Under `/neural_controller_notebook_lift/`: `observation` (288), `policy_output` (8 raw normalized outputs), `position_command` (12 applied targets), `motor_commands` (12 rows of position, velocity, effort, kp, kd, estimated PD torque). Derive the requested eight targets from `policy_output` using the documented action map; target velocities are observation-frame offsets 64–71. Measured encoders remain available through the existing joint-state broadcaster.

`lift_status` indices: phase, active canonical leg, pending leg, desired clearance, clearance rate, phase elapsed, attempt elapsed, timeout flag, recovery failure, geometry-supported flag, fault code, last command dt, IMU age, four relative wheel bottoms. Phases 0 stand, 1 lift, 2 hold, 3 lower. Actor phase returns to stand after the lower ramp while recovery is still checked. Faults: 1 clock, 2 sensors/IMU age, 3 tilt, 4 model/core/estimated PD, 5 operator stop, 6 activation. Investigate and reactivate deliberately after a fault; do not suppress it to get a trial running.
