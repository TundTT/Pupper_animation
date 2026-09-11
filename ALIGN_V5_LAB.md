# V5 alignment hardware trial

This branch selects the final **35,225,600-step** checkpoint from [W&B run fe72769ebe2543bd](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/fe72769ebe2543bd). Training source is `e663e03bfd4ad6a2d094b7d9c9ed3c20902f4e10`. It completed 64/64 nominal sequences and 45/64 randomized sequences. **Full simulation acceptance failed; physical validation is pending.** This is an explicitly requested supervised trial candidate, not an automatically selected passing checkpoint. Original audit results are preserved in [hardware_testing/align_v5_2026-09-11](hardware_testing/align_v5_2026-09-11).

## Prepare before the lab

On the target Pi, use `/home/pi/robot-code-leglift`. `/home/pi/pupperv3-monorepo` belongs to another robot. Inspect existing processes and local changes first; do not overwrite them or build while that checkout is controlling a robot.

```bash
cd /home/pi/robot-code-leglift
git status --short --branch
git pull --ff-only origin robot-code
git lfs pull
bash scripts/prepare_align_v5.sh
```

The preparation command does not start hardware. It verifies the exact policy hash, modes, gains, joint/CAN order and limits; builds the calibration, hardware, neural-controller, joystick and description packages; runs the relevant tests; and checks the installed overlay. A missing `/dev/input/js0` means the controller must be connected before the final preflight can pass. Do not proceed after a failed build or test.

## Start and calibrate in the lab

Read [STARTUP_CALIBRATION.md](STARTUP_CALIBRATION.md). An agent must ask for and receive explicit physical-positioning confirmation before launching. **Startup performs encoder homing, even though the policy starts inactive.** Use the documented proximal encoder-homing pose and agreed marked-ring reference; keep the robot stationary. The reference must be physically verified with the wheel hardware currently mounted.

After confirmation, in terminal 1:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/pi/robot-code-leglift
source ros2_ws/install/local_setup.bash
python3 scripts/check_align_v5.py --installed
ros2 launch neural_controller alignment_trial.launch.py
```

This dedicated launch uses only motor control, state broadcasters, the joystick and alignment. Other policy buttons are disabled. It does not start cameras, vision, animation, heater bridges, or an automatic wheel sequence. Do not also start the D-pad/general launcher: it would create a second stack.

Once homing and controller spawning finish, in terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/pi/robot-code-leglift
source ros2_ws/install/local_setup.bash
python3 scripts/calibrate_robot.py capture
python3 scripts/calibrate_robot.py status
ros2 control list_controllers
```

Capture is interactive. Follow its physical confirmation prompt; it records stationary encoders in this live session. `status` must report a valid calibration ID and FR/FL/BR/BL homes. Alignment must still be **inactive** before the first X press. An agent may use `--operator-confirmed` only after actual confirmation for this startup, as documented in STARTUP_CALIBRATION.md.

## First motion

Use a catch/support arrangement that permits normal bodyweight loading. Keep heating off for this motion test.

| Control | Behavior |
| --- | --- |
| First X | Activate alignment in stand, smoothly enter its neutral pose; no wheel selected |
| Next X | Request front-left; lift, align, lower and hold automatically |
| Later X presses | Request front-right, back-right, back-left in order |
| X during a lift | Lower the previous wheel before beginning the newly requested wheel |
| PS button (configured index 12) | Emergency stop; zero motor position stiffness and apply damping; this can remove support, so use the catch |
| Options (configured index 9) | Reactivate alignment in stand using the same valid calibration; X then begins at front-left |

Wait for the initial two-second stand transition and stable ground contact before requesting the first wheel. Test one wheel first. Normal lift takes about 6.5 seconds; rotation duration depends on angle; lowering takes 10 seconds. Completion is logged as `LOWER -> HOLD`. A blocked attempt times out after 48 seconds and requests normal lowering. That timed-out command cannot restart itself; request stand or another wheel before retrying. If measured motion prevents LOWER from finishing, stop the trial rather than issuing repeated requests.

For a **soft return to stand**, while alignment is active:

```bash
ros2 topic pub --once /wheel_align_hybrid_command_index std_msgs/msg/Int32 '{data: 0}'
```

This requests the existing descent trajectory. It is not an emergency stop. Do not switch to locomotion or shut down the stack while a wheel is elevated as a substitute for lowering. After all wheels complete, reentry resets completion flags while retaining startup calibration. Restarting/homing the hardware instead requires a new capture.

## Logs for the trial

Terminal 1 reports phase changes and timeout warnings. Record useful topics in another sourced terminal before pressing X:

```bash
ros2 bag record -o "align-v5-$(date +%Y%m%d-%H%M%S)" \
  /joint_states /imu_sensor_broadcaster/imu /joy /rosout \
  /wheel_align_hybrid_command_index \
  /neural_controller_wheel_align_hybrid/alignment_status \
  /neural_controller_wheel_align_hybrid/policy_output \
  /neural_controller_wheel_align_hybrid/policy_inference_latency_seconds
```

`alignment_status` is a 13-value array at the policy rate: phase (IDLE=0, LIFT=1, ROTATE=2, VERIFY=3, LOWER=4, HOLD=5), requested command, active command, trajectory progress, estimated floor clearance (m), conservative wheel spacing (m), conservative body spacing (m), rotation enabled, completed-wheel bitmask (FR/FL/BR/BL bits 0/1/2/3), active target error (rad), residual gain, elapsed lifted-attempt seconds, timed-out command (-1 when none). Command indices are stand=0, FL=1, FR=2, BR=3, BL=4. Clearance values are model estimates, not physical distance sensors.

## Compatibility evidence and remaining physical checks

| Item | Evidence |
| --- | --- |
| ARM64 Pi 5, Debian 12, ROS Jazzy, RTNeural CPU inference | [robot-info software contract at b311700](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/robot_info/SOFTWARE.md); no JAX, GPU or training runtime is installed on the robot |
| Canonical joints and CAN channels | [robot-info hardware contract](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/robot_info/HARDWARE.md), checked against current components.xacro |
| Eight position joints, four velocity hubs; gains 5/.25 and 0/.35 | Hardware contract's wheel profile, current YAML and exact trained export; the robot-info leg-profile third-joint stops do not apply to the currently documented continuous wheels |
| Continuous hub configuration and proximal limits | [current components.xacro](ros2_ws/src/pupper_v3_description/description/components.xacro); preserved rather than replacing it with the older robot-info leg profile |
| 520 Hz manager, action repeat 10 = 52 Hz inference | Current config and v5 export use exactly 10/520 seconds. The robot-info legacy 50 Hz trainer description does not describe this v5 trainer |
| 83 observations, eight residual outputs, four deterministic hub servos | Ported v5 controller explicitly supports this ABI. The robot-info legacy requirement for twelve network outputs is superseded for this behavior; all twelve motors still receive commands |
| Encoder/IMU coordinates and inference | Controller tests exercise canonical ordering, wrapped hub angles, position offsets, gravity and raw angular velocity. Trained Brax-to-export error is 5.96e-7; RTNeural and actual plugin checks are included in preparation |
| Fixed startup home, fresh wheel holds on entry | Shared live-session calibration gate in robot-code, retained during the port; no X-time calibration or automatic startup-home topic |
| Ring/base convention | User-provided reshaping notes specify `base_target = wrap(home + pi)`; recorded proximal pose and physical marked-ring setup still require operator verification |

The software cannot establish that the physical wheel geometry, load distribution, joint zero offsets or ring orientation exactly match simulation. In particular, comments about gravity-droop poses derived from the older leg geometry are not fresh measurements of the wheel assembly. Do not compensate for a visibly wrong pose with an improvised sign flip or guessed offset. Check the physical convention before enabling motion.

The export is unchanged from the reviewed weights. The runtime port adds the shared-calibration adapter, diagnostic output and the 48-second soft-abort supervisor; it preserves the v5 references, filters, gains and collision gates. Legacy walking and hybrid policy files remain available; no training was run and no physical robot was started by this software integration.

## Local verification, September 11

- ROS Jazzy x86-64 build passed for calibration, hardware interface, neural controller, joystick and animation packages. Vendor RTNeural and existing hardware code emit compiler warnings; they did not prevent these checks.
- Seven targeted neural-controller CTests passed, including real trained inference, actual plugin lifecycle/update, soft interruption, timeout and retry behavior, stop during startup/between inference ticks, reentry after wheel revolutions, stale-session rejection, minimal launch and legacy policy regressions.
- Three calibration CTests and the actual joystick node/fake-manager test passed. The latter checks activation failure, pending presses, command ordering, stop priority and reset of the leg cycle on reentry.
- The ported C++ motion equations matched Python over 10,000 steps across all six phases to absolute tolerance 3e-10.
- Exact checkpoint SHA-256: `83a3e4caa8481d22fcbff5ad39947057707c87c4fa394343c8abd0ddf6212cc3`.
- Export SHA-256: `05f62e90101597ffbbd9d6a26dce443b3f1af831ba5a5879607f1709b19860a9`. Trained RTNeural action error versus Brax fixtures: at most 1.07288e-6.
- The source preflight passed, including all twelve joint axes and mounting transforms against the saved training geometry.

The Pi was initially unreachable. The subsequent lab setup below completes the ARM64 software checks. The original randomized audit failure remains visible; these software checks do not turn it into a passing physical policy.

## Pi software preparation, September 11

Prepared `/home/pi/robot-code-leglift` on `pupper` at `10.140.55.163`, through commit `ecc6e75`. All six packages built on the ARM64 Pi, all 11 selected CTests passed (three calibration, one joystick, seven neural), and the installed-overlay preflight passed with the exact reviewed v5 export. The connected `/dev/input/js0` identifies as the DualSense Wireless Controller; `js1` is its motion-sensor device. See [the actual Pi build/test log](hardware_testing/align_v5_2026-09-11/pi-setup-build.txt).

Two Pi compatibility fixes were required: use the canonical generated controller-parameter header (the old deprecated header contained duplicate definitions), and pass calibration overrides through YAML files. The older Pi spawner overwrites repeated `--controller-ros-args` arguments. Launch tests now exercise the installed spawner parser and resolve its parameter file; both the older Pi and newer local Jazzy versions passed. Hardware overrides require calibration; simulation overrides remain simulation-only.

The Pi's three existing local edits were preserved: homing-reference logging, raw encoder reference values, and disabled optional nodes in the general launch. They are backed up in `/home/pi/align-v5-setup-backup-c1ef4d2/pre-update.patch` and retained git stashes. The untracked `motion_capture/` directory was left in place. The other robot's checkout was not changed.

The legacy `dpad-launch-trigger.service` was stopped for preparation and remains stopped; it is still enabled for a future reboot. Do not use D-pad startup alongside the dedicated launch. No hardware stack or policy motion was started during these checks. **Physical positioning confirmation, fresh homing, live-session calibration capture, and motion validation remain pending.**
