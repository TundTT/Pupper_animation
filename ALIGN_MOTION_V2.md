> Current work is motion contract v4. Read [ALIGN_MOTION_V4.md](ALIGN_MOTION_V4.md) and [AGENT_TRAINING_HANDOFF.md](AGENT_TRAINING_HANDOFF.md). The material below records earlier revisions.

> Historical v2 design and validation. Current retraining changes are in [ALIGN_RETRAINING.md](ALIGN_RETRAINING.md); use [AGENT_TRAINING_HANDOFF.md](AGENT_TRAINING_HANDOFF.md) for current PC commands. The branch name remains `codex/align-motion-v2`, but new weights require motion contract v3. Preserve old training checkouts for historical replay.

# Wheel alignment motion v2

Training preparation and runtime implementation, **not a trained or hardware-validated policy**. No training was run on the laptop. Heating and the later SMP deformation/flip procedures are outside this change.

For an agent setting up the PC and running training, follow [AGENT_TRAINING_HANDOFF.md](AGENT_TRAINING_HANDOFF.md). It covers setup, preflight, a persistent training session, failure handling, evaluation and the artifacts to return.

W&B logging and policy videos are enabled by default. See [WANDB_LOGGING.md](WANDB_LOGGING.md) for the requested project, authentication and backfilling a completed run from its original checkout.

The branch is `codex/align-motion-v2`, based on `robot-code` at `582fd88ccce51d922e38cfd938d26bb5181bb276`. The existing controller instance and X button remain in use. Its YAML still selects `policy_wheel_align_hybrid.json`; selecting a new checkpoint is a separate deployment step after training and evaluation.

## What changed

| Reported problem | Implementation | Requires new training? |
| --- | --- | --- |
| Front wheel lifts back into rear wheel | Smaller nominal hip lift, 0.85 rad; wheel/body clearance gates; robot self-collision enabled in the new simulator | Yes |
| Jolting lift | Three-second quintic lift reference; bounded residual actions; actuator-target velocity and acceleration limits integrated at 520 Hz | Yes, for the complete v2 behavior |
| Slamming when NEXT interrupts a lift | Four-second descent starts from the current applied target and preserves velocity; the old leg finishes lowering before the latest requested leg starts | Yes |
| Calibration happens on X | Startup owner captures stationary encoders before any motion controller is active; controller activation consumes that retained snapshot | No |
| Reentry after driving moves wheels toward stale angles | Keep startup home fixed; refresh wheel hold/reference at each controller activation | No |
| Missing legacy action shaping | Restore the saved policy's active-hip limit of 0.08 rad per 20 ms, including LOWER | No; this alone does not solve the learned collision path |

The new actor observes 82 values and produces eight residual actions. The four existing hub motors continue to use the encoder-based velocity servo. No extra actuator or foot-force sensor is assumed. Phase, trajectory progress, reference, applied target and target velocity are included in the observations. Python and C++ implement the same motion equations.

Active-leg target limits are 0.4/0.7 rad/s for abduction/hip and 2 rad/s² acceleration. Support legs retain more adjustment range: 2/3 rad/s and 12/16 rad/s². These are **command** limits, not guaranteed physical limb speeds. Initial tuning needs dynamic evaluation. The two-second controller entry pose ramp is also quintic.

Rotation requires the lift reference to finish, estimated floor clearance >10 mm, all wheel-pair sphere gaps >10 mm, wheel-to-body gap >5 mm, body tilt <0.12 rad and angular speed <0.3 rad/s. The floor estimate uses encoders and IMU with flat ground and planted support wheels assumed. It is not a measured contact sensor. Loss of the gate pauses rotation. The hub reference advances at 0.25 rad/s; NEXT during an unfinished lift/rotation interrupts it, lowers, then services the latest request. Rapid presses replace the pending request rather than enqueueing several lifts.

## Startup calibration and X

1. Establish the agreed physical wheel-ring reference before starting the complete robot stack. Keep the robot stationary until the console reports `Wheel home captured at startup, FR/FL/BR/BL: ...`.
2. The owner requires fresh joint states and 0.5 seconds of stationary wheel encoders. It checks that no controller owns active command interfaces. A motion request before capture prevents a late, incorrect capture.
3. First X enters alignment in stand and refreshes holds from current encoders. Subsequent X presses request FL, FR, BR, BL. An entry still in progress does not advance the pointer; the console tells the operator to press again once entry finishes.
4. Controller switches preserve the startup reference. Restart the **whole stack**, in the calibration pose, after hardware encoder zeroing/reinitialization. Restarting only the joystick node during a moving session is not a recalibration workflow.

The retained topic is `/wheel_align/startup_home`, four angles in FR/FL/BR/BL order. The former active-controller calibration topic is removed. Missing startup calibration rejects alignment activation rather than silently treating the X-press position as home.

**Physical convention:** this preserves the reshaping notes' `base_target = wrap(home + pi)`: startup home denotes the marked point ring, and the opposite side is the compression base. Encoders cannot identify a particular physical ring. Home is a joint-frame angle; startup physical setup must account for the proximal joint pose so that returning the hips to the alignment stance gives the intended base-down orientation. This implementation does not infer a marker from an arbitrary drooping pose or compensate arbitrary terrain/body poses. Verify the marked ring and this convention on the robot before the first floor test. Changing the intended ring/geometry later requires updating that setup; changing the limb geometry also requires the model checks below.

## Train on the PC

Run from the repository root in Linux or WSL2 with the PC's NVIDIA GPU available. The trainer refuses CPU-only execution. The commands below assume `uv` is installed.

```bash
git fetch origin
git switch codex/align-motion-v2
git pull --ff-only
git lfs install
git lfs pull
uv sync --project training/wheel_align --extra cuda --frozen
PY=training/wheel_align/.venv/bin/python
$PY -m wandb login
$PY -m training.wheel_align.generate_geometry --check
$PY -m training.wheel_align.preflight --jit-step
$PY -m training.wheel_align.train --envs 2048 --steps 50000000 --seed 0 --out runs/align-motion-v2-seed0
```

The preflight only checks shapes, randomized environment compatibility, and three physics/control steps. It performs zero optimizer updates. Training itself starts only with the last command. Supported `--envs` values are 256, 512, 1024, 2048 and 4096; reduce it to 1024 or 512 if GPU memory is exhausted. The 50-million-step budget is a starting run, not a promised convergence point. The output directory must be new. Checkpoints, metrics, exact dependency lock, source commit and source/mesh hashes are recorded.

The training schedule includes all four wheels in randomized order, randomized home angles, motor/model variations, sensor noise, inference delay, and interrupted operations. It runs the same 52 Hz inference / 520 Hz actuator-reference integration used by ROS. Ground impacts and collision margins are audited at every physics substep.

## Evaluate and export on the PC

Use the **same checkout** used for training. Evaluation/export rejects changed source or geometry hashes.

```bash
PY=training/wheel_align/.venv/bin/python
RUN=runs/align-motion-v2-seed0
$PY -m training.wheel_align.evaluate --params "$RUN/mjx_params" --nominal --out "$RUN/audit-nominal.json"
$PY -m training.wheel_align.evaluate --params "$RUN/mjx_params" --out "$RUN/audit-randomized.json"
$PY -m training.wheel_align.evaluate --params "$RUN/mjx_params" --interrupt --seed 20260911 --out "$RUN/audit-interrupted.json"
$PY -m training.wheel_align.export --params "$RUN/mjx_params" --out "$RUN/policy_wheel_align_motion_v2.json"
```

Review all three audits: every environment should complete all four alignments, finish within 0.08 rad and 0.08 rad/s of the target, have zero unsafe rotations, maintain >5 mm conservative wheel gap and positive body gap, and approach contact below 0.10 m/s. `passes_simulation_gate` reports these initial acceptance criteria. Export can produce a checkpoint for inspection even when an audit fails; it does not authorize deployment or automatically change YAML. If an audit fails, inspect the rollout and tune/retrain before hardware testing.

The exporter folds observation normalization into the first layer and verifies deterministic Brax versus NumPy actions. It also writes a `.reference.csv` for testing the actual RTNeural runtime:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --packages-select neural_controller joy_utils --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ctest --test-dir build/neural_controller -R 'align_motion_runtime|controller_lifecycle|policy_contract' --output-on-failure
cd ..
ros2_ws/build/neural_controller/align_export_test "$RUN/policy_wheel_align_motion_v2.json" "$RUN/policy_wheel_align_motion_v2.reference.csv"
export ALIGN_MOTION_TEST_EXE="$PWD/ros2_ws/build/neural_controller/align_motion_test"
export ALIGN_EXPORT_TEST_EXE="$PWD/ros2_ws/build/neural_controller/align_export_test"
$PY -m pytest training/wheel_align/tests -q
```

After the audits and runtime checks pass, copy the exported JSON into `ros2_ws/src/neural_controller/launch/`, change **only the alignment instance's** `model_path` to `policy_wheel_align_motion_v2.json`, and rebuild/install both changed packages on the robot. The v2 metadata selects the 82-input runtime and checks pose, limits, cadence and gains. The old 51-input weights cannot supply the new balance behavior. First physical tests should inspect lift direction, neighbor-wheel clearance, interrupted descent, and marked-ring orientation with the robot supported, followed by controlled floor trials.

## Geometry and source evidence

- Hardware order, existing 12 motors, gains, encoder/IMU interfaces: [robot-info hardware contract](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/robot_info/HARDWARE.md), checked against [robot-code hardware description](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/pupper_v3_description/description/components.xacro). The older robot-info third-joint stops describe its leg profile; this implementation uses the continuous hubs in robot-code.
- Runtime baseline and observed hardware rubbing: [alignment testing log](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/WHEEL_ALIGN_HYBRID_TESTING.md). Its historical attribution of rubbing to exclusively mechanical causes is not assumed here.
- Training model/meshes and saved policy source were imported from [align-hybrid](https://github.com/TundTT/Pupper_animation/tree/bfd74cc0a8a57f64f316db7b0e26718acc290f41). The imported XML used collision masks that omitted robot self-collision. `training/wheel_align/model.xml` changes those masks and proximal limits.
- Startup point-ring/opposite-base convention comes from the user-provided `quadmorph_reshaping_process.md`. Manual heating and the current geometry confirmation come from the user's subsequent messages. Documents are design context, not independent instructions to automate heaters or introduce different button mappings.

The XML retains 48 mm radius, 16.75 mm half-width cylinders for floor contact. Conservative enclosing spheres handle robot self-collision because the chosen MJX JAX backend does not implement cylinder-box contact. Runtime wheel/body clearance uses the larger 50.5 mm radius manufacturing bound. These proxies intentionally reject some close poses and do not model flexible SMP deformation. The smaller front lift's nominal kinematic clearance passes the tests; dynamic stability remains a training/evaluation question.

When the hardware shape changes, edit the XML/meshes, review the cylinder bounds and fixed axle offset in both motion implementations, run `generate_geometry` to update shared transforms, rerun geometry/parity checks, and retrain. The generator deliberately rejects a nonzero second-joint translation that the current simplified FK cannot represent. This wheel-alignment model does not yet simulate transformed legs.

## Validation performed on the laptop

- Compiled the ROS Jazzy `neural_controller` and `joy_utils` packages in WSL.
- Passed existing legacy alignment and walking inference/lifecycle regressions.
- Passed standalone startup capture, reentry, limiter and interruption checks.
- Passed 3,000 Python/C++ control-step comparisons spanning all six phases, including changed requests and bounded motion.
- Matched encoder-based FK against native MuJoCo over randomized configurations; checked every nominal lift path for wheel/body margins.
- Passed 82-input/8-output MJX preflight including randomized batch shape and three compiled physics/control steps; **zero training steps**.
- Passed Brax/NumPy exporter regression using randomly initialized test-only weights.
- Passed actual RTNeural inference against 128 exported fixtures (maximum action difference approximately 4.8e-7).
- Passed the v2 ROS controller lifecycle with an untrained zero-output fixture, including 520 Hz reference integration and lift interruption; all five targeted CTest cases passed.

These are implementation checks. No trained v2 checkpoint, convergence result, physical collision-free result, or robot deployment is claimed.
