> Current startup procedure: [STARTUP_CALIBRATION.md](STARTUP_CALIBRATION.md). Every fresh stack startup needs physical confirmation and saved session calibration before policy activation. Historical X-time capture/recalibration instructions below are superseded.

# Wheel-align Phase 1 export

The requested checkpoint is saved as
`ros2_ws/src/neural_controller/launch/policy_wheel_align.json`.
It is a valid RTNeural dense-network JSON, with the alignment contract and
training provenance included. **It is not yet a loadable hardware controller.**
No controller instance or joystick binding was added; `config.yaml`, `launch.py`,
and the existing policies are unchanged.
The repository's existing Git LFS rule tracks the 2,201,938-byte JSON. Its local
LFS object is saved alongside the commit; no Git or LFS objects were pushed.

Source: `/home/theerawit/Pupper_animation-combine/mujoco_playground/workspace/output/wheel-align_2026-09-07_20-41-57/mjx_params_best`.
Checkpoint SHA256:
`04d1bcf07ef7b37a6416d0c6835e987417e659611e6035b722b1cbca70ad99a5`.
This is the selected **4,014,080-step** checkpoint of the second refinement,
not its final 12,042,240-step checkpoint. Its saved `robustness_comparison.json`
reports **45/64** full successes, no falls, 4.804 cm p95 drift, and 0.02481 rad
p95 held error. The **46/64** result belongs to the preceding
`wheel-align_2026-09-07_20-25-49` run. The explicit requested path was retained.
These are prior simulation results, not new hardware validation.

The exporter reads the run's saved `config.json` and frozen
`training_source/configs.py`. The JSON includes the full training config,
checkpoint/config/source hashes, selection metrics, and robustness comparison.
It requires no JAX/Brax installation for inference through RTNeural.

The network is `376 -> 128 ELU -> 128 ELU -> 3 tanh`. The Gaussian scale outputs
and critic are discarded; deterministic inference uses the three actor means
followed by tanh. For input-by-output first-layer weights, normalization is
folded as `W' = W / std[:,None]`, `b' = b - (mean/std) @ W`. The saved standard
deviations are used directly, with no extra epsilon or normalized-input clip.
Runtime callers supply the pre-normalization observation, clipped to [-100,100].

Each frame has **94 entries**, and four frames are packed newest first. The
JSON's `observation_layout` gives exact offsets, sizes, and transformations.
Reset writes the first observed frame followed by **three zero frames**;
this differs from the current robot controller's repeated-first-frame startup.
The simulation's `set_command` also rebuilds and shifts observations when an
external command is applied; a port must preserve observation/update ordering.

Leg-indexed arrays use `front_r, front_l, back_r, back_l`. Commands use
`0=stand, 1=front_l, 2=front_r, 3=back_r, 4=back_l`.
The joint block still contains relative angles on abduction/hip rows and
sign-corrected wheel speeds divided by 20.833333333333332 on wheel rows.
Wheel target and reference errors are trigonometric angle features in separate
blocks; raw continuous wheel angles do not replace the speed observations.

The three outputs mean clearance trim ±0.001 m around 0.005 m, stance X correction
±0.012 m, and stance Y correction ±0.012 m. They are inputs to a stateful feedback
controller, not 12 joint offsets or wheel velocities. Accordingly, the export
uses `policy_action_scale` and nests physical joint gains under
`feedback_controller`; it does not declare misleading direct-joint
`action_scale`/`action_types` metadata.

## Why there is no named ROS instance yet

`NeuralController::on_init()` requires exactly 12 outputs and supports only
`locomotion`, `leg_lift`, and `wheel` observation layouts. This network has three
outputs and declares `behavior: wheel_align`. Copying the leg-lift YAML block
would fail configuration, and padding/remapping the outputs cannot reproduce
the trained controller. The existing launch-directory install rule already
includes the new JSON in the package share directory.

The frozen `training_source/wheel_align_env.py` contains the required adapter:
`_select`, `_leg_action`, `_advance_wheel_reference`, `_advance_phase`, and
`_position_ctrl`, plus observation and reset state. It includes support shifts
derived from wheel-geometry Jacobians, clearance feedback, selected-hip slew
limits, dwell/contact guards, and persistent calibrated targets. Wheel position
PD uses kp=2, kd=0.35 and the nearest equivalent unwrapped target, refreshed every
physics step (0.004 s); the policy runs at 0.02 s. Wheels have no absolute angle
clamp. The older wheel-drive velocity controller does not implement these rules.

Before hardware integration, resolve these concrete inputs and choices:

- Provide validated per-wheel ground clearance and normal-load estimates, plus
  body displacement, body linear velocity, and heading relative to activation.
  Training reads these from simulation; the current controller's IMU/joint
  observation builder does not supply them. Choose the hardware sensing or
  estimation source, ground-plane reference, and wheel geometry calibration.
- Define how each wheel's physical `calibration_home` is acquired and how
  tracked/completed/target state survives mode switches. A target is a latched
  calibration plus pi, never current encoder angle plus pi on every selection.
- Port and validate the feedback adapter, including startup, history, interrupted
  rotations, lowering/reweighting, high-rate wrapped position PD, and external
  command handling (`automatic_commands=False`). Then add its controller instance
  and inactive spawner, and register it for deactivation on mode switches.
- Choose a joystick button/command route for alignment; no existing binding was
  reassigned or guessed. Command indices themselves are fixed by the policy.

## Reproduce export and verify inference

From the `robot-code` worktree, use the existing combine training environment
(Brax 0.14.2, JAX 0.6.2, NumPy 2.2.6, ml-collections 1.1.0):

```bash
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  /home/theerawit/Pupper_animation-combine/mujoco_playground/.venv/bin/python \
  scripts/export_wheel_align_policy.py \
  --params /home/theerawit/Pupper_animation-combine/mujoco_playground/workspace/output/wheel-align_2026-09-07_20-41-57/mjx_params_best \
  --out /tmp/policy_wheel_align.json \
  --reference-out /tmp/wheel-align-reference.csv
cmp /tmp/policy_wheel_align.json ros2_ws/src/neural_controller/launch/policy_wheel_align.json

# Set CXX to an available C++17 compiler if c++ is not on PATH.
"${CXX:-c++}" -std=c++17 -O2 \
  -DRTNEURAL_USE_EIGEN=1 -DRTNEURAL_DEFAULT_ALIGNMENT=16 \
  -Iros2_ws/src/neural_controller/modules/RTNeural \
  -Iros2_ws/src/neural_controller/modules/RTNeural/modules/Eigen \
  -Iros2_ws/src/neural_controller/modules/RTNeural/modules/json \
  ros2_ws/src/neural_controller/test/wheel_align_policy_test.cpp \
  -o /tmp/wheel-align-policy-test
/tmp/wheel-align-policy-test \
  ros2_ws/src/neural_controller/launch/policy_wheel_align.json \
  /tmp/wheel-align-reference.csv
```

The exporter checks serialized float32 weights against the original Brax
deterministic inference path on 512 seeded inputs, including all command indices,
zero-history startup, and clipping boundaries. The standalone C++ test parses the
JSON with the same RTNeural Eigen backend as the ROS plugin and checks the same
cases against Brax. Both passed: maximum absolute action error was `3.854e-6`
for NumPy/JSON and `1.289e-6` for RTNeural (tolerance `3e-4`). The C++ check used
`/home/theerawit/micromamba/envs/ros_jazzy/bin/x86_64-conda-linux-gnu-c++`.
These checks validate network export/inference, not the
missing hardware observation and feedback adapter. Temporary fixtures/binaries
are not committed, and the test needs no robot connection.
