# Policy Interface Contract

## Controller And Model

The deployed plugin is `neural_controller/NeuralController`. It runs RTNeural float32 inference and requires exactly 12 outputs in canonical joint order. If model metadata omits `behavior`, the controller treats it as `locomotion` for legacy compatibility; new models must set it explicitly.

Supported activation names in the vendored dynamic RTNeural loader are:

- `tanh`
- `relu`
- `sigmoid`
- `softmax`
- `elu`

Do not export unsupported activations such as Swish. The output layer should use `tanh` so that each output is a normalized command before scaling.

## Single-Frame Observations

All vectors use the ordering shown. The controller clips the completed observation to `[-observation_limit, observation_limit]`, with a default limit of 100.

### Locomotion

Frame size: 36.

| Range | Meaning |
| --- | --- |
| 0-2 | Body angular velocity `x, y, z` |
| 3-5 | Projected gravity `x, y, z` |
| 6-8 | Commanded body velocity `vx, vy, yaw_rate` |
| 9-11 | Desired world orientation direction `x, y, z` |
| 12-23 | Joint position minus default position, canonical order |
| 24-35 | Previous normalized policy action, canonical order |

### Leg Lift

Fixed frame size: 30, plus one element for every configured command state.

| Range | Meaning |
| --- | --- |
| 0-2 | Body angular velocity `x, y, z` |
| 3-5 | Projected gravity `x, y, z` |
| 6 through `5 + N` | Command-state one-hot vector in exported `command_states` order |
| `6 + N` through `17 + N` | Joint position minus default position, canonical order |
| `18 + N` through `29 + N` | Previous normalized policy action, canonical order |

For the existing five-state convention, the frame size is 35. A different number of command states changes the model input size and must be embedded in metadata.

### Wheel

Frame size: 33.

| Range | Meaning |
| --- | --- |
| 0-2 | Body angular velocity `x, y, z` |
| 3-5 | Projected gravity `x, y, z` |
| 6-8 | Command `vx, 0, yaw_rate`; lateral command is forced to zero |
| 9-20 | Mixed joint block in canonical order |
| 21-32 | Previous normalized policy action, canonical order |

In the mixed joint block, position-controlled joints contain position minus default. Indices 2, 5, 8, and 11 contain normalized, sign-corrected wheel velocity. Training must reproduce the exact normalization and sign convention.

## Observation History

`observation_history` single frames are concatenated newest first. On controller activation, the first measured frame is copied into every history slot and previous actions are zero. Training resets should reproduce this initialization rather than filling old frames with physically impossible zeros.

The network input width must equal:

```text
single_frame_size * observation_history
```

For leg lift, `single_frame_size = 30 + len(command_states)`.

## Actions

The controller interprets each output by the configured action type:

- Position: `clip(default_joint_pos + fade * output * action_scale, policy_lower, policy_upper)`.
- Velocity: `fade * output * action_scale`.

`fade` ramps from zero to one during policy fade-in. The stored previous action is the faded normalized output, not the physical command. The controller also writes policy `kp` and `kd` values to the hardware interface.

Legacy models may use one scalar action scale. New models must export a 12-element vector so that units, wheel direction signs, and per-joint authority are explicit. Apply sign correction exactly once.

The current JSON ABI accepts `kp` and `kd` only as scalars and expands each across all 12 joints. Per-joint gains are configured as `kps` and `kds` in ROS parameters. This distinction matters for the wheel profile: its mixed position and velocity gains cannot be applied through scalar JSON overrides. Existing plural `kps` and `kds` fields in a model are descriptive metadata only; the current controller does not consume them. They must match the launch YAML used with the policy.

For every position joint, the full scaled target envelope must fit inside the hardware profile after policy clipping. For every velocity joint, the absolute scale must not exceed the hardware velocity limit.

## Timing

The current controller configuration uses a 520 Hz controller-manager update and repeats each neural action for 10 updates, giving a nominal 52 Hz policy rate. The Stanford-derived trainer uses a 0.02 second environment step and a 0.004 second physics step, corresponding to 50 Hz policy updates and five physics substeps.

Do not assume nominal timing is exact on hardware. Measure callback and sensor age distributions, keep the deployed action-repeat behavior in simulation, and randomize modestly around measured delay and jitter. Timing differences must be handled during training, not through an improvised lab wrapper.

## Activation Sequence

On activation, the controller:

1. Moves toward `default_joint_pos` using initialization gains for the configured initialization duration.
2. Begins policy inference with correctly initialized observation history.
3. Fades normalized policy output into the command over the configured fade-in duration.

Training evaluation should exercise this transition and the initial physical pose.

## Required Model Metadata

New exported JSON policies must include:

- `behavior`
- `in_shape`
- `layers`
- `observation_history`
- `observation_layout`
- `joint_names`
- `action_types`
- `action_scale`
- `default_joint_pos`
- `joint_lower_limits`
- `joint_upper_limits`
- Gain declaration: scalar `kp` and `kd` for a uniform-gain policy, or descriptive `kps` and `kds` plus matching ROS configuration for a mixed-gain policy
- `hardware_profile`
- `robot_contract_schema_version`
- `source_commit`
- `training_model_sha256`
- `training_control_dt`

Behavior-specific metadata must include `command_states` for leg lift and command bounds for commanded locomotion or wheel policies. An orientation command must contain three finite values and represent a unit direction vector.

The controller already checks important embedded metadata when present, including joint names, action types, input shape, output count, command bounds, and orientation command. Metadata is part of the deployment ABI, not optional documentation.
