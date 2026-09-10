# Hardware Contract

## Compute And Link

- Computer: Raspberry Pi 5 Model B Rev 1.1, ARM64.
- Processor: four Cortex-A76 cores.
- Memory: approximately 7.9 GiB.
- Accelerator: Hailo-8 PCIe device (`1e60:2864`).
- Motor-control transport: SPI devices exposed by the operating system.
- Storage observed during the 2026-09-09 audit: approximately 118 GiB root filesystem.

Storage capacity is descriptive rather than an API. The processor architecture, SPI transport, and available memory directly affect build and deployment choices.

## Canonical Joint Order

Every observation, action, gain, limit, and model metadata vector uses this order:

| Index | Joint | CAN channel | Motor ID |
| ---: | --- | ---: | ---: |
| 0 | `leg_front_r_1` | 2 | 1 |
| 1 | `leg_front_r_2` | 2 | 2 |
| 2 | `leg_front_r_3` | 2 | 3 |
| 3 | `leg_front_l_1` | 1 | 1 |
| 4 | `leg_front_l_2` | 1 | 2 |
| 5 | `leg_front_l_3` | 1 | 3 |
| 6 | `leg_back_r_1` | 4 | 1 |
| 7 | `leg_back_r_2` | 4 | 2 |
| 8 | `leg_back_r_3` | 4 | 3 |
| 9 | `leg_back_l_1` | 3 | 1 |
| 10 | `leg_back_l_2` | 3 | 2 |
| 11 | `leg_back_l_3` | 3 | 3 |

CAN channel assignment is front-left 1, front-right 2, back-left 3, and back-right 4. Motor IDs are the `_1`, `_2`, and `_3` suffixes.

## Leg Position Profile

All angles are radians. The soft position limits are the usable hardware envelope supplied to ROS control. Current homed positions are operator-confirmed gravity-droop poses.

| Joint | Homed position | Soft minimum | Soft maximum | Velocity maximum |
| --- | ---: | ---: | ---: | ---: |
| `leg_front_r_1` | 0.990537 | -1.12 | 2.41 | 30 |
| `leg_front_r_2` | -0.179972 | -0.32 | 3.04 | 30 |
| `leg_front_r_3` | -1.045141 | -2.69 | 0.61 | 15 |
| `leg_front_l_1` | -0.990539 | -2.41 | 1.12 | 30 |
| `leg_front_l_2` | 0.179973 | -3.04 | 0.32 | 30 |
| `leg_front_l_3` | 0.950836 | -0.61 | 2.69 | 15 |
| `leg_back_r_1` | 0.990537 | -1.12 | 2.41 | 30 |
| `leg_back_r_2` | -0.179972 | -0.32 | 3.04 | 30 |
| `leg_back_r_3` | -1.045141 | -2.69 | 0.61 | 15 |
| `leg_back_l_1` | -0.990539 | -2.41 | 1.12 | 30 |
| `leg_back_l_2` | 0.179973 | -3.04 | 0.32 | 30 |
| `leg_back_l_3` | 0.950836 | -0.61 | 2.69 | 15 |

Shared limits are 3.0 Nm effort, 10 maximum proportional gain, and 1 maximum derivative gain.

The third-joint hard calibrated ranges currently represented in source are:

| Joints | Hard minimum | Hard maximum |
| --- | ---: | ---: |
| Right `_3` joints | -2.615937 | 0.525655 |
| Left `_3` joints | -0.619960 | 2.521632 |

Raw encoder references in the current description are `-0.015800`, `0.103600`, `-0.202800`, and `-0.517100` for front-right, front-left, back-right, and back-left `_3` joints respectively. They are informational calibration records, not a runtime cross-check. Homing velocity, homing proportional gain, and homing threshold are zero in the current leg profile. Comments that still describe a fixed-reference cross-check may be stale.

## Wheel Velocity Profile

The wheel configuration changes every `_3` joint from position to velocity control. Its description values come from the hardware-tested wheel profile at `origin/wheel` commit `5a3b980057c0a91975431239c1d714a37d8c472b`, not from the default leg `components.xacro` on this branch. A wheel deployment must bring over the matching wheel robot description, controller configuration, and policy as one profile; changing only the policy leaves third-joint velocity clamped by the leg profile.

Inspect the wheel hardware source without switching branches:

```powershell
git show origin/wheel:Stanford/pupperv3-monorepo/ros2_ws/src/pupper_v3_description/description/components.xacro
git show origin/wheel:Stanford/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/config.yaml
```

The wheel action types are:

```text
position, position, velocity,
position, position, velocity,
position, position, velocity,
position, position, velocity
```

For wheel joints:

- Homing is disabled.
- Position limits use non-operative sentinels `[-1000, 1000]`.
- Velocity maximum is 30 rad/s.
- Reference policy gains are `kp = 0`, `kd = 0.35`.
- Reference initialization gains are `kp = 0`, `kd = 0.1`.
- Reference emergency-stop derivative gain is 1.0.
- Wheel angle is not a useful policy state; normalized, sign-corrected velocity is observed instead.
- Direction correction belongs in the action scale exactly once.

Position-controlled joints in the wheel profile use reference policy gains `kp = 5`, `kd = 0.25` and initialization gains `kp = 7.5`, `kd = 0.25`.

## IMU Contract

- Device path: BNO055 hardware integration.
- Mount transform: yaw `0`, pitch `-2.35619`, roll `0` radians.
- Requested sensor period: 10,000 microseconds.
- Orientation state order: quaternion `x, y, z, w`.
- Angular velocity order: `x, y, z`.
- Linear acceleration order: `x, y, z`.
- Additional hardware interface: `time_since_measurement_seconds`.

The neural controller computes projected gravity by rotating the world vector `[0, 0, -1]` into the body frame using the measured quaternion. Training must use the same convention.

## Source Locations

Use these paths when verifying or changing the contract:

- `ros2_ws/src/pupper_v3_description/description/components.xacro`
- `ros2_ws/src/pupper_v3_description/description/pupper_v3.urdf.xacro`
- `ros2_ws/src/pupper_v3_description/description/pupper_v3.ros2_control.xacro`
- `ros2_ws/src/control_board_hardware_interface/`
- `ros2_ws/src/neural_controller/`
- `Stanford/pupperv3-monorepo/ros2_ws/src/`

Any physical recalibration must update source, `robot_info/robot_contract.json`, and this document in the same change.
