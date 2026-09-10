# Software And Runtime Contract

## Audited Robot Environment

The following was observed read-only on the robot on 2026-09-09:

| Component | Value |
| --- | --- |
| Operating system | Debian GNU/Linux 12 (bookworm) |
| Kernel | `6.12.47+rpt-rpi-2712` |
| Architecture | ARM64 |
| ROS distribution | Jazzy at `/opt/ros/jazzy` |
| Python | 3.11.2 |
| GCC/G++ | 12.2.0 |
| CMake | 3.25.1 |
| Git | 2.39.5 |
| Git LFS | 3.3.0 |
| colcon-core | 0.20.0 |

The exact patch versions may change. Code must remain compatible with Debian 12, ROS 2 Jazzy, Python 3.11, GCC 12, CMake 3.25, and ARM64 unless the target image is deliberately upgraded and this contract is updated.

## Required ROS Capability

The target image includes the core packages used by this stack, including:

- `controller_manager`
- `ros2_control`
- `ros2_controllers`
- `joy_linux`
- `teleop_twist_joy`

The 2026-09-09 audit did not find `foxglove_bridge`, `camera_ros`, `topic_tools`, or `vision_msgs`. Treat those as optional unless a behavior explicitly declares them as a tested dependency. Missing visualization, camera, or relay packages must not prevent core motor-control launch.

## Workspace And Overlay

The project checkout used for this robot is `/home/pi/robot-code-leglift`, with ROS workspace `/home/pi/robot-code-leglift/ros2_ws`. A separate checkout at `/home/pi/pupperv3-monorepo` belongs to another robot and must not be modified or launched for this target.

Source order is mandatory:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/pi/robot-code-leglift/ros2_ws
source install/local_setup.bash
```

Without both setup files, ROS package discovery can report project packages as missing even when the build exists.

Use a clean, targeted build before hardware testing:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/pi/robot-code-leglift/ros2_ws
colcon build --symlink-install
source install/local_setup.bash
ros2 pkg prefix neural_controller
```

The neural controller built successfully on this target during the audit. That result is evidence that the present toolchain can build it, not permission to skip future clean-build checks.

## Launch Integration

The D-pad launch service invokes:

```text
/usr/bin/python3 /home/pi/robot-code-leglift/scripts/dpad_launch_trigger.py
```

Scripts intended for that service must run under the system Python and must not depend on an unactivated home-directory virtual environment. Launch files should make nonessential visualization and camera nodes optional.

## Dependency Rules

- Pin Python training dependencies with the existing `uv` workflow when using the modular Stanford-derived trainer.
- Record any native library required by the deployed controller.
- Do not assume x86 wheels or binaries will work on ARM64.
- Resolve Git LFS policy files before copying or building a policy.
- Verify that JSON model files contain JSON, not a Git LFS pointer.
- Do not add a runtime dependency solely to translate a policy into the documented interface.

## Reproducibility Record

Every policy handoff should identify:

- Source commit used to train and export.
- Training configuration and random seeds.
- Python environment lockfile.
- Robot contract schema version and selected profile.
- Model SHA-256 hash.
- Target build command and result.
- Any optional ROS dependencies required by its launch file.

Never commit robot passwords, private keys, tokens, Wi-Fi credentials, or temporary access instructions. Network addresses are operational details, not compatibility contracts.
