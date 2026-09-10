# Reference Implementations

This map answers “where should I start?” without declaring any experiment current, deployed, best, or pending. Use the pinned code as an implementation reference, then apply the authoritative contract in this directory.

## Runtime Integration

These paths are present on this branch:

| Concern | Path |
| --- | --- |
| Neural controller | `ros2_ws/src/neural_controller/src/neural_controller.cpp` |
| Controller parameters | `ros2_ws/src/neural_controller/src/neural_controller_parameters.yaml` |
| Policy-contract helpers | `ros2_ws/src/neural_controller/include/neural_controller/policy_contract.hpp` |
| Controller launch configuration | `ros2_ws/src/neural_controller/launch/config.yaml` |
| Controller contract test | `ros2_ws/src/neural_controller/test/walk_policy_test.cpp` |
| Robot hardware description | `ros2_ws/src/pupper_v3_description/description/components.xacro` |
| Physical hardware interface | `ros2_ws/src/control_board_hardware_interface/` |
| MuJoCo ROS hardware interface | `ros2_ws/src/pupperv3_mujoco_sim/` |
| General offline policy checker | `robot_info/validate_policy.py` |
| Walking parity and provenance checker | `scripts/check_walk_policy.py` |

Read the controller before changing a training tensor. It is the deployed consumer of observations and model metadata.

## Stanford-Derived Locomotion

Pinned baseline commit: `13c46c8630a682c9bfbfb5699a9b89db4ac06c5a`.

| Concern | Historical path |
| --- | --- |
| Modular trainer | `Stanford/pupperv3-monorepo/ai/rl/pupper_mjx_rl_training.py` |
| Hydra configuration | `Stanford/pupperv3-monorepo/ai/rl/conf/config.yaml` |
| Configuration test | `Stanford/pupperv3-monorepo/ai/rl/test_config.py` |
| Training guide | `Stanford/pupperv3-monorepo/ai/rl/README.md` |
| Public notebook | `Stanford/training/Pupper_RL_PUBLIC.ipynb` |

Example inspection:

```powershell
git show 13c46c8630a682c9bfbfb5699a9b89db4ac06c5a:Stanford/pupperv3-monorepo/ai/rl/pupper_mjx_rl_training.py
```

Use this architecture for a new locomotion-style policy and replace historical numeric values with the current `leg_position` contract.

## Leg Lift

Pinned all-four-leg behavior reference: `0a2b6cf309a0bfc4280f2df8b9971fe5a8787238`.

| Concern | Historical path |
| --- | --- |
| Task environment | `mujoco_playground/workspace/leg_lift_env.py` |
| Behavior-aware exporter | `mujoco_playground/workspace/export_policy.py` |
| ROS controller integration | `Stanford/pupperv3-monorepo/ros2_ws/src/neural_controller/src/neural_controller.cpp` |
| Controller configuration | `Stanford/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/config.yaml` |

Later training harness reference containing `train_leg_lift.py`: commit `6b55e30ff224193a2a21c8dc3dbb88e029a417c8`.

```powershell
git show 0a2b6cf309a0bfc4280f2df8b9971fe5a8787238:mujoco_playground/workspace/leg_lift_env.py
git show 6b55e30ff224193a2a21c8dc3dbb88e029a417c8:mujoco_playground/workspace/train_leg_lift.py
```

The reusable interface is the `leg_lift` observation with an explicit command-state one-hot vector and position actions under the `leg_position` profile. Task rewards and training results are examples, not compatibility authority.

## Wheel Locomotion

Pinned wheel-profile reference: `5a3b980057c0a91975431239c1d714a37d8c472b`.

| Concern | Historical path |
| --- | --- |
| Wheel task environment | `mujoco_playground/workspace/wheel_env.py` |
| Behavior-aware exporter | `mujoco_playground/workspace/export_policy.py` |
| Wheel robot hardware profile | `Stanford/pupperv3-monorepo/ros2_ws/src/pupper_v3_description/description/components.xacro` |
| Matching controller configuration | `Stanford/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/config.yaml` |
| Keyboard command utility | `Stanford/pupperv3-monorepo/scripts/wheel_keyboard_teleop.py` |

```powershell
git show 5a3b980057c0a91975431239c1d714a37d8c472b:mujoco_playground/workspace/wheel_env.py
git show 5a3b980057c0a91975431239c1d714a37d8c472b:Stanford/pupperv3-monorepo/ros2_ws/src/pupper_v3_description/description/components.xacro
```

The wheel description, controller configuration, and policy form one hardware profile. Reusing only the network while leaving the leg description active is an incompatible partial deployment.

## Selection Rule

Prefer, in order:

1. The closest behavior-specific environment above.
2. The verified Stanford-derived trainer architecture.
3. The current runtime controller and hardware source.
4. New code only for behavior-specific functionality not already represented.

Before training, reconcile the chosen reference with `robot_info/robot_contract.json`. Pinned references preserve provenance; the contract preserves present compatibility.
