# Robot Compatibility Reference

This branch is the authoritative engineering reference for creating software and policies that are compatible with this physical Pupper robot before entering the lab. The objective is compatibility by construction: reuse the verified Stanford-derived interfaces, train against the deployed observation and action contracts, and avoid adding a translation or adaptation layer after training.

This branch does not track project progress, deployed-policy status, or which experiment is currently best. Those records should remain in task-specific documents that are curated manually.

## Agent Workflow

When asked to create robot code or train a policy:

1. Read this file and the relevant files in `robot_info/`.
2. Select an existing hardware profile instead of inventing a new interface.
3. Start from the Stanford-derived training and controller paths described in `robot_info/BASELINE.md`.
4. Implement the deployed observation, action, timing, and metadata contracts before training.
5. Export a self-describing RTNeural JSON model and run `robot_info/validate_policy.py`.
6. Complete the checks in `robot_info/LAB_READY.md` before declaring the work ready for hardware testing.

Do not optimize only for simulation videos and defer robot integration. A policy is not complete if its joint order, observation layout, action interpretation, timing, or runtime dependencies still require reconstruction in the lab.

## Reference Map

- `robot_info/BASELINE.md`: verified upstream provenance and reusable starting points.
- `robot_info/REFERENCE_PATHS.md`: pinned trainers, exporters, controller paths, and behavior-specific starting points.
- `robot_info/HARDWARE.md`: physical robot, actuator, CAN, limit, and IMU contracts.
- `robot_info/SOFTWARE.md`: target-computer runtime, ROS overlay, build, and launch assumptions.
- `robot_info/POLICY_INTERFACE.md`: controller observations, actions, history, timing, and model format.
- `robot_info/TRAINING.md`: sim-to-real workflow and export requirements.
- `robot_info/LAB_READY.md`: pre-lab and hardware-test checklists.
- `robot_info/robot_contract.json`: machine-readable compatibility data.
- `robot_info/validate_policy.py`: offline policy metadata and hardware-envelope checker.

## Authority Order

When references disagree, resolve them in this order:

1. A repeatable physical measurement on this robot.
2. `robot_info/robot_contract.json` and the documents in `robot_info/`.
3. The current controller, hardware, and robot-description source code.
4. The verified imported Stanford baseline at commit `13c46c8`.
5. Task-specific notes, old launch files, comments, and experiment logs.

Do not silently choose between conflicting values. Record the evidence, update the machine-readable contract and relevant document together, then update affected code or training configurations.

## Compatibility Standard

Robot-ready work should be approximately drop-in compatible with the target system:

- Canonical 12-joint order and correct action type are fixed before training.
- The training observation is identical to the deployed controller observation.
- Policy outputs already have the controller's expected meaning and scale.
- The model is exported in the RTNeural format supported on the Raspberry Pi.
- Required ROS packages and launch paths exist in the target overlay.
- A clean target-architecture build succeeds before the lab session.
- Remaining lab work is controlled testing and feedback, not interface redesign.

Small parameter tuning is expected. Rewriting tensor layouts, remapping joints, replacing unsupported activations, adding runtime observation conversion, or repairing a target build is not considered small tuning.

## Quick Checks

Validate a policy without connecting to the robot:

```powershell
python robot_info/validate_policy.py path/to/policy.json --profile leg_position --strict
python robot_info/validate_policy.py path/to/policy.json --profile wheel_velocity --strict
```

Inspect the original imported baseline without changing branches:

```powershell
git show 13c46c8:Stanford/pupperv3-monorepo/ai/rl/README.md
git show 13c46c8:Stanford/training/Pupper_RL_PUBLIC.ipynb
```

Never store robot passwords, private keys, or access tokens in this branch.
