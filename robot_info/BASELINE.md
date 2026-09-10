# Verified Baseline

## Provenance

The repository began with working code imported in commit `13c46c8` (`Initial commit: Pupper V3 transition animation workspace`). Treat that commit as the clean historical reference for the Stanford-derived architecture, not as a claim that every old constant is still correct for the present robot.

| Reference | Location | Use |
| --- | --- | --- |
| Pupper V3 monorepo snapshot | `Stanford/pupperv3-monorepo/` | ROS hardware, controllers, descriptions, launch structure, and modular RL code |
| Public training notebook | `Stanford/training/Pupper_RL_PUBLIC.ipynb` | End-to-end training and export reference |
| Training robot description | `Stanford/training/pupper_v3_description/` | Simulation model used by the imported training workflow |
| Modular RL guide | `Stanford/pupperv3-monorepo/ai/rl/README.md` | Environment setup, configuration, training, and export workflow |

The imported monorepo identifies `Nate711/pupperv3-monorepo` as its source. Portions of the robot description and neural-controller lineage also identify G-Levine sources. Preserve those upstream references when tracing behavior or comparing implementations.

## Inspect Without Switching Branches

Use Git to inspect the baseline directly:

```powershell
git show 13c46c8:Stanford/pupperv3-monorepo/ai/rl/README.md
git show 13c46c8:Stanford/pupperv3-monorepo/ai/rl/pupper_mjx_rl_training.py
git show 13c46c8:Stanford/training/Pupper_RL_PUBLIC.ipynb
git ls-tree -r --name-only 13c46c8 Stanford/pupperv3-monorepo
```

## What To Preserve

Reuse the baseline's interfaces and deployment path wherever possible:

- ROS 2 control hardware and controller separation.
- Joint naming and 12-element ordering.
- MuJoCo/MJX environment structure.
- Observation-history construction.
- Position-target policy convention for legged behaviors.
- RTNeural JSON export and C++ inference path.
- Domain randomization for latency, sensor noise, gains, friction, mass, center of mass, and disturbances.

The modular baseline uses `uv sync`, `uv run python test_config.py`, and `pupper_mjx_rl_training.py` with Hydra configuration. New training code may improve organization, but it should retain the deployed mathematical contract.

## What Not To Freeze

Historical code can contain values for a different calibration, mechanical configuration, or software image. Do not copy these without checking the current contract:

- Homed joint positions and encoder references.
- Joint limits, gains, action scales, and wheel behavior.
- Controller update rates and action repetition.
- Installed ROS packages and launch-time optional nodes.
- Robot filesystem paths.

The current authoritative values are in `robot_info/robot_contract.json` and summarized in the other files in this directory.

## Reuse Rule

For a new behavior, begin with the closest existing profile:

- Use `leg_position` for locomotion, stance, pose, leg lift, and other position-controlled leg behaviors.
- Use `wheel_velocity` only when the third joint of each leg is configured as a velocity-controlled wheel.

If neither profile represents the physical system, first document and validate a new profile. Do not hide the mismatch inside a policy wrapper.
