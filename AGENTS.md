# Required robot startup

Read [STARTUP_UPPER_HOME.md](STARTUP_UPPER_HOME.md) before any stack launch. The operator requires **automatic mechanical-stop homing of motors 1 and 2, then return to the saved upper home on every fresh hardware activation**. Use `bash scripts/run_robot_stack.sh --supported` after actual support confirmation; confirmation already provided for the current setup persists. Do not ask the user to manually pose the upper legs. Keep hubs unpowered, then obtain manual marked-hub alignment confirmation and capture the live-session calibration before enabling policies. Reuse a running valid session when switching policies.

Do not substitute the old hanging reference, Stanford zero, or a policy stance. The runtime home is `ros2_ws/src/pupper_v3_description/description/upper_home.yaml`; its source is `hardware_testing/start_pose/upper_home.json`. Rebuild after changing it. Tests that disable after homing are diagnostics, not the normal startup path. Historical startup narratives in other files are superseded by STARTUP_UPPER_HOME.md.

The robot checkout is `/home/pi/robot-code-leglift`. Inspect processes and overlay before starting; never run a duplicate hardware owner or the other robot's checkout. Missing support confirmation, homing failure, or missing current-session hub calibration must not be bypassed. Heating remains manual. Read-only inspection and source edits do not require physical calibration. Cite hardware source and distinguish observed results from hypotheses.

# Latest leg and wheel policies

For locomotion policy work, read [LATEST_POLICIES.md](LATEST_POLICIES.md) first.
`policies/latest.json` identifies the current backpack + 9 mm leg and wheel exports.
They replace the existing walking and wheel controller files and are already wired
in source. Do not substitute historical gap-only policies or the legacy
`policy_latest.json`. Physical installation and hardware validation remain pending.

# Policy training and experiment logging

The user's standing preference is to log future policy training to Weights & Biases and include a real checkpoint rollout video in the run's Media tab.

- Default destination: entity `QuadMorph`, project `wheel-leg lift and align triangle base`, unless the user names another destination.
- Reuse `training/wandb_logging.py` where available. Enable online logging by default, retain local logs/checkpoints, support explicit offline operation, and accurately report whether data reached W&B.
- Log actual policy rollouts as `wandb.Video`, not only file/artifact attachments. Label simulation, seed, checkpoint step and early termination; preserve failed audits. Video quality/reward is not proof of passing the task.
- Store configuration, source provenance, environment steps, evaluation metrics and checkpoint/export hashes. Use existing credentials or `wandb login`; never put API keys in source, logs, commits or chat.
- Preserve original training checkouts for historical replay and do not bypass provenance checks. PC training does not authorize training on another machine or deploying to the robot.
- For alignment training, read `AGENT_TRAINING_HANDOFF.md`, `WANDB_LOGGING.md` and `ALIGN_MOTION_V2.md` on `origin/codex/align-motion-v2` (use `git show` if those files are absent here). Those are training-branch instructions, not authorization to start training or merge that branch into `robot-code`. Current robot startup follows STARTUP_CALIBRATION.md here.
