# Robot startup and calibration

Latest user request (September 12): retain the successful repeat-floor-test home
as a permanent, easily replaceable start pose and eliminate routine manual
recapture. See `hardware_testing/start_pose/README.md` and `start_pose.json` there.
The preset is saved, but automatic physical homing is not implemented: verify
motor feedback across power loss before replacing the current offset workflow.
The following instructions describe the currently installed startup procedure.

Read [STARTUP_CALIBRATION.md](STARTUP_CALIBRATION.md) before any hardware-stack startup, restart, ordinary use, branch test or policy test. This is the user's standing requirement.

- Before a fresh startup, prompt the user to support the robot in the documented encoder-homing pose and position the marked wheel rings for the agreed calibration reference. **Wait for their explicit confirmation before starting the hardware stack.** A request to start/test is not evidence that physical positioning is complete.
- After startup/homing, capture and save the shared calibration before enabling policy-controlled motion. Use `python3 scripts/calibrate_robot.py capture`; an agent may add `--operator-confirmed` only after the user's actual confirmation for that startup. Never self-confirm, guess values or disable the hardware gate.
- Check `python3 scripts/calibrate_robot.py status`. Report the calibration ID, wheel homes and storage path. Reuse valid calibration when switching policies within the same live encoder session. A fresh hardware activation/zeroing requires a new confirmed capture; a saved old file is not sufficient.
- Inspect existing processes and the selected overlay before launching. Do not start a duplicate stack or launch the other robot's checkout. If the reference pose or physical marks are unclear, ask the user instead of treating a legacy comment as hardware truth.
- Ordinary read-only inspection, local tests and source edits do not require physical calibration. Calibration does not authorize deployment, training, or additional robot motion. Heating remains manual.
- Cite the actual hardware source for hardware claims; distinguish user-confirmed facts, implementation facts, legacy comments and hypotheses.

# Policy training and experiment logging

The user's standing preference is to log future policy training to Weights & Biases and include a real checkpoint rollout video in the run's Media tab.

- Default destination: entity `QuadMorph`, project `wheel-leg lift and align triangle base`, unless the user names another destination.
- Reuse `training/wandb_logging.py` where available. Enable online logging by default, retain local logs/checkpoints, support explicit offline operation, and accurately report whether data reached W&B.
- Log actual policy rollouts as `wandb.Video`, not only file/artifact attachments. Label simulation, seed, checkpoint step and early termination; preserve failed audits. Video quality/reward is not proof of passing the task.
- Store configuration, source provenance, environment steps, evaluation metrics and checkpoint/export hashes. Use existing credentials or `wandb login`; never put API keys in source, logs, commits or chat.
- Preserve original training checkouts for historical replay and do not bypass provenance checks. PC training does not authorize training on another machine or deploying to the robot.
- For alignment training, read `AGENT_TRAINING_HANDOFF.md`, `WANDB_LOGGING.md` and `ALIGN_MOTION_V2.md` on `origin/codex/align-motion-v2` (use `git show` if those files are absent here). Those are training-branch instructions, not authorization to start training or merge that branch into `robot-code`. Current robot startup follows STARTUP_CALIBRATION.md here.
