# Policy training and experiment logging

The user's standing preference is to log future policy training to Weights & Biases and include a real policy rollout video in the run's Media tab.

- Default destination: entity `QuadMorph`, project `wheel-leg lift and align triangle base`, unless the user names another destination.
- Reuse `training/wandb_logging.py` for experiment identity, metrics, media and artifacts where applicable. New training entry points should enable online logging by default, retain local logs/checkpoints, and upload a final policy video. Support explicit offline operation when needed, and accurately report whether data reached W&B.
- Log videos as `wandb.Video`, not only as an artifact/file attachment. Record an actual checkpoint rollout; label simulation, seed, checkpoint step and any early termination. Preserve failed-audit results. A good-looking video or reward is not proof that the policy passes its task criteria.
- Store run configuration, source provenance, environment steps, evaluation metrics and checkpoint/export hashes. Keep API keys out of source, logs, commits and chat; authenticate with `wandb login` or the user's existing credential setup.
- Preserve the original training checkout when backfilling historical runs. Use the checkpoint's original environment/control implementation, and do not bypass its provenance checks just to produce a video.
- Training on the PC does not authorize training on another machine or deploying to the robot. Follow the user's compute and deployment instructions.
- For wheel alignment, read `AGENT_TRAINING_HANDOFF.md`, `WANDB_LOGGING.md` and `ALIGN_MOTION_V4.md`. `ALIGN_MOTION_V2.md` is historical context. Cite the hardware source when making hardware claims; distinguish the user's confirmed facts from legacy comments and hypotheses.
