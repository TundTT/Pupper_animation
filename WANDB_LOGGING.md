# W&B logging and policy videos

Default destination: [QuadMorph / wheel-leg lift and align triangle base](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/workspace).

The entity is `QuadMorph`; the project name contains literal spaces: `wheel-leg lift and align triangle base`. Do not pass the workspace URL or URL-encoded `%20` text as the SDK project name.

## PC agent: upload the completed September 10 run

The user authorized uploading this run, including its failed audits and a video. **No retraining is required or requested for this upload.** The PC agent reported the original checkout at `/home/theerawit/Pupper_animation-align-motion-v2`, source commit `4d9f1df`, and run `runs/align-motion-v2-seed0-20260910T191937Z`.

Keep that original checkout unchanged. Its local exporter precision fix was not present on the remote branch when this integration was written. A normal pull/merge there would change source hashes and prevent faithful historical replay. Run the new uploader from a separate worktree instead:

```bash
ORIGINAL=/home/theerawit/Pupper_animation-align-motion-v2
TOOLS=/home/theerawit/Pupper_animation-wandb
RUN="$ORIGINAL/runs/align-motion-v2-seed0-20260910T191937Z"
git -C "$ORIGINAL" status --short
git -C "$ORIGINAL" fetch origin
# If TOOLS already exists, inspect it and reuse/update an appropriate tools
# checkout; do not overwrite it or create a second training process.
git -C "$ORIGINAL" worktree add --detach "$TOOLS" origin/codex/align-motion-v2
cd "$TOOLS"
uv sync --project training/wheel_align --extra cuda --frozen
PY="$TOOLS/training/wheel_align/.venv/bin/python"
"$PY" -m wandb login
MUJOCO_GL=egl "$PY" -m training.wheel_align.log_run \
  --run-dir "$RUN" --source-root "$ORIGINAL"
```

Use the existing W&B credentials or have the user complete the CLI login locally. Do not ask the user to paste an API key into chat, and do not put it in a command history, commit or report. The account must have write access to the named project. Authentication or access errors are real blockers, not a reason to silently change projects or claim that an offline run was uploaded.

`log_run` reads the actual `config.json`, `metrics.jsonl`, audits and final checkpoint. It replays history using environment steps as the chart axis, uploads checkpoint/audit/export artifacts, and launches a separate video worker that imports the **original checkout's** environment and motion implementation after checking its source hashes. It does not weaken audit criteria or change the old run's configuration.

The video is a full nominal four-wheel sequence, or ends at the first termination. It uses the saved trained actor in MJX, then renders the recorded positions with MuJoCo. The overlay shows phase, active/requested commands, trajectory progress, wheel target errors and completed-wheel count. A CSV contains target/actual wheel angles and the same phase/clearance diagnostics. One video is diagnostic; it is not the full 64-environment audit.

The uploader logs the MP4 as `wandb.Video` under `policy/rollout`, so it appears in the run's **Media** tab. Uploading an MP4 only as an artifact would not satisfy this requirement. It then checks the cloud run for training steps, audit status and a video file before reporting success. Return the exact run URL and confirm that the Media tab shows the rollout. A completed upload should still say `simulation_audit_status: failed` if the saved audits failed.

Local outputs are `$RUN/wandb_run.json`, `$RUN/videos/policy-final.mp4`, its `.trace.csv` and `.json`, and SDK logs under `$RUN/wandb/`. Preserve `wandb_run.json`: rerunning the same command reuses the run identity and resumes metric history from the saved cloud/local progress. It can rerender/relog the video and create another artifact version, but should not create another W&B experiment.

If EGL rendering fails, inspect the PC's rendering support and try an available supported backend (for example `MUJOCO_GL=osmesa` when OSMesa is installed). Do not substitute a fake video, disable collision checks, or retrain just to obtain media. Metrics already uploaded remain in the same run; the command reports incomplete and can be retried after fixing rendering. No C++ compiler or ROS installation is required for this upload. Their previously pending runtime checks remain pending.

The old report's maximum angle error near pi does **not** establish that each wheel converged to the opposite target. Failure to enter/complete rotation could also produce a large maximum error. Use the video/trace to investigate. Likewise 0.125 m/s exceeds the 0.10 m/s audit threshold; preserve the actual audit JSON rather than the report's “OK-ish” label.

## Future training

The current retraining revision is motion contract v5 on branch `codex/align-motion-v2`. Follow [AGENT_TRAINING_HANDOFF.md](AGENT_TRAINING_HANDOFF.md) in a separate checkout, preserving the old v2/v3 runs for the upload command above. New audits include phase durations, blocked-gate durations, rotation time and per-wheel errors; these also appear in W&B summary fields.

New alignment runs log online by default to this project. Authenticate once with the training environment's `python -m wandb login`, then use the normal `train` command. Optional `--wandb-entity` and `--wandb-project` select an explicitly requested alternative destination.

- Scalar training/evaluation history uses `train/env_steps` as its horizontal axis. Local `metrics.jsonl` and checkpoints are still written.
- A nominal policy video is recorded approximately every 25 million steps, at the next checkpoint callback, and at the final step. Use `--video-every-steps 0` for final-video-only logging; the final video remains required.
- Media keys are `policy/rollout` for checkpoint videos and `policy/final` for the final policy. Videos and CSV traces are also retained under the run's `videos/` directory.
- A periodic rendering error is reported and retried without discarding training. The final checkpoint is saved before final rendering; failure to produce its video is an explicit error, not a successful media handoff.
- `evaluate.py` updates the same online W&B run with audit summaries and artifacts when `wandb_run.json` is present. Offline SDK sessions cannot resume: their later audits stay local and need a separate upload after synchronization. A high training reward is never labeled as passing the task audit.
- `--no-wandb` is an explicit local-only override. `--wandb-mode offline` records SDK data locally for later `wandb sync`; report it as **not uploaded** until synchronization is verified. Prefer online mode for the requested workflow.

W&B run resumption is logging resumption. The v5 trainer's `--init-from` transfers compatible actor weights and normalization between curriculum stages; it starts a fresh optimizer, critic and step count. It does not resume an interrupted optimizer state.

Future policy entry points should use `training/wandb_logging.py` and provide their own genuine checkpoint rollout through `ExperimentLogger.video(...)`. This standing preference is also recorded in the root `AGENTS.md`; do not apply the alignment robot geometry or 82-input observation format to unrelated policies just to reuse the logger.

The SDK integration follows W&B's [media logging](https://docs.wandb.ai/models/track/log/media), [Video API](https://docs.wandb.ai/models/ref/python/data-types/video), and [explicit run-ID resumption](https://docs.wandb.ai/models/runs/resuming) documentation. Dependencies are locked in `training/wheel_align/uv.lock`.

V4 uses three separate, linked curriculum runs (foundation/single/sequence), each with a final actual checkpoint video. The audit scope is stored in W&B: intermediate single-wheel passes are not full-sequence passes. Refer to AGENT_TRAINING_HANDOFF.md for the staged launch commands.

V5 selects checkpoints by balanced task audits using `training.wheel_align.select_checkpoint`. Selected checkpoint videos have separate Media keys, scopes and steps; `policy/final` remains the final training weights. Follow the current handoff for selected-checkpoint transfer and audit summary fields.
