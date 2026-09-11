# PC agent handoff: alignment v5

Use the latest `origin/codex/align-motion-v2`. The branch name is unchanged; the motion contract is now **v5, 83 observations / 8 actions**. Read [ALIGN_MOTION_V5.md](ALIGN_MOTION_V5.md). Start a fresh foundation run: v2/v3/v4 weights are incompatible with v5. Preserve the failed v4 run `align-motion-v4-seed0-20260911T005825Z-foundation`, W&B `5c2f3adfb8704f65`, source `0b55cf1`, and its original checkout for replay.

The user will send this handoff to start training on the NVIDIA PC. No laptop training, robot deployment, controller selection, heater automation, merge into robot-code, or changes to another agent's calibration work are authorized. Inspect running jobs first; do not duplicate an existing job. Use Linux/WSL2, a Linux-filesystem checkout and a persistent tmux session. Preserve existing dependencies/drivers and use the locked environment.

## Prepare a separate checkout

Adapt the existing path after inspection; do not overwrite an existing destination:

```bash
ORIGINAL=/home/theerawit/Pupper_animation-align-motion-v4
NEXT=/home/theerawit/Pupper_animation-align-motion-v5
git -C "$ORIGINAL" status --short
git -C "$ORIGINAL" fetch origin
git -C "$ORIGINAL" worktree add --detach "$NEXT" origin/codex/align-motion-v2
cd "$NEXT"
git log -1 --oneline
git lfs install
git lfs pull
uv sync --project training/wheel_align --extra cuda --frozen
PY="$PWD/training/wheel_align/.venv/bin/python"
"$PY" -m wandb login
nvidia-smi
"$PY" -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices())"
```

Use existing W&B credentials or local CLI login. Never put keys in chat, logs, commands or source. Destination: entity `QuadMorph`, project `wheel-leg lift and align triangle base` (literal spaces). Online logging and checkpoint videos in Media are the default.

## Preflight

```bash
"$PY" -m training.wheel_align.generate_geometry --check
"$PY" -m training.wheel_align.generate_reference --check
"$PY" -m training.wheel_align.preflight --jit-step
mkdir -p runs/checks
if command -v c++ >/dev/null; then
  c++ -std=c++17 -O2 -Iros2_ws/src/neural_controller/include \
    -Iros2_ws/src/joy_utils/include ros2_ws/src/neural_controller/test/align_motion_test.cpp \
    -o runs/checks/align_motion_test
  export ALIGN_MOTION_TEST_EXE="$PWD/runs/checks/align_motion_test"
  "$ALIGN_MOTION_TEST_EXE"
else
  unset ALIGN_MOTION_TEST_EXE ALIGN_EXPORT_TEST_EXE
  printf '%s\n' 'PC C++ verification pending; consult laptop source validation.'
fi
"$PY" -m pytest training/wheel_align/tests -q
```

The tests include native MuJoCo full motions and interruption recovery with no actor/optimizer. Do not weaken gates or ignore a present compiler/test failing. If this PC lacks ROS/C++, report its checks pending; a trained export still needs RTNeural verification before future deployment. Optional Warp import notices alone do not indicate failure of the selected JAX MJX backend.

## Train, select and transfer

Keep this checkout frozen throughout the chain: do not edit, pull or regenerate sources between stages. Total requested training remains 5M foundation + 10M single + 35M sequence steps. PPO may round upward to complete batches. Selection audits add evaluation compute, not training steps. All internal evaluations now use 64 environments.

Run from the repository root in tmux:

```bash
set -euo pipefail
PY="$PWD/training/wheel_align/.venv/bin/python"
CHAIN="align-motion-v5-seed0-$(date -u +%Y%m%dT%H%M%SZ)"
F="runs/${CHAIN}-foundation"
S="runs/${CHAIN}-single"
Q="runs/${CHAIN}-sequence"
mkdir -p runs
printf '%s\n' "$F" "$S" "$Q" > runs/align-motion-v5-active-chain.txt
selected() {
  "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1]+"/selected_checkpoint.json")); assert r["status"]=="passed"; print(r["params"])' "$1"
}

"$PY" -u -m training.wheel_align.train --stage foundation --steps 5000000 \
  --envs 2048 --seed 0 --out "$F" 2>&1 | tee "$F.console.log"
"$PY" -u -m training.wheel_align.select_checkpoint --run-dir "$F" 2>&1 | tee "$F.selection.log"

"$PY" -u -m training.wheel_align.train --stage single --steps 10000000 \
  --init-from "$(selected "$F")" --envs 2048 --seed 0 --out "$S" 2>&1 | tee "$S.console.log"
"$PY" -u -m training.wheel_align.select_checkpoint --run-dir "$S" 2>&1 | tee "$S.selection.log"

"$PY" -u -m training.wheel_align.train --stage sequence --steps 35000000 \
  --init-from "$(selected "$S")" --envs 2048 --seed 0 --out "$Q" 2>&1 | tee "$Q.console.log"
"$PY" -u -m training.wheel_align.select_checkpoint --run-dir "$Q" 2>&1 | tee "$Q.selection.log"

"$PY" -m training.wheel_align.export --params "$(selected "$Q")" \
  --out "$Q/selected/policy_wheel_align_motion_v5.json"
```

Do not create F/S/Q beforehand: training refuses an existing directory. Transfer copies actor weights and observation normalization; optimizer, critic and environment-step count start fresh. This is not optimizer resumption. Only `selected/mjx_params` should be transferred or exported. Root `mjx_params` remains the last checkpoint and may fail audits.

Selection tests positive-step checkpoints newest-first; it does not use reward ranking or params_0. Foundation and single require balanced 64-case audits on two seeds. Sequence requires nominal, randomized and interrupted 64-case audits. A candidate stops at its first failed audit, preserving that result, then the selector tests an earlier checkpoint. If none pass, selection exits nonzero and the chain stops. Do not bypass this, increase the training budget, or silently start another experiment. Report `selection/report.json` and failures. Cached audits are reused only with matching checkpoint hash, scope, seed and environment count.

## Media, acceptance and report

Training logs a stage-scope nominal video and, for intermediate stages, a separately labeled full-angle sequence diagnostic. A foundation video represents one front-left case, not all 64 audit cases. Selection also records videos of the actual selected checkpoint under `policy/selected_<scope>`. These can differ from `policy/final`, which is the final training checkpoint. Verify both scope and checkpoint step in the Media tab.

W&B summaries `selected_checkpoint_simulation_status`, `selected_checkpoint_stage`, `selected_checkpoint_step`, `selected_checkpoint_sha256` and `selection_handoff_complete` describe selection. Existing final-checkpoint audit failures remain visible. Intermediate passes are scoped to their stage. Selected checkpoint, selection report/audits, video traces and provenance are uploaded as artifacts. Checkpoints selected earlier in training retain their original step on the video axis.

Acceptance retains the existing 10 mm rotation gate and physical audit thresholds: every requested wheel completes, all selected angles/speeds settle below 0.08 rad / rad/s, no timeout or unsafe rotation, wheel gap >5 mm, body gap >0, peak approach <0.10 m/s. Audits now include the complete post-timeout lowering/settling or report that it did not finish. Do not interpret zero unsafe rotations as successful alignment; the gate can block a stalled wheel.

Return source commit, all W&B URLs, current session/reattachment command, absolute run/log paths, positive training steps, selected steps/checkpoint hashes, selection reports and export/CSV. Verify the actual selected export with ROS Jazzy `align_export_test` and the five targeted CTests when available; otherwise return it with trained-runtime verification pending. Never claim the untrained laptop fixture is the trained export. No deployment is authorized.

If GPU OOM occurs, preserve the failed directory and retry the affected stage in a fresh directory with 1024 or 512 environments and the same valid selected parent. Supported counts are 256/512/1024/2048/4096. A params_0 file alone is not evidence of learning. If a process continues beyond the agent turn, leave tmux running and report real status without claiming completion or promising unscheduled monitoring.
