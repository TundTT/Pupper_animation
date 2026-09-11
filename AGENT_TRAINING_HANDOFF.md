# PC agent handoff: alignment v4 curriculum

Use the latest `origin/codex/align-motion-v2`. The branch name is unchanged; the policy/runtime contract is now **v4**. Read [ALIGN_MOTION_V4.md](ALIGN_MOTION_V4.md) for the diagnosis, motion changes, hardware sources and validation limits. Prior v2/v3 weights are incompatible. Heating stays manual.

The user will send this handoff to the PC agent to start training on the NVIDIA PC. Do not train on the laptop, deploy to the robot, change the selected robot policy, merge this branch into robot-code, or overwrite other agents' work. Ordinary setup/checks and the staged training below are within that handoff. Do not start another process if this job is already running. Use Linux/WSL2, preferably a Linux-filesystem checkout, and a persistent terminal such as tmux. Do not change drivers or dependency versions to work around failures.

## Preserve existing runs and prepare a new checkout

The original v2 checkout was `/home/theerawit/Pupper_animation-align-motion-v2`; the latest v3 run was `align-motion-v3-seed0-20260910T211747Z`, commit `8be7fd3`, W&B `17d412eac8004407`. Preserve both original training checkouts for faithful replay. Inspect existing paths/jobs before running the example:

```bash
ORIGINAL=/home/theerawit/Pupper_animation-align-motion-v2
NEXT=/home/theerawit/Pupper_animation-align-motion-v4
git -C "$ORIGINAL" status --short
git -C "$ORIGINAL" fetch origin
# If NEXT exists, inspect and reuse an appropriate clean checkout; do not overwrite it.
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

Use existing W&B credentials or local CLI login; never put an API key in chat, source or logs. Default destination is entity `QuadMorph`, project `wheel-leg lift and align triangle base`. Online metrics, artifacts and actual checkpoint videos in Media are enabled by default. Each curriculum stage is a separate W&B run with parent-checkpoint provenance.

## Preflight

```bash
"$PY" -m training.wheel_align.generate_geometry --check
"$PY" -m training.wheel_align.generate_reference --check
"$PY" -m training.wheel_align.preflight --jit-step
"$PY" -m training.wheel_align.native_audit
"$PY" -m training.wheel_align.native_audit --interrupt
mkdir -p runs/checks
if command -v c++ >/dev/null; then
  c++ -std=c++17 -O2 -Iros2_ws/src/neural_controller/include \
    -Iros2_ws/src/joy_utils/include ros2_ws/src/neural_controller/test/align_motion_test.cpp \
    -o runs/checks/align_motion_test
  export ALIGN_MOTION_TEST_EXE="$PWD/runs/checks/align_motion_test"
  "$ALIGN_MOTION_TEST_EXE"
else
  unset ALIGN_MOTION_TEST_EXE ALIGN_EXPORT_TEST_EXE
  printf '%s\n' 'PC C++ verification pending; retain the laptop validation report.'
fi
"$PY" -m pytest training/wheel_align/tests -q
```

The native audit has no actor/learning: it exercises the reference, actual rigid-body physics, half-turn servos and interruption recovery. Require all four completed, wheel gap >10 mm, body gap >5 mm and peak approach <0.10 m/s. Do not weaken thresholds. The previous PC lacked a compiler/ROS; missing tools may be reported as pending on that PC because the source is checked on the laptop, but a present compiler/test failing is a real failure to resolve. A trained export still needs RTNeural validation before deployment. Optional Warp messages alone do not mean the selected MJX JAX backend failed.

## Train and audit each stage before advancing

Approximately 50 million environment steps total: foundation 5M, single 10M, sequence 35M. The foundation emphasizes unload/lift/hold/lower with small hub-angle errors and nominal dynamics. Single trains full-angle individual wheels with moderate randomization. Sequence uses the original wider dynamics randomization, random leg orders, and interruption/retry cases. Single-wheel evaluation balances the requested leg across its 64 environments.

Run the following from the repository root in the persistent session. The stage audit is required to pass before transferring weights to the next stage. If a stage fails, preserve it and report the per-wheel failures/video; do not silently launch extra runs, increase budgets, or bypass the check.

```bash
set -euo pipefail
PY="$PWD/training/wheel_align/.venv/bin/python"
CHAIN="align-motion-v4-seed0-$(date -u +%Y%m%dT%H%M%SZ)"
F="runs/${CHAIN}-foundation"
S="runs/${CHAIN}-single"
Q="runs/${CHAIN}-sequence"
mkdir -p runs
printf '%s\n' "$F" "$S" "$Q" > runs/align-motion-v4-active-chain.txt

"$PY" -u -m training.wheel_align.train --stage foundation --steps 5000000 \
  --envs 2048 --seed 0 --out "$F" 2>&1 | tee "$F.console.log"
"$PY" -m training.wheel_align.evaluate --stage foundation --nominal \
  --params "$F/mjx_params" --out "$F/audit-foundation.json"
"$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passes_simulation_gate"], "Foundation audit failed; stop and report"' "$F/audit-foundation.json"

"$PY" -u -m training.wheel_align.train --stage single --steps 10000000 \
  --init-from "$F/mjx_params" --envs 2048 --seed 0 --out "$S" 2>&1 | tee "$S.console.log"
"$PY" -m training.wheel_align.evaluate --stage single \
  --params "$S/mjx_params" --out "$S/audit-single.json"
"$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passes_simulation_gate"], "Single-wheel audit failed; stop and report"' "$S/audit-single.json"

"$PY" -u -m training.wheel_align.train --stage sequence --steps 35000000 \
  --init-from "$S/mjx_params" --envs 2048 --seed 0 --out "$Q" 2>&1 | tee "$Q.console.log"
"$PY" -m training.wheel_align.evaluate --params "$Q/mjx_params" \
  --nominal --out "$Q/audit-nominal.json"
"$PY" -m training.wheel_align.evaluate --params "$Q/mjx_params" \
  --out "$Q/audit-randomized.json"
"$PY" -m training.wheel_align.evaluate --params "$Q/mjx_params" \
  --interrupt --seed 20260911 --out "$Q/audit-interrupted.json"
```

Do not create F/S/Q directories beforehand: the trainer refuses an existing output directory. `--init-from` transfers the actor and observation normalizer after exact v4 contract/source checks; the optimizer, critic and step count start fresh. This is **not** optimizer resumption and cannot load the failed v3 checkpoint. Each stage's budget is additional to its parent's budget. Do not edit/pull/regenerate sources in this checkout after the first stage starts, including between stages. Source hashes must match throughout the chain. Keep further experiments in another checkout.

Check a live process, GPU use, finite metrics and increasing positive training steps. A params_0 file is not proof of learning. Report the session/reattachment command, source commit, absolute run/log paths, latest positive step and checkpoint. If the process continues beyond the agent turn, leave the session running; do not claim training finished or promise monitoring without a real mechanism. In case of GPU OOM, preserve the failed directory and retry the affected stage in a fresh directory with 1024 or 512 environments, retaining the same valid parent. Supported counts: 256/512/1024/2048/4096.

## Acceptance, export and report

All three final audits must pass. For the full sequence, `requested_completed` equals the all-four completion count. For intermediate stages it measures only the selected wheel; their passing status is explicitly scoped to that stage, not a full-sequence approval. Inspect `per_wheel.attempted/completed`, timeout counts, actual support load, blocked gate time, phase durations, peak impact by phase, conservative spacing and final angles/speeds. Peak impact remains <0.10 m/s, conservative wheel gap >5 mm and body gap >0. Completed and aligned/settled counts must both equal 64. Do not interpret an accumulated episode metric as a success count.

The command scheduler advances after completion, pauses in stand between wheels, and records a failure after a 48-second per-wheel timeout. A deliberate interruption gets 20 extra seconds for lowering/retry. No timeout is labeled success. Full audits/videos have a 320-second maximum and stop once all environments finish plus a settling pause, or terminate. Runtime joystick requests remain immediate interruptions followed by supported descent; the auto scheduler is for simulation/training only.

```bash
"$PY" -m training.wheel_align.export --params "$Q/mjx_params" \
  --out "$Q/policy_wheel_align_motion_v4.json"
```

Export writes the matching `.reference.csv`; do not overwrite an existing export. If exporting a failed audit for inspection, label it failed. Where ROS Jazzy is available, build neural_controller/joy_utils with BUILD_TESTING=ON, run the five targeted CTest cases described in ALIGN_MOTION_V2.md, and run `align_export_test` against the **trained v4 export and CSV**. If unavailable, return the files and mark runtime verification pending. A unit-test fixture is not the trained export.

Return all stage W&B URLs and confirm real videos in Media, source/dependency/GPU details, checkpoint hashes, budgets/actual steps, elapsed time, audit JSONs, final export/CSV and any pending verification. Artifacts live in gitignored run directories; pushing source does not transfer them. No robot startup/deployment is authorized here. The current robot-code calibration work is separate and must be integrated/reviewed before any future physical test; do not replace it with this branch's older startup implementation.
