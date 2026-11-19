#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL=egl
PY="$PWD/training/wheel_align/.venv/bin/python"
CHAIN="align-motion-v4-seed0-$(date -u +%Y%m%dT%H%M%SZ)"
F="runs/${CHAIN}-foundation"
S="runs/${CHAIN}-single"
Q="runs/${CHAIN}-sequence"
mkdir -p runs
printf '%s\n' "$F" "$S" "$Q" > runs/align-motion-v4-active-chain.txt

echo "=== STAGE: foundation ==="
"$PY" -u -m training.wheel_align.train --stage foundation --steps 5000000 \
  --envs 2048 --seed 0 --out "$F" 2>&1 | tee "$F.console.log"
train_exit=${PIPESTATUS[0]}
printf '%s\n' "$train_exit" > "$F.exit-code.txt"
"$PY" -m training.wheel_align.evaluate --stage foundation --nominal \
  --params "$F/mjx_params" --out "$F/audit-foundation.json"
"$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passes_simulation_gate"], "Foundation audit failed; stop and report"' "$F/audit-foundation.json"
echo "=== STAGE foundation PASSED ==="

echo "=== STAGE: single ==="
"$PY" -u -m training.wheel_align.train --stage single --steps 10000000 \
  --init-from "$F/mjx_params" --envs 2048 --seed 0 --out "$S" 2>&1 | tee "$S.console.log"
train_exit=${PIPESTATUS[0]}
printf '%s\n' "$train_exit" > "$S.exit-code.txt"
"$PY" -m training.wheel_align.evaluate --stage single \
  --params "$S/mjx_params" --out "$S/audit-single.json"
"$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passes_simulation_gate"], "Single-wheel audit failed; stop and report"' "$S/audit-single.json"
echo "=== STAGE single PASSED ==="

echo "=== STAGE: sequence ==="
"$PY" -u -m training.wheel_align.train --stage sequence --steps 35000000 \
  --init-from "$S/mjx_params" --envs 2048 --seed 0 --out "$Q" 2>&1 | tee "$Q.console.log"
train_exit=${PIPESTATUS[0]}
printf '%s\n' "$train_exit" > "$Q.exit-code.txt"
"$PY" -m training.wheel_align.evaluate --params "$Q/mjx_params" \
  --nominal --out "$Q/audit-nominal.json"
"$PY" -m training.wheel_align.evaluate --params "$Q/mjx_params" \
  --out "$Q/audit-randomized.json"
"$PY" -m training.wheel_align.evaluate --params "$Q/mjx_params" \
  --interrupt --seed 20260911 --out "$Q/audit-interrupted.json"
echo "=== STAGE sequence COMPLETE (audits above; check passes_simulation_gate) ==="
echo "ALL_STAGES_DONE"
