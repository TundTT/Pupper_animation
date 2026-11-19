#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL=egl
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

echo "=== STAGE: foundation (train) ==="
"$PY" -u -m training.wheel_align.train --stage foundation --steps 5000000 \
  --envs 2048 --seed 0 --out "$F" 2>&1 | tee "$F.console.log"
printf '%s\n' "${PIPESTATUS[0]}" > "$F.exit-code.txt"
echo "=== STAGE: foundation (select) ==="
"$PY" -u -m training.wheel_align.select_checkpoint --run-dir "$F" 2>&1 | tee "$F.selection.log"
echo "=== STAGE foundation SELECTED: $(selected "$F") ==="

echo "=== STAGE: single (train) ==="
"$PY" -u -m training.wheel_align.train --stage single --steps 10000000 \
  --init-from "$(selected "$F")" --envs 2048 --seed 0 --out "$S" 2>&1 | tee "$S.console.log"
printf '%s\n' "${PIPESTATUS[0]}" > "$S.exit-code.txt"
echo "=== STAGE: single (select) ==="
"$PY" -u -m training.wheel_align.select_checkpoint --run-dir "$S" 2>&1 | tee "$S.selection.log"
echo "=== STAGE single SELECTED: $(selected "$S") ==="

echo "=== STAGE: sequence (train) ==="
"$PY" -u -m training.wheel_align.train --stage sequence --steps 35000000 \
  --init-from "$(selected "$S")" --envs 2048 --seed 0 --out "$Q" 2>&1 | tee "$Q.console.log"
printf '%s\n' "${PIPESTATUS[0]}" > "$Q.exit-code.txt"
echo "=== STAGE: sequence (select) ==="
"$PY" -u -m training.wheel_align.select_checkpoint --run-dir "$Q" 2>&1 | tee "$Q.selection.log"
echo "=== STAGE sequence SELECTED: $(selected "$Q") ==="

echo "=== EXPORT ==="
"$PY" -m training.wheel_align.export --params "$(selected "$Q")" \
  --out "$Q/selected/policy_wheel_align_motion_v5.json"

echo "ALL_STAGES_DONE"
