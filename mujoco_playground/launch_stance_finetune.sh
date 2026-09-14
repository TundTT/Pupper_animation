#!/usr/bin/env bash
# Fine-tune the wheel policy at the reduced 0.65 rad abduction splay.
# Warm-started from backpack_2026-09-13, the policy currently deployed on the robot.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export PYTHONPATH="$HERE"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

BOARD=/home/theerawit/Pupper_animation/agent_coordination/board.py

exec python3 "$BOARD" --agent wheel run --gpu 1 -- \
  "$HERE/.venv/bin/python" -m workspace.train \
    --num_timesteps 100000000 \
    --learning_rate 5e-5 \
    --init_params "$HERE/../trained_policies/backpack_2026-09-13/mjx_params" \
    --use_wandb \
    --wandb_project pupper-wheel \
    --wandb_entity QuadMorph
