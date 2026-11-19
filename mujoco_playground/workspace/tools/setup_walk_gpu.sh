#!/usr/bin/env bash
# Run on the remote Linux CUDA host, from anywhere inside this checkout.
set -euo pipefail
cd "$(dirname "$0")/../.."
uv venv --python 3.12 .venv-walk
uv pip install --python .venv-walk/bin/python -r workspace/requirements-walk-cuda.txt
uv pip check --python .venv-walk/bin/python
nvidia-smi
.venv-walk/bin/python -c 'import jax; print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices()), "JAX CUDA is not available"'
.venv-walk/bin/python -m workspace.check_walk
printf '%s\n' 'Ready. Use .venv-walk/bin/python directly; do not use uv run or sync the wheel environment.'
