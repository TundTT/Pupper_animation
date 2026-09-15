#!/usr/bin/env bash
# CPU simulation only; no hardware and no ROS.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
 echo 'Usage: run_native_align_diagnostic.sh MODEL_XML PYTHON_EXECUTABLE [OUTPUT_DIR]. Requires native MuJoCo3.6.0/numpy and a C++17 compiler (CXX). Runs four actual-actor nominal diagnostics; no hardware/GPU.'
 exit 0
fi
[[ $# == 2 || $# == 3 ]] || { echo 'Use --help' >&2; exit 2; }
model="$(realpath "$1")";python="$(realpath "$2")"
repo="$(cd "$(dirname "$0")/../.." && pwd)";cd "$repo"
: "${CXX:=c++}"
out=${3:-hardware_testing/notebook_lift/native_align_diagnostic}
"$CXX" -std=c++17 -O2 -fPIC -shared -DRTNEURAL_DEFAULT_ALIGNMENT=16 -DRTNEURAL_USE_EIGEN=1 \
 -Iros2_ws/src/neural_controller/include \
 -Iros2_ws/src/neural_controller/modules/RTNeural \
 -Iros2_ws/src/neural_controller/modules/RTNeural/modules/Eigen \
 hardware_testing/notebook_lift/native_align_bridge.cpp -o hardware_testing/notebook_lift/native_align_bridge.so
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "$python" \
 scripts/notebook_lift/native_align_diagnostic.py --model "$model" \
 --policy ros2_ws/src/neural_controller/launch/policy_notebook_lift.json \
 --library hardware_testing/notebook_lift/native_align_bridge.so --out "$out"
