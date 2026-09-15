#!/usr/bin/env bash
# No ROS or robot connection required. CXX may name a compiler wrapper.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
controller_dir="$repo_dir/ros2_ws/src/neural_controller"
rt_dir="$controller_dir/modules/RTNeural"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/leg-to-wheel-policy-test.XXXXXX")"
trap 'rm -rf "$test_dir"' EXIT
"${CXX:-c++}" -std=c++17 -O2 -DRTNEURAL_USE_EIGEN=1 -DRTNEURAL_DEFAULT_ALIGNMENT=16 \
  -I"$controller_dir/include" -I"$rt_dir" -I"$rt_dir/modules/Eigen" \
  -I"$rt_dir/modules/json" "$controller_dir/test/leg_to_wheel_policy_test.cpp" \
  -o "$test_dir/leg_to_wheel_policy_test"
"$test_dir/leg_to_wheel_policy_test" "$controller_dir/launch/policy_leg_to_wheel.json" \
  "$controller_dir/test/leg_to_wheel_policy_reference.csv" \
  "$repo_dir/policies/leg_to_wheel/evidence/runtime_reference.csv"
