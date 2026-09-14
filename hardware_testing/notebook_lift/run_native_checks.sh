#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo"
: "${CXX:=c++}"
out=hardware_testing/notebook_lift
for test in walking_frame keyframe_position align_motion notebook_lift_observation notebook_lift_targets; do
 "$CXX" -std=c++17 -O2 -Iros2_ws/src/neural_controller/include "ros2_ws/src/neural_controller/test/${test}_test.cpp" -o "$out/${test}_test_binary"
done
"$out/walking_frame_test_binary" > "$out/regressions.log"
"$out/keyframe_position_test_binary" >> "$out/regressions.log"
"$out/align_motion_test_binary" >> "$out/regressions.log"
"$out/notebook_lift_observation_test_binary" "$out/observation_reference.csv" > "$out/observation_test.log"
"$out/notebook_lift_targets_test_binary" "$out/target_reference.csv" > "$out/targets_test.log"
for backend in stl eigen; do
 flags=()
 [[ "$backend" == eigen ]] && flags+=(-DRTNEURAL_USE_EIGEN=1)
 "$CXX" -std=c++17 -O2 -DRTNEURAL_DEFAULT_ALIGNMENT=16 "${flags[@]}" \
  -Iros2_ws/src/neural_controller/include \
  -Iros2_ws/src/neural_controller/modules/RTNeural \
  -Iros2_ws/src/neural_controller/modules/RTNeural/modules/Eigen \
  ros2_ws/src/neural_controller/test/notebook_lift_core_test.cpp -o "$out/core_${backend}_test_binary"
 "$out/core_${backend}_test_binary" ros2_ws/src/neural_controller/launch/policy_notebook_lift.json \
  "$out/network_reference.csv" "$out/geometry_reference.csv" > "$out/core_${backend}_test.log"
done
