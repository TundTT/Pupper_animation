#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:?Pass an output directory}
mkdir -p "$out"
g++ -std=c++17 -O2 "$root/tools/current_pose_hold/test.cpp" -o "$out/hold_test"
"$out/hold_test"
g++ -std=c++17 -O2 -pthread -Dlinux \
  -I"$root/ros2_ws/src/control_board_hardware_interface/include" \
  "$root/tools/current_pose_hold/transport_test.cpp" -o "$out/transport_test"
"$out/transport_test"
g++ -std=c++17 -O2 -pthread -Dlinux \
  -I"$root/ros2_ws/src/control_board_hardware_interface/include" \
  "$root/tools/current_pose_hold/main.cpp" \
  "$root/ros2_ws/src/control_board_hardware_interface/src/rt/rt_spi.cpp" \
  -o "$out/current_pose_hold"
