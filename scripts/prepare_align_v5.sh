#!/usr/bin/env bash
# Build and validate this checkout. Does not start hardware or modify calibration.
set -eo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
cd "$repo_dir"
if pgrep -f '/ros2_control_node([[:space:]]|$)' >/dev/null; then
  printf '%s\n' 'A hardware controller manager is running. Stop it through the established procedure before rebuilding.' >&2
  exit 1
fi
python3 scripts/check_align_v5.py
cd ros2_ws
# Sequential compilation avoids exhausting the Pi's memory on RTNeural templates.
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --executor sequential --packages-select robot_calibration \
  control_board_hardware_interface neural_controller joy_utils pupper_v3_description animation_controller_py \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
source install/local_setup.bash
ctest --test-dir build/robot_calibration --output-on-failure
ctest --test-dir build/joy_utils -R alignment_joystick --output-on-failure
ctest --test-dir build/neural_controller \
  -R 'startup_launch|hybrid_controller_lifecycle|hybrid_policy_contract|walk_policy_contract|align_v5' \
  --output-on-failure
cd "$repo_dir"
python3 scripts/check_align_v5.py --installed
printf '%s\n' 'Software checks passed. Follow ALIGN_V5_LAB.md for physical setup and startup calibration.'
