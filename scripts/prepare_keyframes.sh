#!/usr/bin/env bash
# Software preparation only. Never launches hardware or changes calibration.
set -eo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
cd "$repo_dir"
if pgrep -f '/ros2_control_node([[:space:]]|$)' >/dev/null; then
  printf '%s\n' 'A controller manager is running. Stop through the established procedure before rebuilding.' >&2
  exit 1
fi
python3 scripts/check_keyframes.py
cd ros2_ws
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --executor sequential --packages-select robot_calibration \
  control_board_hardware_interface neural_controller joy_utils pupper_v3_description animation_controller_py \
  --cmake-force-configure --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
source install/local_setup.bash
# Test the installed plugin, not a build-tree RPATH fallback.
export LD_LIBRARY_PATH="$repo_dir/ros2_ws/install/neural_controller/lib:${LD_LIBRARY_PATH:-}"
ctest --test-dir build/robot_calibration --output-on-failure
ctest --test-dir build/joy_utils -R alignment_joystick --output-on-failure
ctest --test-dir build/neural_controller \
  -R 'keyframe_controller|keyframe_manager|startup_launch|hybrid_controller_lifecycle|hybrid_policy_contract|walk_policy_contract|align_v5' --output-on-failure
cd "$repo_dir"
python3 scripts/check_keyframes.py --install-base "$repo_dir/ros2_ws/install"
printf '%s\n' 'Software checks complete. Read KEYFRAME_LAB.md for pending target and physical checks.'
