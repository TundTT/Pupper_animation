#!/usr/bin/env bash
# Software only. No launch, motor commands or calibration capture.
set -eo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
if pgrep -f '/ros2_control_node([[:space:]]|$)' >/dev/null; then
  printf '%s\n' 'Controller manager is running. Do not replace its libraries.' >&2
  exit 1
fi
cd "$repo_dir"
python3 scripts/check_inverted_triangle.py
build_base="${TRIANGLE_BUILD_BASE:-$repo_dir/ros2_ws/build}"
install_base="${TRIANGLE_INSTALL_BASE:-$repo_dir/ros2_ws/install}"
cd ros2_ws
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --executor sequential \
  --build-base "$build_base" --install-base "$install_base" \
  --packages-select robot_calibration control_board_hardware_interface neural_controller joy_utils pupper_v3_description animation_controller_py \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
source "$install_base/local_setup.bash"
export LD_LIBRARY_PATH="$install_base/neural_controller/lib:${LD_LIBRARY_PATH:-}"
ctest --test-dir "$build_base/robot_calibration" --output-on-failure
ctest --test-dir "$build_base/neural_controller" -R 'inverted_triangle|startup_launch|keyframe_controller|keyframe_position|walk_policy_contract|wheel_policy_contract|hybrid_controller_lifecycle' --output-on-failure
ctest --test-dir "$build_base/joy_utils" -R alignment_joystick --output-on-failure
cd "$repo_dir"
python3 scripts/check_inverted_triangle.py --install-base "$install_base"
python3 -m pytest hardware_testing/inverted_triangle/test_reference.py -q
printf '%s\n' 'Software preparation passed. Follow INVERTED_TRIANGLE_LAB.md; hardware has not been started.'
