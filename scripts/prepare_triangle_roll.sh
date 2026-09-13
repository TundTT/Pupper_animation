#!/usr/bin/env bash
# Build/check an isolated overlay only. No hardware launch, calibration or motion.
set -eo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
cd "$repo_dir"
python3 scripts/check_triangle_roll.py
build_base="${ROLL_BUILD_BASE:-$repo_dir/ros2_ws/build-roll}"
install_base="${ROLL_INSTALL_BASE:-$repo_dir/ros2_ws/install-roll}"
# Never overwrite libraries used by a live stack, including a previous roll run.
if pgrep -f '/ros2_control_node([[:space:]]|$)' >/dev/null; then
  printf '%s\n' 'A hardware stack is running. Preserve its overlay; arrange a supported shutdown before building on the robot.' >&2
  exit 1
fi
cd ros2_ws
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --executor sequential \
  --build-base "$build_base" --install-base "$install_base" \
  --packages-select robot_calibration control_board_hardware_interface neural_controller joy_utils pupper_v3_description animation_controller_py \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
source "$install_base/local_setup.bash"
export LD_LIBRARY_PATH="$install_base/neural_controller/lib:${LD_LIBRARY_PATH:-}"
ctest --test-dir "$build_base/robot_calibration" --output-on-failure
ctest --test-dir "$build_base/neural_controller" -R 'triangle_roll|inverted_triangle|startup_launch|walk_policy_contract|wheel_policy_contract' --output-on-failure
ctest --test-dir "$build_base/joy_utils" -R alignment_joystick --output-on-failure
cd "$repo_dir"
python3 scripts/check_triangle_roll.py --install-base "$install_base"
python3 -m pytest hardware_testing/triangle_roll/test_roll_reference.py -q
printf '%s\n' 'Software preparation complete. Follow TRIANGLE_ROLL_LAB.md. Hardware remains inactive.'
