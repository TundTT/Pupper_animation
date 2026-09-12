#!/usr/bin/env bash
# Build and software checks only. Does not start the robot or capture calibration.
set -eo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
cd "$repo_dir"
if pgrep -f '/ros2_control_node([[:space:]]|$)' >/dev/null; then
  printf '%s\n' 'Controller manager is running. Resolve the live session before rebuilding.' >&2
  exit 1
fi
python3 scripts/check_locomotion_policies.py
cd ros2_ws
# Override for an isolated local build; defaults to the selected checkout on Pi.
build_base="${LOCOMOTION_BUILD_BASE:-$repo_dir/ros2_ws/build}"
install_base="${LOCOMOTION_INSTALL_BASE:-$repo_dir/ros2_ws/install}"
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --executor sequential \
  --build-base "$build_base" --install-base "$install_base" \
  --packages-select robot_calibration control_board_hardware_interface neural_controller \
  joy_utils pupper_v3_description cmd_vel_mux animation_controller_py \
  --cmake-force-configure --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
source "$install_base/local_setup.bash"
export LD_LIBRARY_PATH="$install_base/neural_controller/lib:${LD_LIBRARY_PATH:-}"
ctest --test-dir "$build_base/robot_calibration" --output-on-failure
ctest --test-dir "$build_base/joy_utils" -R 'alignment_joystick|locomotion_joystick' --output-on-failure
ctest --test-dir "$build_base/neural_controller" \
  -R 'startup_launch|locomotion_manager|walk_policy_contract|wheel_policy_contract|hybrid_controller_lifecycle' --output-on-failure
cd "$repo_dir"
python3 scripts/check_locomotion_policies.py --package-share "$install_base/neural_controller/share/neural_controller"
printf '%s\n' 'Software checks passed. Follow LOCOMOTION_LAB.md before any hardware startup.'
