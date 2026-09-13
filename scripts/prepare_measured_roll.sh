#!/usr/bin/env bash
# Software build and motor-free tests only. Never launches a hardware stack.
set -eo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if pgrep -f '/ros2_control_node([[:space:]]|$)' >/dev/null; then
  echo 'A hardware stack is running. Keep its installation intact; arrange a supported shutdown before preparation.' >&2
  exit 1
fi
source /opt/ros/jazzy/setup.bash
source "$repo_dir/ros2_ws/install-roll/local_setup.bash"
cd "$repo_dir"
python3 scripts/check_triangle_roll.py
build_base="$repo_dir/ros2_ws/build-measured-roll"
install_base="$repo_dir/ros2_ws/install-measured-roll"
export CMAKE_BUILD_PARALLEL_LEVEL=2
cd ros2_ws
colcon build --executor sequential --packages-select neural_controller \
  --build-base "$build_base" --install-base "$install_base" \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_base/neural_controller" --target measured_roll_test triangle_roll_test --parallel 2
cmake --install "$build_base/neural_controller"
source "$install_base/local_setup.bash"
export LD_LIBRARY_PATH="$install_base/neural_controller/lib:${LD_LIBRARY_PATH:-}"
export ROS_DOMAIN_ID=193
export ROS_LOCALHOST_ONLY=1
ctest --test-dir "$build_base/neural_controller" \
  -R '^(measured_roll_core|triangle_roll_core|triangle_roll_lifecycle|triangle_roll_stand_lifecycle)$' --output-on-failure
cd "$repo_dir"
"$build_base/neural_controller/measured_roll_test" hardware_testing/triangle_roll/measured-parity.csv
python3 scripts/check_triangle_roll.py --install-base ros2_ws/install-roll --controller-install-base ros2_ws/install-measured-roll
python3 -m pytest hardware_testing/triangle_roll/test_roll_reference.py -q
echo 'Software checks passed. No motor stack was started. Follow TRIANGLE_ROLL_LAB.md for the supervised test.'
