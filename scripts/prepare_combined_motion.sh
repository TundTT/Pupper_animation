#!/usr/bin/env bash
# Software only. Separate installation; never launch or restart hardware.
set -eo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install_base="$repo_dir/ros2_ws/install-combined"
if grep -l -F "$install_base/" /proc/[0-9]*/maps 2>/dev/null | head -1 | grep -q .; then
  echo 'The combined installation is mapped by a running process; do not replace it.' >&2
  exit 1
fi
source /opt/ros/jazzy/setup.bash
source "$repo_dir/ros2_ws/install-roll/local_setup.bash"
source "$repo_dir/ros2_ws/install-upper-home/local_setup.bash"
cd "$repo_dir"
python3 scripts/check_locomotion_policies.py
python3 scripts/check_notebook_lift.py
python3 -m unittest discover -s ros2_ws/src/neural_controller/test -p test_motion_buttons.py -v
export CMAKE_BUILD_PARALLEL_LEVEL=2
cd ros2_ws
colcon build --executor sequential --packages-select cmd_vel_mux \
  --build-base "$repo_dir/ros2_ws/build-combined" --install-base "$install_base" \
  --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
colcon build --executor sequential --packages-select neural_controller \
  --build-base "$repo_dir/ros2_ws/build-combined" --install-base "$install_base" \
  --cmake-target neural_controller \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DQUADMORPH_FOCUSED_TESTS=ON
cmake --build "$repo_dir/ros2_ws/build-combined/neural_controller" --parallel 2 \
  --target joint_pose_controller hub_roll_controller notebook_lift_controller \
  notebook_lift_core_test notebook_lift_align_test notebook_lift_controller_test \
  notebook_lift_observation_test notebook_lift_targets_test walking_frame_test walking_handoff_test \
  walk_policy_test wheel_policy_test measured_roll_test triangle_roll_controller_test
cmake --install "$repo_dir/ros2_ws/build-combined/neural_controller"
source "$install_base/local_setup.bash"
export LD_LIBRARY_PATH="$install_base/neural_controller/lib:${LD_LIBRARY_PATH:-}"
export ROS_DOMAIN_ID=193 ROS_LOCALHOST_ONLY=1
ctest --test-dir "$repo_dir/ros2_ws/build-combined/neural_controller" \
  -R '^(walking_frame|walking_handoff|walk_policy_contract_and_inference|wheel_policy_contract_and_inference|measured_roll_core|triangle_roll_lifecycle|triangle_roll_stand_lifecycle|notebook_lift.*)$' --output-on-failure --no-tests=error
cd "$repo_dir"
python3 scripts/check_locomotion_policies.py --package-share "$install_base/neural_controller/share/neural_controller"
python3 scripts/check_notebook_lift.py --package-share "$install_base/neural_controller/share/neural_controller"
echo 'Software ready in install-combined. No hardware stack started. Follow COMBINED_MOTION_LAB.md.'
