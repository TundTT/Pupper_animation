#!/usr/bin/env bash
# Test an already-built isolated overlay; never starts a hardware node.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source "$1/install/local_setup.bash"
export LD_LIBRARY_PATH="$1/install/neural_controller/lib:${LD_LIBRARY_PATH:-}"
test_status=0
ctest --test-dir "$1/build/neural_controller" \
  -R 'inverted_triangle|startup_launch|keyframe_controller|keyframe_position|walk_policy_contract|wheel_policy_contract|hybrid_controller_lifecycle' --output-on-failure || test_status=1
ctest --test-dir "$1/build/robot_calibration" --output-on-failure || test_status=1
ctest --test-dir "$1/build/joy_utils" -R alignment_joystick --output-on-failure || test_status=1
exit "$test_status"
