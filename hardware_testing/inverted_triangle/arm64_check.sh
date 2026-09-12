#!/usr/bin/env bash
# Offline ARM64/QEMU check. /source is read-only ros2_ws/src, /result is disposable output.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
mkdir -p /triangle-workspace/src
cp -a /source/neural_controller /source/robot_calibration /source/joy_utils \
  /source/pupper_v3_description /source/control_board_hardware_interface /source/animation_controller_py /triangle-workspace/src/
cd /triangle-workspace
colcon --log-base /result/log build --build-base /result/build --install-base /result/install \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source /result/install/local_setup.bash
export LD_LIBRARY_PATH="/result/install/neural_controller/lib:${LD_LIBRARY_PATH:-}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
uname -m > /result/architecture.txt
dpkg-query -W > /result/packages.txt
sha256sum /result/install/neural_controller/lib/libneural_controller.so > /result/plugin.sha256
test_status=0
ctest --test-dir /result/build/neural_controller \
  -R 'inverted_triangle|keyframe_controller|keyframe_position|walk_policy_contract|wheel_policy_contract|hybrid_controller_lifecycle' --output-on-failure || test_status=1
timeout 180 python3 -m pytest /triangle-workspace/src/neural_controller/test/startup_launch_test.py -q || test_status=1
ctest --test-dir /result/build/robot_calibration --output-on-failure || test_status=1
ctest --test-dir /result/build/joy_utils -R alignment_joystick --output-on-failure || test_status=1
exit "$test_status"
