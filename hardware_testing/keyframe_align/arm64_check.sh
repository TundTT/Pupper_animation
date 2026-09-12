#!/usr/bin/env bash
# Run inside Dockerfile.arm64's image with /source read-only and /result writable.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
mkdir -p /workspace/src
cp -a /source/neural_controller /source/robot_calibration /source/joy_utils /source/pupper_v3_description /workspace/src/
cd /workspace
colcon --log-base /result/log build --build-base /result/build --install-base /result/install \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source /result/install/local_setup.bash
export LD_LIBRARY_PATH="/result/install/neural_controller/lib:${LD_LIBRARY_PATH:-}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
uname -m > /result/architecture.txt
dpkg-query -W > /result/packages.txt
sha256sum /result/install/neural_controller/lib/libneural_controller.so > /result/plugin.sha256
# Preserve every group's outcome, including emulation failures. Never report a
# successful preparation merely because a later independent test passed.
test_status=0
ctest --test-dir /result/build/neural_controller \
  -R 'keyframe_controller|keyframe_manager|hybrid_controller_lifecycle|hybrid_policy_contract|walk_policy_contract|align_v5' --output-on-failure || test_status=1
# QEMU startup/import time is not a motor deadline. Run the same pure launch
# assertions with a bounded emulation allowance; hardware timing gates are unchanged.
timeout 180 python3 -m pytest /workspace/src/neural_controller/test/startup_launch_test.py -q || test_status=1
ctest --test-dir /result/build/robot_calibration --output-on-failure || test_status=1
ctest --test-dir /result/build/joy_utils -R alignment_joystick --output-on-failure || test_status=1
exit "$test_status"
