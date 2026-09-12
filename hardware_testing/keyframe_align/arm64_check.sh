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
ctest --test-dir /result/build/neural_controller \
  -R 'keyframe_controller|startup_launch|hybrid_controller_lifecycle|hybrid_policy_contract|walk_policy_contract|align_v5' --output-on-failure
ctest --test-dir /result/build/robot_calibration --output-on-failure
ctest --test-dir /result/build/joy_utils -R alignment_joystick --output-on-failure
uname -m > /result/architecture.txt
dpkg-query -W > /result/packages.txt
sha256sum /result/install/neural_controller/lib/libneural_controller.so > /result/plugin.sha256
