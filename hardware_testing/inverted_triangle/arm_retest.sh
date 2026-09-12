#!/usr/bin/env bash
# Diagnose shared-suite failures without changing a runtime gate or rebuilding.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /result/install/local_setup.bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="/triangle-workspace/src/robot_calibration:${PYTHONPATH:-}"
if [[ "${1:-storage}" == joy ]]; then
  export ALIGN_JOY_TEST_EXE=/result/install/joy_utils/lib/joy_utils/estop_controller
  timeout 180 python3 -m pytest /triangle-workspace/src/joy_utils/test/alignment_joystick_test.py -v -s -o cache_dir=/tmp/triangle-joy-cache
else
  timeout 180 python3 -m pytest /triangle-workspace/src/robot_calibration/test/test_calibration.py -v -s -o cache_dir=/tmp/triangle-storage-cache
fi
