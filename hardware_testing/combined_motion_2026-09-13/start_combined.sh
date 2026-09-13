#!/usr/bin/env bash
set -eo pipefail
if [ "${1:-}" != "--operator-confirmed-startup-pose" ]; then
  echo 'Requires actual confirmation of the supported usual startup pose, tips down, and connected controller.' >&2
  exit 1
fi
if pgrep -f '/ros2_control_node([[:space:]]|$)' >/dev/null; then
  echo 'Hardware manager already running; reuse its session instead of restarting.' >&2
  exit 1
fi
if pgrep -f '^python3 /home/pi/robot-code-leglift/ros2_ws/install-combined/neural_controller/lib/neural_controller/motion_buttons.py' >/dev/null; then
  echo 'A button handler remains from an earlier launch. Inspect and stop that launch before starting another.' >&2
  exit 1
fi
test -e /dev/input/js0
cd /home/pi/robot-code-leglift
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/local_setup.bash
source ros2_ws/install-roll/local_setup.bash
source ros2_ws/install-combined/local_setup.bash
for package in cmd_vel_mux teleop_twist_joy joy_linux neural_controller; do
  ros2 pkg prefix "$package" >/dev/null
done
python3 scripts/check_locomotion_policies.py --package-share "$PWD/ros2_ws/install-combined/neural_controller/share/neural_controller"
exec ros2 launch neural_controller combined_motion.launch.py
