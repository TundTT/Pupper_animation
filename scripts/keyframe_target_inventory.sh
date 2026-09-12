#!/usr/bin/env bash
# Read-only target inspection. Does not launch, stop, home, activate or calibrate.
# Save stdout to an operator-selected file for the pre-lab comparison.
set -eo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf '%s\n' '=== Host and source ==='
hostname
uname -m
cat /etc/os-release
git -C "$repo_dir" rev-parse HEAD
git -C "$repo_dir" status --short
printf '%s\n' '=== Installed ROS packages ==='
source /opt/ros/jazzy/setup.bash
if [[ -f "$repo_dir/ros2_ws/install/local_setup.bash" ]]; then
  source "$repo_dir/ros2_ws/install/local_setup.bash"
fi
for package in controller_manager controller_interface hardware_interface realtime_tools generate_parameter_library robot_calibration neural_controller joy_utils; do
  if prefix=$(ros2 pkg prefix "$package" 2>/dev/null); then
    printf '%s %s\n' "$package" "$prefix"
    sed -n 's/.*<version>\(.*\)<\/version>.*/version=\1/p' "$prefix/share/$package/package.xml"
  else
    printf 'MISSING %s\n' "$package"
  fi
done
printf '%s\n' '=== Gamepad device names ==='
for device in /sys/class/input/js*/device/name; do [[ ! -f "$device" ]] || cat "$device"; done
printf '%s\n' '=== Existing controller managers ==='
pgrep -af '/ros2_control_node([[:space:]]|$)' || true
printf '%s\n' '=== Launch-user scheduling limits ==='
ulimit -r
ulimit -l
for pid in $(pgrep -f '/ros2_control_node([[:space:]]|$)' || true); do
  chrt -p "$pid" || true
  cat "/proc/$pid/limits" || true
done
printf '%s\n' '=== Legacy launch service status ==='
systemctl is-active dpad-launch-trigger.service robot.service || true
systemctl is-enabled dpad-launch-trigger.service robot.service || true
printf '%s\n' 'Inventory only; no readiness or physical-calibration assertion.'
