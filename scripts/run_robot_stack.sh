#!/usr/bin/env bash
# --supported represents actual operator confirmation, never an inferred pose.
set -euo pipefail
if [[ "${1:-}" != "--supported" ]]; then
  echo 'Confirm the robot is supported with joints clear, then use --supported.' >&2
  exit 2
fi
shift
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
# ROS setup files may reference unset shell variables.
set +u
source /opt/ros/jazzy/setup.bash
for overlay in install install-roll install-combined install-upper-home; do
  source "$repo/ros2_ws/$overlay/local_setup.bash"
done
set -u
for package in control_board_hardware_interface pupper_v3_description robot_calibration; do
  actual="$(ros2 pkg prefix "$package")"
  expected="$repo/ros2_ws/install-upper-home/$package"
  [[ "$actual" == "$expected" ]] || { echo "Wrong overlay for $package: $actual" >&2; exit 1; }
done
cmp "$repo/ros2_ws/src/pupper_v3_description/description/upper_home.yaml" \
    "$repo/ros2_ws/install-upper-home/pupper_v3_description/share/pupper_v3_description/description/upper_home.yaml"
if fuser /dev/spidev0.0 /dev/spidev0.1 >/dev/null 2>&1; then
  echo 'Hardware already in use. Reuse the running stack instead of starting another.' >&2
  exit 1
fi
export QUADMORPH_STARTUP_SUPPORTED_BOOT="$(cat /proc/sys/kernel/random/boot_id)"
launch_file="${1:-combined_motion.launch.py}"
if (( $# )); then shift; fi
exec ros2 launch "$repo/ros2_ws/src/neural_controller/launch/$launch_file" "$@"
