#!/usr/bin/env bash
# Explicit trial only. --supported must reflect actual operator confirmation.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
 echo 'Usage: bash scripts/run_notebook_lift.sh --supported [--align]. Starts the dedicated inactive lift trial; --align enables gated startup-home+180deg rotation. Verifies candidate/upper-home overlays. Hardware activation performs upper-joint homing; read STARTUP_UPPER_HOME.md and obtain actual support confirmation first.'
 exit 0
fi
[[ ( $# == 1 || ( $# == 2 && ${2:-} == --align ) ) && $1 == --supported ]] || { echo 'Confirm actual support and clear joints, then use --supported; see STARTUP_UPPER_HOME.md.' >&2; exit 2; }
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source /opt/ros/jazzy/setup.bash
for overlay in install install-roll install-combined install-upper-home install-notebook-lift; do source "$repo/ros2_ws/$overlay/local_setup.bash"; done
set -u
for package in control_board_hardware_interface pupper_v3_description robot_calibration; do
 [[ "$(ros2 pkg prefix "$package")" == "$repo/ros2_ws/install-upper-home/$package" ]] || { echo "Wrong saved-home overlay for $package" >&2; exit 1; }
done
[[ "$(ros2 pkg prefix neural_controller)" == "$repo/ros2_ws/install-notebook-lift/neural_controller" ]] || { echo 'Wrong notebook lift overlay' >&2; exit 1; }
cmp "$repo/ros2_ws/src/pupper_v3_description/description/upper_home.yaml" "$repo/ros2_ws/install-upper-home/pupper_v3_description/share/pupper_v3_description/description/upper_home.yaml"
python3 "$repo/scripts/check_notebook_lift.py" --package-share "$(ros2 pkg prefix --share neural_controller)"
python3 "$repo/scripts/notebook_lift/build_receipt.py" check --package-share "$(ros2 pkg prefix --share neural_controller)"
if fuser /dev/spidev0.0 /dev/spidev0.1 >/dev/null 2>&1; then echo 'Hardware already owned. Reuse that session; do not start a duplicate stack.' >&2; exit 1; fi
export QUADMORPH_STARTUP_SUPPORTED_BOOT="$(cat /proc/sys/kernel/random/boot_id)"
trial=notebook_lift_trial.launch.py
[[ ${2:-} == --align ]] && trial=notebook_lift_align_trial.launch.py
exec ros2 launch neural_controller "$trial"
