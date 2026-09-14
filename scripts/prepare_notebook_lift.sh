#!/usr/bin/env bash
# Build/check only. Does not contact or launch robot hardware.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
 echo 'Usage: bash scripts/prepare_notebook_lift.sh [--build]. Default: source artifact checks. --build: build isolated ROS overlay and run pure-core/fake-interface tests; no hardware launch.'
 exit 0
fi
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
python3 scripts/check_notebook_lift.py
[[ $# == 0 ]] && exit 0
[[ $# == 1 && $1 == --build ]] || { echo 'Use --help' >&2; exit 2; }
set +u
source /opt/ros/jazzy/setup.bash
for overlay in install install-roll install-combined install-upper-home; do source "$repo/ros2_ws/$overlay/local_setup.bash"; done
set -u
cd ros2_ws
colcon build --packages-select neural_controller --build-base build-notebook-lift --install-base install-notebook-lift --cmake-args -DBUILD_TESTING=ON -DQUADMORPH_FOCUSED_TESTS=ON
set +u
source install-notebook-lift/local_setup.bash
set -u
ROS_DOMAIN_ID=194 ROS_LOCALHOST_ONLY=1 ctest --test-dir build-notebook-lift/neural_controller --output-on-failure --no-tests=error -R 'notebook_lift|walking_frame|walking_handoff|keyframe_position|align_v5_motion_contract'
python3 "$repo/scripts/check_notebook_lift.py" --package-share "$(ros2 pkg prefix --share neural_controller)"
python3 "$repo/scripts/notebook_lift/build_receipt.py" write --package-share "$(ros2 pkg prefix --share neural_controller)"
