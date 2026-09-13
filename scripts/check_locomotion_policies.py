#!/usr/bin/env python3
"""Read-only checks of latest locomotion exports and their installed runtime configuration."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml
from check_walk_policy import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-share', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    share = args.package_share or repo / 'ros2_ws/src/neural_controller'
    walking = check(repo, share)
    manifest = json.loads((repo / 'policies/latest.json').read_text())['policies']['wheel']
    raw = (share / 'launch/policy_wheel.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['export_sha256'], 'Wrong wheel weights or LFS pointer'
    wheel = json.loads(raw)
    config = yaml.safe_load((share / 'launch/config.yaml').read_text())
    params = config['neural_controller_wheel']['ros__parameters']
    assert params['model_path'].endswith('/launch/policy_wheel.json')
    for key in ('default_joint_pos', 'action_types', 'kps', 'kds', 'init_kps', 'init_kds'):
        assert params[key] == wheel[key], f'Wheel YAML/export mismatch: {key}'
    assert params['joint_names'] == config['neural_controller_walk_v2']['ros__parameters']['joint_names']
    assert params['estop_kd'] == 1.0 and params['repeat_action'] == 10
    assert params['gain_multiplier'] == 1.0
    assert wheel['in_shape'] == [None, 132] and wheel['observation_history'] == 4
    assert wheel['action_types'] == ['position', 'position', 'velocity'] * 4
    assert (share / 'launch/locomotion_trial.launch.py').is_file()
    trial = yaml.safe_load((share / 'launch/locomotion_wheel_trial.yaml').read_text())
    assert trial['neural_controller_wheel']['ros__parameters'] == {
        'calibration_required': True, 'cmd_vel_topic': '/wheel_cmd_vel'}
    print(json.dumps({'status': 'PASS', 'walking_sha256': walking['export_sha256'],
                      'wheel_sha256': manifest['export_sha256'],
                      'triangle': 'neural_controller_walk_v2', 'circle': 'neural_controller_wheel',
                      'scope': 'Source/installed files only; hardware validation pending.'}, indent=2))


if __name__ == '__main__':
    main()
