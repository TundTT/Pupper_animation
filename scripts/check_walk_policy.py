#!/usr/bin/env python3
"""Read-only preflight for the selected walking policy; never connects to motors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


def check(repo, package_share=None):
    controller = package_share or repo / 'ros2_ws/src/neural_controller'
    manifest = json.loads((repo / 'hardware_testing/walk_gap9_2026-09-11/selection.json').read_text())
    policy_path = controller / 'launch/policy_walk_v2.json'
    data = policy_path.read_bytes()
    assert not data.startswith(b'version https://git-lfs'), 'Fetch the actual policy with git lfs pull'
    assert hashlib.sha256(data).hexdigest() == manifest['export_sha256'], 'Wrong walking policy file'
    policy = json.loads(data)
    config = yaml.safe_load((controller / 'launch/config.yaml').read_text())
    settings = config['neural_controller_walk_v2']['ros__parameters']
    assert settings['model_path'].endswith('/launch/policy_walk_v2.json')
    assert policy['in_shape'] == [None, 144] and policy['observation_history'] == 4
    assert policy['layers'][-1]['shape'] == [None, 12]
    assert policy['layers'][-1]['activation'] == 'tanh'
    assert policy['action_scale'] == [.5, .25, 1.1] * 4
    for key in ('joint_names', 'default_joint_pos', 'action_types'):
        assert settings[key] == policy[key], f'Config/export mismatch: {key}'
    assert policy['action_types'] == ['position'] * 12
    assert settings['gain_multiplier'] == 1.0
    assert settings['init_kps'] == [policy['kp']] * 12
    assert settings['init_kds'] == [policy['kd']] * 12
    assert settings['max_body_angle'] == .65
    assert settings['repeat_action'] == 10
    assert config['controller_manager']['ros__parameters']['update_rate'] == 520
    assert policy['kp'] == 5.0 and policy['kd'] == .25
    assert policy['command_low'] == [-.35, -.15, -.8]
    assert policy['command_high'] == [.35, .15, .8]
    assert policy['orientation_command'] == [0, 0, 1]
    buttons = config['joy_util_node']['ros__parameters']
    mapping = dict(zip(buttons['switch_button_indices'], buttons['controller_names']))
    assert mapping[3] == 'neural_controller_walk_v2', 'Square must select walking'
    assert mapping[2] == 'neural_controller_wheel', 'Triangle must preserve wheel mode'
    # X is handled by the existing dedicated alignment path, not the generic map.
    assert 0 not in mapping, 'X must not also bind a generic controller switch'
    assert buttons['wheel_align_hybrid_button_index'] == 0
    assert buttons['wheel_align_hybrid_controller_name'] == 'neural_controller_wheel_align_hybrid'
    root = ET.parse(repo / 'ros2_ws/src/pupper_v3_description/description/components.xacro')
    hardware = {j.attrib['name']: {p.attrib['name']: float(p.text) for p in j.findall('param')}
                for j in root.findall('.//joint')}
    ranges = []
    for i, name in enumerate(policy['joint_names']):
        h = hardware[name]
        lower = max(policy['default_joint_pos'][i] - abs(policy['action_scale'][i]),
                    policy['joint_lower_limits'][i])
        upper = min(policy['default_joint_pos'][i] + abs(policy['action_scale'][i]),
                    policy['joint_upper_limits'][i])
        assert h['position_min'] <= lower < upper <= h['position_max'], f'{name}: soft limit mismatch'
        assert h.get('hard_limit_min', -math.inf) < lower
        assert upper < h.get('hard_limit_max', math.inf)
        assert h['kp_max'] >= policy['kp'] and h['kd_max'] >= policy['kd']
        assert h['effort_max'] == 3.0
        assert h['homing_velocity'] == h['homing_kp'] == h['homing_torque_threshold'] == 0.0
        margin = None
        if 'hard_limit_min' in h:
            assert 'homing_reference_raw' in h
            margin = min(lower - h['hard_limit_min'], h['hard_limit_max'] - upper)
        ranges.append(dict(joint=name,minimum=lower,maximum=upper,hard_limit_margin=margin))
    return dict(status='PASS',policy=str(policy_path),export_sha256=manifest['export_sha256'],
                checkpoint_step=manifest['checkpoint_step'],joint_target_ranges=ranges,
                nominal_policy_hz=52.0,
                timing_note='520/10 is 52 Hz nominal; existing hardware scheduling aims for 50 Hz. Measure observation topic rate on the robot.',
                scope='Files and target envelopes only; no ROS build, motor activation, measured-state overshoot, or hardware validation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-share', type=Path, help='Check installed policy/config instead of source')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check(Path(__file__).resolve().parents[1], args.package_share)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
