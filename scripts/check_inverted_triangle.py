#!/usr/bin/env python3
"""Read-only triangle artifact/hardware compatibility check; no ROS activation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import yaml
from check_align_v5 import check as check_geometry

ROOT = Path(__file__).resolve().parents[1]
NAME = 'neural_controller_inverted_triangle'


def require(value, message):
    if not value:
        raise ValueError(message)


def check(install=None):
    controller = ROOT / 'ros2_ws/src/neural_controller'
    description = ROOT / 'ros2_ws/src/pupper_v3_description'
    if install:
        for package in ('robot_calibration', 'control_board_hardware_interface', 'neural_controller', 'joy_utils', 'pupper_v3_description'):
            prefix = Path(subprocess.check_output(['ros2', 'pkg', 'prefix', package], text=True).strip()).resolve()
            require(prefix.is_relative_to(install.resolve()), f'Wrong overlay: {package}: {prefix}')
            if package == 'neural_controller':
                installed = prefix / 'share' / package
                for name in ('inverted_triangle_plan.json', 'inverted_triangle_config.yaml', 'inverted_triangle_trial.launch.py', 'calibration_hardware.yaml'):
                    require((installed/'launch'/name).read_bytes().replace(b'\r\n', b'\n') ==
                            (controller/'launch'/name).read_bytes().replace(b'\r\n', b'\n'), 'Stale installed ' + name)
                require((prefix/'lib/libneural_controller.so').is_file(), 'Missing installed controller')
                controller = installed
            if package == 'pupper_v3_description':
                description = prefix / 'share' / package
    check_geometry(controller, description)  # Includes current axes/origins/CAN and 9 mm geometry.
    source = ROOT / 'hardware_testing/inverted_triangle/source'
    manifest = json.loads((source/'export_provenance.json').read_text())
    plan = json.loads((source/'plan.json').read_text())
    require(hashlib.sha256((source/'plan.json').read_bytes()).hexdigest() == manifest['plan_sha256'], 'Original plan changed')
    for stage in plan['stages']:
        require(hashlib.sha256((source/stage['candidate']).read_bytes()).hexdigest() == stage['candidate_sha256'], 'Source candidate changed')
    raw = (controller/'launch/inverted_triangle_plan.json').read_bytes().replace(b'\r\n', b'\n')
    require(hashlib.sha256(raw).hexdigest() == manifest['export_sha256'], 'Unreviewed runtime plan')
    data = json.loads(raw)
    cfg = yaml.safe_load((controller/'launch/inverted_triangle_config.yaml').read_text())
    c = cfg[NAME]['ros__parameters']
    require(cfg['controller_manager']['ros__parameters'][NAME]['type'] == 'neural_controller/InvertedTriangleController', 'Wrong plugin')
    require(data['axial_gap_m'] == .009 and data['total_steps'] == 67278, 'Unexpected gap/trajectory')
    require(c['joint_names'] == data['joint_names'] and c['action_types'] == ['position']*12, 'Joint mapping/mode mismatch')
    require(c['kps'] == data['kp'] and c['kds'] == data['kd'], 'Gain mismatch')
    require(c['default_joint_pos'] == data['initial'] and abs(c['max_body_angle']-math.radians(8)) < 1e-12, 'Start pose/tilt metadata mismatch')
    require(c['use_imu'] and c['repeat_action'] == 1 and c['gain_multiplier'] == 1 and c['estop_kd'] == 0, 'Runtime/stop mismatch')
    joints = {j.attrib['name']: j for j in ET.parse(description/'description/components.xacro').findall('.//joint')}
    for i, name in enumerate(data['joint_names']):
        hw = {p.attrib['name']: float(p.text) for p in joints[name].findall('param')}
        require('hard_limit_min' not in hw and 'hard_limit_max' not in hw, 'Hardware hard-limit gain override: ' + name)
        require(data['kp'][i] <= hw['kp_max'] and data['kd'][i] <= hw['kd_max'], 'Gain clamps: ' + name)
        require(c['joint_lower_limits'][i] == hw['position_min'] and c['joint_upper_limits'][i] == hw['position_max'], 'Limit mismatch: ' + name)
        for segment in data['segments']:
            require(hw['position_min'] <= segment['target'][i] <= hw['position_max'], 'Unmapped target outside envelope: ' + name)
    print('PASS: original 9 mm plan/candidates, exported phase hashes, current CAN/axes/origins, continuous hub profile, PD gains and limits')
    print('Mapping, physical clearances, target timing and loaded motion require the documented lab checks.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install-base', type=Path)
    check(p.parse_args().install_base)
