#!/usr/bin/env python3
"""Verify saved target consistency and an expanded real-hardware URDF, without motion."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path)
    args = parser.parse_args()
    target = yaml.safe_load((ROOT/'ros2_ws/src/pupper_v3_description/description/upper_home.yaml').read_text())
    evidence = json.loads((ROOT/'hardware_testing/start_pose/upper_home.json').read_text())
    profile = json.loads((ROOT/'hardware_testing/start_pose/start_pose.json').read_text())
    expected = {f'leg_{leg}_{motor}' for leg in ('front_r', 'front_l', 'back_r', 'back_l') for motor in (1, 2)}
    assert set(target) == expected
    assert target == evidence['joint_positions']
    assert target == {k: profile['joint_positions'][k] for k in expected}
    if args.urdf:
        root = ET.parse(args.urdf).getroot()
        parameters = {p.attrib['name']: p.text for p in root.findall('./ros2_control/hardware/param')}
        assert parameters['startup_upper_home'] == 'true'
        assert {k for k in parameters if k.startswith('upper_home_')} == {'upper_home_'+k for k in expected}
        joints = {j.attrib['name']: {p.attrib['name']: p.text for p in j.findall('param')}
                  for j in root.findall('./ros2_control/joint')}
        for name, angle in target.items():
            assert float(parameters['upper_home_'+name]) == angle
            assert float(joints[name]['position_min']) <= angle <= float(joints[name]['position_max'])
    print('Eight upper-home targets agree across runtime, evidence, profile and supplied URDF.')


if __name__ == '__main__':
    main()
