#!/usr/bin/env python3
"""Capture an explicit triangle-to-encoder mapping; never moves or recalibrates the robot."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros2_ws/src/robot_calibration'))
from robot_calibration.storage import capture_lock, current_session, directory, load_current
from robot_calibration.cli import sample_robot


def make_mapping(plan, calibration, positions):
    if len(positions) != 12 or not all(math.isfinite(q) for q in positions):
        raise ValueError('Need twelve finite measured angles')
    offsets = [0.0] * 12
    for i, (q, model) in enumerate(zip(positions, plan['initial'])):
        if i % 3 == 2:
            offsets[i] = q - model  # Preserve encoder winding, never choose a nearest half-turn.
        elif abs(q - model) > .03:
            raise ValueError(f'{plan["joint_names"][i]} must be within 0.03 rad of {model}; measured {q}. '
                             'No automatic approach or proximal offset guessing is implemented.')
    return dict(schema_version=1, mapping_id=uuid.uuid4().hex,
                calibration_id=calibration['calibration_id'], encoder_session_id=calibration['encoder_session_id'],
                wheel_home=calibration['wheel_home'], joint_names=plan['joint_names'],
                operator_confirmed_inverted_start=True, axial_gap_m=.009,
                plan_sha256=plan['plan_sha256'], captured_q=positions, model_to_encoder_offset=offsets,
                convention='Confirmed rigid triangles, same physical start as reviewed model; proximal coordinates unchanged')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['capture', 'status'])
    p.add_argument('--operator-confirmed-inverted-start', action='store_true',
                   help='ONLY after actual confirmation of 9 mm gaps, rigid triangles and matching inverted start pose')
    p.add_argument('--replace', action='store_true')
    p.add_argument('--roll', action='store_true', help='Use the simultaneous roll plan and separate mapping')
    a = p.parse_args()
    plan_path = ROOT / ('ros2_ws/src/neural_controller/launch/triangle_roll_plan.json' if a.roll else 'ros2_ws/src/neural_controller/launch/inverted_triangle_plan.json')
    plan = json.loads(plan_path.read_text())
    expected = json.loads((ROOT / ('hardware_testing/triangle_roll/export_provenance.json' if a.roll else 'hardware_testing/inverted_triangle/source/export_provenance.json')).read_text())
    if hashlib.sha256(plan_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected['export_sha256']:
        raise ValueError('Unreviewed exported plan')
    path = directory() / ('triangle-roll-map.json' if a.roll else 'inverted-triangle-map.json')
    if a.command == 'capture' and not a.operator_confirmed_inverted_start:
        if not sys.stdin.isatty():
            raise ValueError('Actual physical confirmation is required before capture')
        print('Support the robot. Motion controllers must be inactive. All four rigid triangles must match '
              'the reviewed inverted start, with 9 mm axial gaps. This is separate from startup ring homing.')
        print('Proximal target FR/FL/BR/BL: [1,-0.29], [-1,0.29], [1,-0.29], [-1,0.29] radians.')
        if input('Type INVERTED only when this physical setup is confirmed: ').strip() != 'INVERTED':
            raise ValueError('Cancelled; no mapping saved')
    with capture_lock():
        calibration = load_current()
        if a.command == 'capture':
            if path.exists() and not a.replace:
                raise ValueError('Mapping exists; inspect status, or explicitly replace after reconfirming the pose')
            positions = sample_robot(15., current_session())
            if load_current()['calibration_id'] != calibration['calibration_id']:
                raise ValueError('Calibration changed')
            record = make_mapping(plan, calibration, positions)
            history = directory() / 'history'
            history.mkdir(exist_ok=True)
            payload = json.dumps(record, indent=2) + '\n'
            (history / ('inverted-' + record['mapping_id'] + '.json')).write_text(payload)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(payload); tmp.replace(path)
        record = json.loads(path.read_text())
        if record['calibration_id'] != calibration['calibration_id'] or record['plan_sha256'] != plan['plan_sha256']:
            raise ValueError('Triangle reference is stale')
    print(json.dumps(record, indent=2))
    print(f'Mapping: {path}. Shared startup calibration preserved. No motor commands sent.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, ImportError) as e:
        print('TRIANGLE REFERENCE NOT READY: ' + str(e), file=sys.stderr)
        raise SystemExit(1)
