#!/usr/bin/env python3
"""Capture confirmed suspended tips-up pose; no motor commands or encoder zero changes."""
import argparse
import json
import math
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ros2_ws/src/robot_calibration'))
from robot_calibration.storage import capture_lock,current_session,directory,load_current
from robot_calibration.cli import sample_robot

def make_reference(calibration,q,plan):
    if len(q)!=12 or not all(math.isfinite(x) for x in q):raise ValueError('Twelve finite encoder positions required')
    # Same installed hardware limits as the pinned roll contract; no offsets.
    import yaml
    cfg=yaml.safe_load((ROOT/'ros2_ws/src/neural_controller/launch/triangle_roll_config.yaml').read_text())['neural_controller_triangle_roll']['ros__parameters']
    target=list(q)
    for leg in range(4):target[3*leg+2]+=(-math.pi if leg%2==0 else math.pi)
    for i in range(12):
        if not cfg['joint_lower_limits'][i]<=min(q[i],target[i])<=max(q[i],target[i])<=cfg['joint_upper_limits'][i]:raise ValueError('Half-turn exceeds joint limits')
    return dict(schema_version=1,purpose='supported_hub_only_v1',reference_id=uuid.uuid4().hex,
        calibration_id=calibration['calibration_id'],encoder_session_id=calibration['encoder_session_id'],
        wheel_home=calibration['wheel_home'],joint_names=plan['joint_names'],plan_sha256=plan['plan_sha256'],
        operator_confirmed_tips_up=True,captured_q=q,target_q=target,
        convention='Hold measured proximal encoder coordinates; right hubs -pi and left hubs +pi')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-confirmed-tips-up',action='store_true')
    parser.add_argument('--replace',action='store_true')
    args=parser.parse_args()
    if not args.operator_confirmed_tips_up:raise ValueError('Actual operator confirmation of suspended tips-up pose is required')
    plan=json.loads((ROOT/'ros2_ws/src/neural_controller/launch/triangle_roll_plan.json').read_text())
    with capture_lock():
        cal=load_current();path=directory()/'hub-roll-reference.json'
        if path.exists() and not args.replace:raise ValueError('Reference exists; inspect or explicitly replace after reconfirmation')
        q=sample_robot(15.,current_session())
        if load_current()['calibration_id']!=cal['calibration_id']:raise ValueError('Calibration changed')
        record=make_reference(cal,q,plan)
        payload=json.dumps(record,indent=2)+'\n'
        history=directory()/'history';history.mkdir(exist_ok=True)
        (history/('hub-roll-'+record['reference_id']+'.json')).write_text(payload)
        tmp=path.with_suffix('.tmp');tmp.write_text(payload);tmp.replace(path)
    print(payload);print(f'Reference saved: {path}. No motor commands sent; startup calibration preserved.')

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError) as e:print('HUB REFERENCE NOT READY: '+str(e),file=sys.stderr);raise SystemExit(1)
