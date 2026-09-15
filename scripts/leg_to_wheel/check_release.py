#!/usr/bin/env python3
"""Verify the self-contained leg-to-wheel source release without ROS or JAX."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    manifest=json.loads((ROOT/'policies/leg_to_wheel/manifest.json').read_text())
    for name,digest in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise SystemExit(f'FAIL: release hash mismatch: {name}')
    policy=json.loads((ROOT/manifest['policy_path']).read_text())
    run=json.loads((ROOT/'policies/leg_to_wheel/checkpoint/run.json').read_text())
    assert policy['behavior']=='leg_to_wheel' and policy['deployment_status']=='source_bundle_requires_hardware_adapter'
    assert policy['in_shape']==[None,140] and policy['single_observation_size']==35
    assert policy['observation_history']==4 and policy['command_states']==['stand','FL','FR','BR','BL']
    assert policy['command_to_foot']==[-1,1,0,2,3] and policy['action_types']==['position']*12
    assert policy['action_scale']==run['config']['action_scale'] and policy['ctrl_dt']==.02
    assert policy['default_joint_pos']==run['home_joint_pos']
    assert policy['kps']==run['kp'] and policy['kds']==run['kd']
    assert policy['joint_lower_limits']==run['joint_lower_limits'] and policy['joint_upper_limits']==run['joint_upper_limits']
    for kind,expected in [('nominal',195),('physics',179)]:
        report=json.loads((ROOT/f'policies/leg_to_wheel/evidence/gentle_hip_{kind}.json').read_text())
        assert len(report['runs'])==195 and sum(r['passed'] for r in report['runs'])==expected
        assert not any(r['fell'] for r in report['runs'])
    assert manifest['hardware_deployed'] is False and manifest['hardware_tested'] is False
    print(f"PASS: {len(manifest['files'])} release hashes; policy contract; nominal 195/195, randomized 179/195; source-only status")

if __name__=='__main__':main()
