#!/usr/bin/env python3
"""Read-only source/installed candidate verification; never starts ROS or motors."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def check(package_share=None):
    manifest=json.loads((ROOT/'hardware_testing/notebook_lift/release.json').read_text())
    for relative,expected in manifest['files'].items():
        path=ROOT/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Candidate file hash mismatch: '+relative)
    share=Path(package_share) if package_share else ROOT/'ros2_ws/src/neural_controller'
    for name in ['policy_notebook_lift.json','notebook_lift_config.yaml','notebook_lift_trial.launch.py','notebook_lift_align_config.yaml','notebook_lift_align_trial.launch.py']:
        source=ROOT/'ros2_ws/src/neural_controller/launch'/name
        if (share/'launch'/name).read_bytes()!=source.read_bytes():raise ValueError('Installed candidate differs: '+name)
    policy=json.loads((share/'launch/policy_notebook_lift.json').read_text())
    assert policy['behavior']=='notebook_lift_v4' and policy['trained_task']=='lift'
    assert policy['in_shape']==[None,288] and policy['policy_action_size']==8 and policy['motor_command_count']==12
    assert policy['checkpoint_sha256']==manifest['checkpoint_sha256']
    assert policy['action_types']==['position']*12
    assert policy['kps']==[5,5,8]*4 and policy['kds']==[.25,.25,1]*4
    assert policy['runtime_settings']['hub_alignment_rotation'] is False
    assert policy['runtime_settings']['desired_clearance_m']==.008
    if package_share:
        prefix=share.parent.parent
        if not (prefix/'lib/libnotebook_lift_controller.so').is_file():raise ValueError('Installed candidate plugin library missing')
        xml=(share/'neural_controller.xml').read_text()
        assert 'neural_controller/NotebookLiftController' in xml
    print('PASS notebook lift artifact/config/ABI hashes'+(' and installed candidate' if package_share else '')+'. Experimental lift candidate with optional PD alignment; no hardware/task certification.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-share',type=Path);a=p.parse_args()
    try:check(a.package_share)
    except (OSError,ValueError,AssertionError,KeyError) as e:print('FAIL:',e,file=sys.stderr);raise SystemExit(1)
