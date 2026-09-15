#!/usr/bin/env python3
"""Verify the approved wheel-blind lift export and its combined-launch adapter."""
import argparse,hashlib,json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def check(package_share=None):
    source=ROOT/'ros2_ws/src/neural_controller'
    share=Path(package_share) if package_share else source
    manifest=json.loads((ROOT/'policies/leg_lift_wheel/integration_manifest.json').read_text())
    for relative,expected in manifest['files'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    export=share/'launch/policy_leg_lift_wheel.json'
    assert hashlib.sha256(export.read_bytes()).hexdigest()=='20868146985fb5d8dae3e80339f35a5e727c133a1bb851d1cd2e268f8dc33636'
    policy=json.loads(export.read_text())
    assert policy['in_shape']==[None,27] and policy['policy_action_size']==8
    assert policy['contract']=='leg-lift-wheel-position-v1'
    assert policy['observation_history']==1 and policy['hub_observations'] is False
    assert policy['rotation_command_observed'] is False
    settings=yaml.safe_load((share/'launch/wheel_lift_config.yaml').read_text())['neural_controller_wheel_lift']['ros__parameters']
    for cfg,key in [('kps','kps'),('kds','kds'),('action_scales','action_scale'),('default_joint_pos','default_joint_pos'),('action_types','action_types'),('joint_lower_limits','joint_lower_limits'),('joint_upper_limits','joint_upper_limits')]:
        assert settings[cfg]==policy[key],cfg
    assert settings['calibration_required'] and settings['action_types']==['position']*12
    launch=(share/'launch/combined_motion.launch.py').read_text()
    assert "path('wheel_lift_config.yaml')" in launch and "'neural_controller_wheel_lift'" in launch
    assert "'neural_controller_notebook_lift'" not in launch
    if package_share:
        for name in ('wheel_lift_config.yaml','combined_motion.launch.py'):
            assert (share/'launch'/name).read_bytes()==(source/'launch'/name).read_bytes(),name
        prefix=share.parent.parent
        assert (prefix/'lib/libwheel_lift_controller.so').is_file()
        assert (prefix/'lib/neural_controller/motion_buttons.py').read_bytes()==(source/'scripts/motion_buttons.py').read_bytes()
        assert 'neural_controller/WheelLiftController' in (share/'neural_controller.xml').read_text()
    print('PASS approved export, position contract, geometry/source hashes and combined integration'+(' (installed)' if package_share else ''))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package-share',type=Path);check(p.parse_args().package_share)
