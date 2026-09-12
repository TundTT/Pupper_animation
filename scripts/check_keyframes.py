#!/usr/bin/env python3
"""Read-only source/installed compatibility check; never starts hardware."""
import argparse,hashlib,json,subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
from check_align_v5 import check as check_unchanged_hardware_geometry

ROOT=Path(__file__).resolve().parents[1]
NAME='neural_controller_keyframe_align'

def require(value,message):
    if not value:raise ValueError(message)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-base',type=Path)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'hardware_testing/keyframe_align/source_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        content=(ROOT/name).read_bytes()
        if name.endswith('.hpp'):content=content.replace(b'\r\n',b'\n')
        require(hashlib.sha256(content).hexdigest()==digest,f'Unreviewed source/config: {name}')
    controller=ROOT/'ros2_ws/src/neural_controller';description=ROOT/'ros2_ws/src/pupper_v3_description'
    if args.install_base:
        base=args.install_base.resolve()
        for package in ('robot_calibration','control_board_hardware_interface','neural_controller','joy_utils','pupper_v3_description','animation_controller_py'):
            prefix=Path(subprocess.check_output(['ros2','pkg','prefix',package],text=True).strip()).resolve()
            require(prefix.is_relative_to(base),f'Wrong overlay for {package}: {prefix}')
            if package=='neural_controller':controller=prefix/'share'/package
            if package=='pupper_v3_description':description=prefix/'share'/package
        for name in ('config.yaml','keyframe_config.json','keyframe_trial.launch.py','alignment_trial.launch.py','calibration_hardware.yaml'):
            require((controller/'launch'/name).read_bytes().replace(b'\r\n',b'\n')==(ROOT/'ros2_ws/src/neural_controller/launch'/name).read_bytes().replace(b'\r\n',b'\n'),f'Stale installed {name}')
        require((controller.parents[1]/'lib/libneural_controller.so').is_file(),'Missing plugin library')
        header_prefix='ros2_ws/src/neural_controller/include/'
        for name,digest in manifest['files'].items():
            if name.startswith(header_prefix):
                installed=controller.parents[1]/'include'/name.removeprefix(header_prefix)
                require(hashlib.sha256(installed.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,f'Stale installed header: {name}')
    # v5 is preserved; its existing validator independently checks all twelve
    # hardware axes, origins, CAN channels, continuous hubs and model geometry.
    check_unchanged_hardware_geometry(controller,description)
    cfg=yaml.safe_load((controller/'launch/config.yaml').read_text())
    require(cfg['controller_manager']['ros__parameters'][NAME]['type']=='neural_controller/KeyframeController','Wrong plugin')
    params=cfg[NAME]['ros__parameters'];old=cfg['neural_controller_wheel_align_hybrid']['ros__parameters']
    require(cfg['controller_manager']['ros__parameters']['update_rate']==520 and params['repeat_action']==1,'Wrong deterministic update cadence')
    for name in ('joint_names','kps','kds','init_kps','init_kds'):
        require(params[name]==old[name],f'Hardware mapping/gain mismatch: {name}')
    require(params['action_types']==['position']*12,'Keyframe alignment requires all angle commands')
    require(params['model_path'].endswith('/keyframe_config.json'),'Wrong keyframe config path')
    require(params['use_imu'] and params['estop_kd']==0 and params['gain_multiplier']==1,'Sensor/stop/gain mismatch')
    # The hardware writer can synthesize gains at opted-in hard limits. This
    # zero-torque fault contract is restricted to the reviewed wheel profile.
    joints={j.attrib['name']:j for j in ET.parse(description/'description/components.xacro').findall('.//joint')}
    for name in params['joint_names']:
        hw={p.attrib['name']:p.text for p in joints[name].findall('param')}
        require('hard_limit_min' not in hw and 'hard_limit_max' not in hw,f'Zero-torque fault requires review of hard-limit gain overrides: {name}')
    motion=json.loads((controller/'launch/keyframe_config.json').read_text())
    require(len(motion['poses'])==4 and all(len(p)==8 for p in motion['poses']),'Wrong pose dimensions')
    require(motion['wheel_control_mode']=='position_pid','Wrong wheel control mode')
    require(0<motion['wheel_position_kp']<=10 and 0<motion['wheel_position_kd']<=1,'Invalid wheel position gains')
    require('layers' not in motion,'Keyframe config must not contain a network')
    print('PASS: deterministic controller assets, unchanged hardware geometry, modes/gains, config and selected overlay')
    print('PENDING: Pi package-version parity, device/gamepad access, scheduling and physical motion. No hardware started.')

if __name__=='__main__':
    main()
