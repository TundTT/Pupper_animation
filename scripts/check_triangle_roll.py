#!/usr/bin/env python3
"""Read-only roll artifact and current robot-code hardware interface checks."""
import argparse,hashlib,json,math,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
import yaml
from check_align_v5 import check as check_geometry
ROOT=Path(__file__).resolve().parents[1]
NAME='neural_controller_triangle_roll'

def require(ok,why):
    if not ok:raise ValueError(why)

def check(install=None,controller_install=None):
    controller=ROOT/'ros2_ws/src/neural_controller';description=ROOT/'ros2_ws/src/pupper_v3_description'
    if install:
        for package in ['robot_calibration','control_board_hardware_interface','neural_controller','joy_utils','pupper_v3_description']:
            prefix=Path(subprocess.check_output(['ros2','pkg','prefix',package],text=True).strip()).resolve()
            expected=controller_install if package=='neural_controller' and controller_install else install
            require(prefix.is_relative_to(expected.resolve()),'Wrong overlay: '+package)
            if package=='neural_controller':
                for name in ['triangle_roll_plan.json','triangle_roll_config.yaml','triangle_roll_trial.launch.py','triangle_roll_stand.launch.py','triangle_roll_stand.yaml','calibration_hardware.yaml']:
                    require((prefix/'share'/package/'launch'/name).read_bytes().replace(b'\r\n',b'\n')==(controller/'launch'/name).read_bytes().replace(b'\r\n',b'\n'),'Stale installed '+name)
                require((prefix/'lib/libneural_controller.so').is_file(),'Missing controller library')
                for name in ['measured_roll.hpp','triangle_roll.hpp','triangle_roll_contract.hpp']:
                    require((prefix/'include/neural_controller'/name).read_bytes().replace(b'\r\n',b'\n')==(controller/'include/neural_controller'/name).read_bytes().replace(b'\r\n',b'\n'),'Stale installed '+name)
            if package=='pupper_v3_description':description=prefix/'share'/package
    check_geometry(controller,description)
    manifest=json.loads((ROOT/'hardware_testing/triangle_roll/export_provenance.json').read_text())
    data=json.loads((controller/'launch/triangle_roll_plan.json').read_text())
    raw=(controller/'launch/triangle_roll_plan.json').read_bytes().replace(b'\r\n',b'\n')
    require(hashlib.sha256(raw).hexdigest()==manifest['export_sha256'],'Unreviewed roll plan')
    header=(controller/'include/neural_controller/triangle_roll_geometry.hpp').read_bytes().replace(b'\r\n',b'\n')
    require(hashlib.sha256(header).hexdigest()==data['geometry_sha256'],'Changed nominal kinematics')
    contract=(controller/'include/neural_controller/triangle_roll_contract.hpp').read_text().split('R"CONTRACT(',1)[1].split(')CONTRACT"',1)[0]
    require(json.loads(contract)==data,'Compiled contract differs from plan')
    require(data['schema_version']==3 and data['source_run']=='360ffcd09344487d' and not data['walking_enabled'],'Wrong source/task')
    require(data['axial_gap_m']==.009 and data['hz']==520,'Gap/rate mismatch')
    cfg=yaml.safe_load((controller/'launch/triangle_roll_config.yaml').read_text());c=cfg[NAME]['ros__parameters']
    require(cfg['controller_manager']['ros__parameters'][NAME]['type']=='neural_controller/TriangleRollController','Wrong plugin')
    require(c['joint_names']==data['joint_names'] and c['action_types']==['position']*12,'Wrong order/modes')
    require(c['kps']==data['kp'] and c['kds']==data['kd'],'Wrong PD gains')
    require(all(abs(a-b)<1e-11 for a,b in zip(c['default_joint_pos'],data['initial'])),'Wrong point-up winding')
    require(c['use_imu'] and c['repeat_action']==1 and c['gain_multiplier']==1 and c['estop_kd']==0 and abs(c['max_body_angle']-math.radians(8))<1e-12,'Timing/sensor/stop configuration mismatch')
    joints={j.attrib['name']:j for j in ET.parse(description/'description/components.xacro').findall('.//joint')}
    for i,name in enumerate(data['joint_names']):
        hw={p.attrib['name']:float(p.text) for p in joints[name].findall('param')}
        require('hard_limit_min' not in hw and 'hard_limit_max' not in hw,'Unexpected hardware gain override')
        require(data['kp'][i]<=hw['kp_max'] and 2*data['kd'][i]<=hw['kd_max'],'Damping schedule would be clamped')
        require(c['joint_lower_limits'][i]==hw['position_min'] and c['joint_upper_limits'][i]==hw['position_max'],'Position limit mismatch')
        require(hw['position_min']<=min(data['initial'][i],data['goal'][i]-.2) and max(data['initial'][i],data['goal'][i]+.2)<=hw['position_max'],'Roll envelope exceeds joint limits')
    print('PASS: pinned simultaneous roll, 9 mm gap, nominal kinematics, current CAN/axes/continuous hubs, doubled damping, stop and position limits. No hardware activation.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install-base',type=Path);p.add_argument('--controller-install-base',type=Path)
    a=p.parse_args();check(a.install_base,a.controller_install_base)
