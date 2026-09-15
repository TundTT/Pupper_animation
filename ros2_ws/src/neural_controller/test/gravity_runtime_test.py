"""Validate the installed description and refusal paths without creating hardware nodes."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import pytest
import yaml


def test_installed_description_uses_gravity_reference_and_continuous_hubs():
    package = Path(get_package_share_directory('pupper_v3_description'))
    expanded = subprocess.check_output(['xacro', str(package / 'description/pupper_v3.urdf.xacro')], text=True)
    root = ET.fromstring(expanded)
    reference = yaml.safe_load((package / 'config/gravity_pose.yaml').read_text())['joint_positions']
    joints = root.findall('./ros2_control/joint')
    assert len(joints) == 12
    addresses = set()
    for joint in joints:
        name = joint.attrib['name']
        params = {p.attrib['name']: p.text for p in joint.findall('param')}
        assert float(params['gravity_position']) == reference[name]
        assert not any(k.startswith('homing_') or k == 'post_homing_position' for k in params)
        addresses.add((int(params['can_channel']), int(params['can_id'])))
        assert {c.attrib['name'] for c in joint.findall('command_interface')} == {'position', 'velocity', 'effort', 'kp', 'kd'}
        physical = root.find(f"./joint[@name='{name}']")
        if name.endswith('_3'):
            assert physical.attrib['type'] == 'continuous'
            assert float(params['position_min']) == -1000.
            assert float(params['position_max']) == 1000.
    assert len(addresses) == 12
    for mesh in root.findall('.//mesh'):
        relative = mesh.attrib['filename'].removeprefix('package://pupper_v3_description/')
        asset = package / relative
        assert asset.is_file()
        assert not asset.read_bytes().startswith(b'version https://git-lfs')


def test_direct_launch_rejects_missing_or_wrong_boot_confirmation(monkeypatch):
    package = Path(get_package_share_directory('neural_controller'))
    spec = importlib.util.spec_from_file_location('runtime', package / 'launch/gravity_runtime.launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv('QUADMORPH_GRAVITY_CONFIRMED_BOOT', raising=False)
    with pytest.raises(RuntimeError, match='confirmation'):
        module.generate_launch_description()
    monkeypatch.setenv('QUADMORPH_GRAVITY_CONFIRMED_BOOT', 'previous-boot')
    with pytest.raises(RuntimeError, match='confirmation'):
        module.generate_launch_description()


def test_installed_packages_have_selected_bundle_marker():
    for name in ('neural_controller', 'robot_calibration', 'control_board_hardware_interface',
                 'cmd_vel_mux', 'pupper_v3_description'):
        share = Path(get_package_share_directory(name))
        assert (share / 'runtime_bundle.txt').read_text().strip() == 'quadmorph-stanford-gravity-v1'


def test_startup_wrapper_cannot_self_confirm_without_an_operator_terminal():
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run([sys.executable, str(root / 'scripts/start_robot.py')],
                            input='', capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert 'Interactive operator confirmation is required' in result.stderr
