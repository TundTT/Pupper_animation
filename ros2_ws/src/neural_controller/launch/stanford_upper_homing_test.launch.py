"""Explicit homing-only diagnostic. Does not start policies or validate hubs."""
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def patch(xml, scope):
    if scope not in ('fr1', 'upper'):
        raise ValueError('Expected fr1 or upper')
    root = ET.fromstring(xml)
    hw = root.find('./ros2_control/hardware')
    if hw is None:
        raise ValueError('Missing hardware')
    if any(p.attrib['name'].startswith('manual_reference') for p in root.iter('param')):
        raise ValueError('Cannot mix manual and active homing')
    ET.SubElement(hw, 'param', name='stanford_upper_homing_test').text = scope
    return ET.tostring(root, encoding='unicode')


def generate_launch_description():
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch_ros.actions import Node
    from launch_ros.parameter_descriptions import ParameterFile
    from launch.substitutions import TextSubstitution
    repo = Path('/home/pi/robot-code-leglift')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if os.environ.get('QUADMORPH_STANFORD_SUPPORTED_BOOT') != boot:
        raise ValueError('Supported-robot confirmation required for this boot')
    prefix = repo/'ros2_ws/install-stanford-homing/control_board_hardware_interface'
    if Path(get_package_share_directory('control_board_hardware_interface')).parents[1] != prefix:
        raise ValueError('Source install-stanford-homing last')
    scope = os.environ.get('QUADMORPH_STANFORD_SCOPE', 'fr1')
    desc = Path(get_package_share_directory('pupper_v3_description'))/'description/pupper_v3.urdf.xacro'
    xml = patch(subprocess.check_output(['xacro', str(desc)], text=True), scope)
    config = ParameterFile(TextSubstitution(text=str(Path(get_package_share_directory('neural_controller'))/'launch/config.yaml')), allow_substs=True)
    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': xml}], output='both'),
        Node(package='controller_manager', executable='ros2_control_node',
             parameters=[config], output='both'),
    ])
