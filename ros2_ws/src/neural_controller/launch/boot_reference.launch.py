"""Boot-bound, passive startup using manually measured references. No motion controllers."""
import json,math,statistics,xml.etree.ElementTree as ET
from pathlib import Path

def patch(xml,refs,snapshot,boot):
 if refs['boot_id']!=boot or snapshot['boot_id']!=boot:raise ValueError('Wrong boot')
 root=ET.fromstring(xml);joints=root.findall('./ros2_control/joint')
 if {x.attrib['name'] for x in joints}!=set(refs['joints']):raise ValueError('Incomplete reference set')
 rows=snapshot['rows'][-50:]
 if len(rows)!=50 or any(b['all_zero_packet'] for r in rows for b in r['boards']):raise ValueError('Incomplete feedback')
 expected={}
 for j in joints:
  name=j.attrib['name'];p={x.attrib['name']:x.text for x in j.findall('param')};c=int(p['can_channel'])-1;k=int(p['can_id'])-1
  values=[-r['boards'][c//2]['q'][2*k+c%2] for r in rows];ref=refs['joints'][name];offset=ref['offset']
  if not all(math.isfinite(v) for v in values+[offset]):raise ValueError('Nonfinite reference/data')
  if max(values)-min(values)>.002:raise ValueError('Joint moving: '+name)
  q=statistics.median(values);expected[name]=q-offset
  if name.endswith('_3') and abs(q-ref['raw_reference'])>.03:raise ValueError('Hub moved since operator reference: '+name)
  for key,value in [('manual_reference_offset',offset),('manual_reference_expected_raw',q)]:
   ET.SubElement(j,'param',name=key).text=str(value)
 hw=root.find('./ros2_control/hardware');ET.SubElement(hw,'param',name='manual_reference_boot_id').text=boot
 return ET.tostring(root,encoding='unicode'),expected

def generate_launch_description():
 import os,subprocess,time
 from ament_index_python.packages import get_package_share_directory
 from launch import LaunchDescription
 from launch_ros.actions import Node
 from launch_ros.parameter_descriptions import ParameterFile
 from launch.substitutions import TextSubstitution
 repo=Path('/home/pi/robot-code-leglift');here=Path(__file__).resolve().parent
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 if os.environ.get('QUADMORPH_REFERENCE_START_CONFIRMED_BOOT')!=boot:raise ValueError('Operator supported-startup confirmation required')
 expected_prefix=repo/'ros2_ws/install-manual-reference/control_board_hardware_interface'
 if Path(get_package_share_directory('control_board_hardware_interface')).parents[1]!=expected_prefix:raise ValueError('Source install-manual-reference last')
 snapshot=here/('startup_raw_'+str(time.time_ns())+'.json')
 subprocess.run(['python3',str(repo/'scripts/read_disabled_spi.py'),'--output',str(snapshot)],check=True,capture_output=True)
 desc=Path(get_package_share_directory('pupper_v3_description'))/'description/pupper_v3.urdf.xacro'
 xml=subprocess.check_output(['xacro',str(desc)],text=True)
 xml,expected=patch(xml,json.loads((here/'boot-references.json').read_text()),json.loads(snapshot.read_text()),boot)
 (here/'expected-startup.json').write_text(json.dumps(expected,indent=2))
 config=ParameterFile(TextSubstitution(text=str(Path(get_package_share_directory('neural_controller'))/'launch/config.yaml')),allow_substs=True)
 return LaunchDescription([Node(package='robot_state_publisher',executable='robot_state_publisher',parameters=[{'robot_description':xml}],output='both'),Node(package='controller_manager',executable='ros2_control_node',parameters=[config],output='both'),*[Node(package='controller_manager',executable='spawner',arguments=[n,'--controller-manager-timeout','60'],output='both') for n in ['joint_state_broadcaster','imu_sensor_broadcaster']]])
