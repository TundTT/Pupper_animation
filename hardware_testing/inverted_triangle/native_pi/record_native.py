"""Record the prepared native overlay; no hardware activation."""
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import xml.etree.ElementTree as ET

root = Path('/home/pi/robot-code-leglift')
install = root/'ros2_ws/install-triangle'
def command(*args):
    return subprocess.check_output(args, text=True).strip()
def sha(path, normalized=False):
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b'\r\n', b'\n') if normalized else raw).hexdigest()
record = {'base_commit': command('git', '-C', str(root), 'rev-parse', 'HEAD'),
          'source_state': 'base commit plus reviewed PS-stop correction; hashes below',
          'platform': platform.platform(), 'python': sys.version,
          'compiler': command('g++', '-dumpfullversion'),
          'cmake': command('cmake', '--version').splitlines()[0],
          'install_base': str(install), 'packages': {}, 'binaries': {},
          'source_hash_convention': 'SHA256 with CRLF normalized to LF',
          'source_files': {}, 'rtprio': resource.getrlimit(resource.RLIMIT_RTPRIO),
          'backup': '/home/pi/robot-code-leglift-backup-before-triangle-ac3d824',
          'motor_stack_started': False, 'physical_motion_tested': False}
for package in ('controller_manager', 'generate_parameter_library', 'joy_linux',
                'robot_calibration', 'control_board_hardware_interface',
                'neural_controller', 'joy_utils', 'pupper_v3_description'):
    prefix = Path(command('ros2', 'pkg', 'prefix', package))
    record['packages'][package] = {'prefix': str(prefix),
        'version': ET.parse(prefix/'share'/package/'package.xml').findtext('version')}
for relative in ('neural_controller/lib/libneural_controller.so',
                 'control_board_hardware_interface/lib/libcontrol_board_hardware_interface.so',
                 'joy_utils/lib/joy_utils/estop_controller'):
    path = install/relative
    linkage = command('ldd', str(path))
    assert 'not found' not in linkage, linkage
    record['binaries'][relative] = {'sha256': sha(path), 'missing_dependencies': False}
prior = json.loads((root/'hardware_testing/inverted_triangle/preparation_manifest.json').read_text())
for relative in prior['source_files']:
    record['source_files'][relative] = sha(root/relative, True)
for relative in ('ros2_ws/src/neural_controller/launch/config.yaml',
                 'ros2_ws/src/pupper_v3_description/description/components.xacro',
                 'ros2_ws/src/control_board_hardware_interface/src/control_board_hardware_interface.cpp',
                 'hardware_testing/start_pose/start_pose.json'):
    record['source_files'][relative] = sha(root/relative, True)
record['gamepad'] = json.loads(Path('/home/pi/triangle-gamepad-check.json').read_text())
record['joy_stream'] = json.loads(Path('/home/pi/triangle-joy-stream.json').read_text())
Path('/home/pi/triangle-native-manifest.json').write_text(json.dumps(record, indent=2)+'\n')
print('Saved native overlay manifest, linked binaries and source hashes.')
