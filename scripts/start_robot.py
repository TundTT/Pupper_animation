#!/usr/bin/env python3
"""Confirm physical setup for one hardware activation, then run the installed stack."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    if not sys.stdin.isatty():
        raise RuntimeError('Interactive operator confirmation is required for this startup')
    # Inspect ownership before asking the operator to reposition anything.
    import fcntl
    with open('/tmp/control_board_hardware.lock', 'a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A hardware owner is already running; do not start another stack')
    from ament_index_python.packages import get_package_share_directory
    package = Path(get_package_share_directory('neural_controller'))
    if not (package / 'launch/gravity_runtime.launch.py').is_file():
        raise RuntimeError('Source this branch\'s install/setup.bash before startup')
    for name in ('neural_controller', 'robot_calibration', 'control_board_hardware_interface',
                 'pupper_v3_description', 'cmd_vel_mux'):
        share = Path(get_package_share_directory(name))
        # Every installed package carries the same bundle marker; prevents mixed overlays.
        if (share / 'runtime_bundle.txt').read_text().strip() != 'quadmorph-stanford-gravity-v1':
            raise RuntimeError('Wrong package overlay: ' + str(share))
        print(name + ': ' + str(share))
    print('Support the robot. Let the legs hang freely and position all marked wheel rings')
    print('at the agreed calibration references. Keep the robot stationary.')
    if input('Type READY only when the physical setup is complete: ').strip() != 'READY':
        raise RuntimeError('Cancelled; hardware was not started')
    env = os.environ.copy()
    env['QUADMORPH_GRAVITY_CONFIRMED_BOOT'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    print('After startup, leave the pose unchanged. In another sourced terminal, run:')
    print('  python3 scripts/calibrate_robot.py capture')
    print('  python3 scripts/calibrate_robot.py status')
    # No detached service or automatic restart can reuse this confirmation.
    return subprocess.call(['ros2', 'launch', 'neural_controller', 'gravity_runtime.launch.py'], env=env)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ImportError, EOFError) as error:
        print('STARTUP BLOCKED: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
