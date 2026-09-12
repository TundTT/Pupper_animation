"""Actual trial button and stick nodes, fake manager, no hardware or robot launch."""
import os
from pathlib import Path
import runpy
import subprocess
import threading
import time

import pytest
import yaml
import rclpy
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import Twist
from launch import LaunchContext
from launch_ros.utilities import evaluate_parameters
from sensor_msgs.msg import Joy
from std_msgs.msg import Empty, Int32


@pytest.mark.parametrize('calibrated', [False, True])
def test_trial_buttons_and_sticks(tmp_path, calibrated):
    # Evaluate the actual installed launch parameters; never execute its hardware nodes.
    share = Path(get_package_share_directory('neural_controller'))
    launch = runpy.run_path(str(share / 'launch/locomotion_trial.launch.py'))
    nodes = launch['generate_launch_description']().entities
    context = LaunchContext()
    configs = []
    for node in nodes[3:6]:  # joystick dispatcher, teleop, command mux
        settings = {}
        for item in evaluate_parameters(context, node._Node__parameters):
            if isinstance(item, dict):
                settings.update(item)
            else:
                # config.yaml is scoped by the node's name.
                all_settings = yaml.safe_load(Path(item).read_text())
                name = ['joy_util_node', 'teleop_twist_joy_node', 'cmd_vel_mux'][len(configs)]
                settings.update(all_settings[name]['ros__parameters'])
        configs.append(settings)
    assert configs[0]['calibration_required'] is True
    # True case bypasses only this child process gate to emulate an existing calibration.
    # False case uses a missing isolated state folder and must reject all activation.
    configs[0]['calibration_required'] = not calibrated
    env = {**os.environ, 'ROS_DOMAIN_ID': '94',
           'QUADMORPH_CALIBRATION_DIR': str(tmp_path / 'calibration')}
    ctx = rclpy.context.Context()
    rclpy.init(context=ctx, domain_id=94)
    node = rclpy.create_node('fake_locomotion_manager', context=ctx)
    requests, stops, leg_commands, velocities, wheel_velocities = [], [], [], [], []

    def switch(request, response):
        requests.append((list(request.activate_controllers), list(request.deactivate_controllers)))
        response.ok = True
        return response

    node.create_service(SwitchController, '/controller_manager/switch_controller', switch)
    node.create_subscription(Empty, '/emergency_stop', lambda m: stops.append(True), 10)
    node.create_subscription(Int32, '/leg_lift_command_index', lambda m: leg_commands.append(m.data), 10)
    node.create_subscription(Twist, '/cmd_vel', lambda m: velocities.append(m), 10)
    node.create_subscription(Twist, '/wheel_cmd_vel', lambda m: wheel_velocities.append(m), 10)
    pub = node.create_publisher(Joy, '/joy', 10)
    executor = MultiThreadedExecutor(context=ctx)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    processes, logs = [], []

    def wait(predicate):
        deadline = time.monotonic() + 8
        while not predicate():
            assert all(p.poll() is None for p in processes)
            assert time.monotonic() < deadline, (pub.get_subscription_count(), len(velocities), [p.read_text() for p in tmp_path.glob('*.log')])
            time.sleep(.03)

    def send(buttons=(), axes=None):
        msg = Joy()
        msg.buttons = [int(i in buttons) for i in range(13)]
        msg.axes = axes or [0.] * 8
        pub.publish(msg)
        time.sleep(.15)

    def press(*buttons):
        send(buttons)
        send()

    try:
        for i, (package, executable, name) in enumerate([
            ('joy_utils', 'estop_controller', 'joy_util_node'),
            ('teleop_twist_joy', 'teleop_node', 'teleop_twist_joy_node'),
            ('cmd_vel_mux', 'cmd_vel_mux_node', 'cmd_vel_mux'),
        ]):
            path = tmp_path / f'{name}.yaml'
            path.write_text(yaml.safe_dump({'/**': {'ros__parameters': configs[i]}}))
            log = (tmp_path / f'{name}.log').open('w')
            logs.append(log)
            exe = Path(get_package_prefix(package)) / 'lib' / package / executable
            args = [str(exe), '--ros-args', '--params-file', str(path), '-r', f'__node:={name}']
            if i == 1:
                args += ['-r', 'cmd_vel:=teleop_cmd_vel']
            processes.append(subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT))
        wait(lambda: pub.get_subscription_count() == 2 and node.count_publishers('/cmd_vel') == 1)
        press(2)
        press(1)
        if calibrated:
            wait(lambda: len(requests) == 2)
            assert requests == [(['neural_controller_walk_v2'], ['neural_controller_wheel']),
                                (['neural_controller_wheel'], ['neural_controller_walk_v2'])]
            send((1,))
            send((1,))  # A held button cannot generate repeated switches.
            send()
            wait(lambda: len(requests) == 3)
        else:
            assert requests == []
        assert leg_commands == []  # Circle never enters the legacy leg cycle.
        count = len(requests)
        press(0, 3, 6)  # X, Square and L2 unbound for this trial.
        assert len(requests) == count
        press(configs[0]['estop_index'], 2, 1)
        wait(lambda: stops and len(requests) == count + 1)
        assert requests[-1][0] == []
        assert set(requests[-1][1]) == {'neural_controller_walk_v2', 'neural_controller_wheel'}
        press(configs[0]['estop_release_index'])
        if calibrated:
            wait(lambda: len(requests) == count + 2)
            assert requests[-1][0] == ['neural_controller_wheel']
        else:
            assert len(requests) == count + 1
        # Both streams are produced continuously; controller selection cannot
        # briefly apply the wheel scale to walking during a switch.
        for sign in (1., -1.):
            velocities.clear()
            wheel_velocities.clear()
            send(axes=[1., sign, 0., 1., 0., 0., 0., 0.])
            wait(lambda: any(abs(v.linear.x - sign * .2) < 1e-6 and abs(v.linear.y - .1) < 1e-6
                             and abs(v.angular.z - .4) < 1e-6 for v in velocities))
            wait(lambda: any(abs(v.linear.x - sign * .4) < 1e-6 and abs(v.angular.z - .4) < 1e-6
                             for v in wheel_velocities))
        time.sleep(.7)  # No /joy packets: timeout zeros BOTH command streams.
        for stream in (velocities, wheel_velocities):
            assert stream[-1].linear.x == stream[-1].linear.y == stream[-1].angular.z == 0.
    finally:
        for process in processes:
            process.terminate()
            process.wait(timeout=5)
        for log in logs:
            log.close()
        executor.shutdown()
        thread.join(timeout=5)
        node.destroy_node()
        rclpy.shutdown(context=ctx)


def test_duplicate_circle_binding_is_rejected(tmp_path):
    config = tmp_path / 'conflict.yaml'
    config.write_text(yaml.safe_dump({'/**': {'ros__parameters': {
        'controller_names': ['neural_controller_wheel'],
        'switch_button_indices': [1], 'leg_lift_button_index': 1,
    }}}))
    exe = Path(get_package_prefix('joy_utils')) / 'lib/joy_utils/estop_controller'
    result = subprocess.run([str(exe), '--ros-args', '--params-file', str(config)],
                            env={**os.environ, 'ROS_DOMAIN_ID': '94'},
                            capture_output=True, text=True, timeout=8)
    assert result.returncode != 0
    assert 'Conflicting joystick binding at button 1' in result.stderr
