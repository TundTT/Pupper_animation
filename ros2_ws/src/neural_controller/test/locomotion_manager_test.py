"""Real ROS manager + GenericSystem only. Never loads the robot hardware plugin."""
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import yaml
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Empty, Float32MultiArray
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import Twist
from controller_manager_msgs.srv import LoadController, ConfigureController, SwitchController
from ament_index_python.packages import get_package_prefix
from robot_calibration.storage import JOINT_NAMES, atomic_json, process_start, make_record


def test_gap_policies_switch_and_run_in_real_manager():
    root=Path(__file__).parents[1]
    config=yaml.safe_load((root/'launch/config.yaml').read_text())
    policies = ['neural_controller_walk_v2', 'neural_controller_wheel']
    policy_config = {}
    for name, filename in zip(policies, ['policy_walk_v2.json', 'policy_wheel.json']):
        params = config[name]['ros__parameters']
        params['model_path'] = str(root / 'launch' / filename)
        params['calibration_required'] = True
        if name == policies[1]:
            params.update(yaml.safe_load((root / 'launch/locomotion_wheel_trial.yaml').read_text())[name]['ros__parameters'])
        policy_config[name] = {'ros__parameters': params}
    q=[1.,0.,-1.,-1.,0.,1.,1.,0.,-1.,-1.,0.,1.]
    fields=['position','velocity','effort','kp','kd']
    # This hardcoded mock plugin is the only hardware implementation in this test.
    xml=['<robot name="keyframe_fixture"><link name="base"/>']
    for j in JOINT_NAMES:
        xml.append(f'<link name="{j}_link"/><joint name="{j}" type="continuous"><parent link="base"/><child link="{j}_link"/><axis xyz="0 0 1"/><limit effort="10" velocity="10"/></joint>')
    xml.append('<ros2_control name="fake" type="system"><hardware><plugin>mock_components/GenericSystem</plugin></hardware>')
    for j,pos in zip(JOINT_NAMES,q):
        xml.append(f'<joint name="{j}">')
        for field in fields:
            xml.append(f'<command_interface name="{field}"/><state_interface name="{field}"><param name="initial_value">{pos if field=="position" else 0}</param></state_interface>')
        xml.append('</joint>')
    xml.append('<sensor name="imu_sensor">')
    for field in ['orientation.x','orientation.y','orientation.z','orientation.w','angular_velocity.x','angular_velocity.y','angular_velocity.z','time_since_measurement_seconds']:
        xml.append(f'<state_interface name="{field}"><param name="initial_value">{1 if field=="orientation.w" else 0}</param></state_interface>')
    xml.append('</sensor></ros2_control></robot>')
    urdf=''.join(xml)
    assert 'control_board_hardware_interface' not in urdf
    with tempfile.TemporaryDirectory(prefix='locomotion-manager-fixture-') as directory:
        work=Path(directory)
        atomic_json(work/'encoder-session.json',dict(schema_version=1,session_id='fake-manager-session',ready=True,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),owner_pid=os.getpid(),owner_start_ticks=process_start(os.getpid())))
        atomic_json(work/'calibration.json',make_record('fake-manager-session',q))
        cfg = {'controller_manager': {'ros__parameters': {
            'update_rate': 520,
            **{name: {'type': 'neural_controller/NeuralController'} for name in policies},
            'joint_state_broadcaster': {'type': 'joint_state_broadcaster/JointStateBroadcaster'}}},
            **policy_config}
        (work/'config.yaml').write_text(yaml.safe_dump(cfg))
        env=dict(os.environ,QUADMORPH_CALIBRATION_DIR=directory)
        rclpy.init()
        node=rclpy.create_node('locomotion_manager_fixture')
        latest={}
        direction = [1.]
        walking_pub = node.create_publisher(Twist, '/cmd_vel', 10)
        wheel_pub = node.create_publisher(Twist, '/wheel_cmd_vel', 10)
        def publish_commands():
            walking, wheel = Twist(), Twist()
            walking.linear.x = .2 * direction[0]
            wheel.linear.x = .4 * direction[0]
            walking_pub.publish(walking)
            wheel_pub.publish(wheel)
        node.create_timer(.05, publish_commands)
        for policy in policies:
            node.create_subscription(Float32MultiArray, f'/{policy}/policy_output',
                                     lambda m, policy=policy: latest.update({policy: list(m.data)}), 10)
            node.create_subscription(Float32MultiArray, f'/{policy}/observation',
                                     lambda m, policy=policy: latest.update({policy + '_obs': list(m.data)}), 10)
        node.create_subscription(DynamicJointState,'/dynamic_joint_states',lambda m:latest.update(joints=m),10)
        description=node.create_publisher(String,'/robot_description',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        description.publish(String(data=urdf))
        stop=node.create_publisher(Empty,'/emergency_stop',10)
        log=(work/'manager.log').open('w')
        exe=Path(get_package_prefix('controller_manager'))/'lib/controller_manager/ros2_control_node'
        process=subprocess.Popen([str(exe),'--ros-args','--params-file',str(work/'config.yaml')],env=env,stdout=log,stderr=subprocess.STDOUT)
        def wait(predicate,seconds=12):
            deadline=time.monotonic()+seconds
            while time.monotonic()<deadline:
                rclpy.spin_once(node,timeout_sec=.02)
                if predicate():return
                if process.poll() is not None:raise AssertionError('Mock manager exited')
            raise AssertionError('Timed out waiting for mock manager condition')
        def call(kind,service,request):
            client=node.create_client(kind,'/controller_manager/'+service)
            assert client.wait_for_service(timeout_sec=12),service
            future=client.call_async(request)
            wait(future.done)
            assert future.result().ok,(service,future.result())
            node.destroy_client(client)
        def switch(activate,deactivate):
            req=SwitchController.Request();req.activate_controllers=activate;req.deactivate_controllers=deactivate;req.strictness=2
            call(SwitchController,'switch_controller',req)
        def measured_commands():
            msg = latest.get('joints')
            if msg is None:
                return None
            values = []
            for joint in JOINT_NAMES:
                if joint not in msg.joint_names:
                    return None
                row = msg.interface_values[msg.joint_names.index(joint)]
                values.append(dict(zip(row.interface_names, row.values)))
            return values
        try:
            for controller in ['joint_state_broadcaster', *policies]:
                call(LoadController,'load_controller',LoadController.Request(name=controller))
                call(ConfigureController,'configure_controller',ConfigureController.Request(name=controller))
            switch(['joint_state_broadcaster'], [])
            previous = []
            for name in [*policies, policies[0]]:
                latest.pop(name, None)
                switch([name], previous)
                wait(lambda: len(latest.get(name, [])) == 12, seconds=10)
                assert all(math.isfinite(v) for v in latest[name])
                wait(lambda: measured_commands() is not None)
                rows = measured_commands()
                assert all(math.isfinite(row[field]) for row in rows for field in fields)
                wheel = name == policies[1]
                wait(lambda: all(abs(row['kp'] - (0. if wheel and i % 3 == 2 else 5.)) < 1e-6
                                 for i, row in enumerate(measured_commands())))
                for sign in (1., -1.):
                    direction[0] = sign
                    expected = sign * (.4 if wheel else .2)
                    wait(lambda: len(latest.get(name + '_obs', [])) >= 9 and
                         abs(latest[name + '_obs'][6] - expected) < 1e-6)
                previous = [name]
            stop.publish(Empty())
            wait(lambda: all(row['kp'] == 0. and row['effort'] == 0.
                             for row in measured_commands()))
            switch([], previous)
        except Exception:
            log.flush();print((work/'manager.log').read_text()[-9000:]);raise
        finally:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close();node.destroy_node();rclpy.shutdown()
