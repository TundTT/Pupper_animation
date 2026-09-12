"""Real ROS manager + GenericSystem only. Never loads the robot hardware plugin."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import yaml
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Int32, Empty, Float64MultiArray
from control_msgs.msg import DynamicJointState
from controller_manager_msgs.srv import LoadController, ConfigureController, SwitchController
from ament_index_python.packages import get_package_prefix
from robot_calibration.storage import JOINT_NAMES, atomic_json, process_start, make_record


def test_real_manager_activation_and_fault_outputs():
    root=Path(__file__).parents[1]
    config=yaml.safe_load((root/'launch/config.yaml').read_text())
    name='neural_controller_keyframe_align'
    params=config[name]['ros__parameters']
    params['model_path']=str(root/'launch/keyframe_config.json')
    params['calibration_required']=True
    q=[1.,0.,.3,-1.,0.,-.4,1.,0.,.5,-1.,0.,-.6]
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
    with tempfile.TemporaryDirectory(prefix='keyframe-manager-fixture-') as directory:
        work=Path(directory)
        atomic_json(work/'encoder-session.json',dict(schema_version=1,session_id='fake-manager-session',ready=True,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),owner_pid=os.getpid(),owner_start_ticks=process_start(os.getpid())))
        atomic_json(work/'calibration.json',make_record('fake-manager-session',q))
        # Supply only the two needed controller configurations.
        cfg={'controller_manager':{'ros__parameters':{'update_rate':520,
            name:{'type':'neural_controller/KeyframeController'},
            'joint_state_broadcaster':{'type':'joint_state_broadcaster/JointStateBroadcaster'}}},
            name:{'ros__parameters':params}}
        (work/'config.yaml').write_text(yaml.safe_dump(cfg))
        env=dict(os.environ,QUADMORPH_CALIBRATION_DIR=directory)
        rclpy.init()
        node=rclpy.create_node('keyframe_manager_fixture')
        latest={}
        node.create_subscription(Float64MultiArray,f'/{name}/alignment_status',lambda m:latest.update(status=list(m.data)),10)
        node.create_subscription(Float64MultiArray,f'/{name}/motor_commands',lambda m:latest.update(commands=list(m.data)),10)
        node.create_subscription(DynamicJointState,'/dynamic_joint_states',lambda m:latest.update(joints=m),10)
        description=node.create_publisher(String,'/robot_description',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        description.publish(String(data=urdf))
        command=node.create_publisher(Int32,'/keyframe_align_command_index',10)
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
        def zero_outputs():
            if latest.get('commands')!=[0.]*60:return False
            msg=latest.get('joints')
            if msg is None:return False
            for j in JOINT_NAMES:
                if j not in msg.joint_names:return False
                values=msg.interface_values[msg.joint_names.index(j)]
                data=dict(zip(values.interface_names,values.values))
                if any(data.get(f)!=0 for f in ('effort','kp','kd')):return False
            return True
        try:
            for controller in ['joint_state_broadcaster',name]:
                call(LoadController,'load_controller',LoadController.Request(name=controller))
                call(ConfigureController,'configure_controller',ConfigureController.Request(name=controller))
            switch(['joint_state_broadcaster',name],[])
            wait(lambda:len(latest.get('status',[]))==28 and latest['status'][23]==1)
            assert latest['status'][25]==0,'Real manager first update must not fault'
            command.publish(Int32(data=9))
            wait(lambda:latest['status'][25]==7 and latest['status'][12]==7 and zero_outputs())
            command.publish(Int32(data=1))
            for _ in range(15):rclpy.spin_once(node,timeout_sec=.02)
            assert latest['status'][25]==7 and zero_outputs(),'Fault must remain latched with zero gains'
            switch([],[name]);wait(zero_outputs)
            latest.pop('status',None)
            switch([name],[])
            wait(lambda:len(latest.get('status',[]))==28 and latest['status'][23]==1 and latest['status'][25]==0)
            stop.publish(Empty())
            wait(lambda:latest['status'][25]==8 and zero_outputs())
            switch([],[name]);wait(zero_outputs)
        except Exception:
            log.flush();print((work/'manager.log').read_text()[-9000:]);raise
        finally:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close();node.destroy_node();rclpy.shutdown()
