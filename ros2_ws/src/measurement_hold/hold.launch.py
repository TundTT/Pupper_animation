"""Measurement hold only. Requires operator-confirmed supported current pose.

Normal policy calibration is NOT created or bypassed. The startup assigns encoder
offsets in the user-arranged measurement pose; that session is not a walking frame.
"""
from launch import LaunchDescription
from launch.substitutions import Command,FindExecutable,PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile

def generate_launch_description():
    config=ParameterFile(PathJoinSubstitution([FindPackageShare('neural_controller'),'launch','config.yaml']),allow_substs=True)
    description={'robot_description':Command([FindExecutable(name='xacro'),' ',PathJoinSubstitution([FindPackageShare('pupper_v3_description'),'description','pupper_v3.urdf.xacro'])])}
    nodes=[Node(package='robot_state_publisher',executable='robot_state_publisher',parameters=[description],output='both'),
        Node(package='controller_manager',executable='ros2_control_node',parameters=[config,{'measurement_hold.type':'measurement_hold/CurrentPose'}],output='both'),
        Node(package='joy_linux',executable='joy_linux_node',name='joy_linux_node',parameters=[config,{'autorepeat_rate':20.}],output='both')]
    for name in ('joint_state_broadcaster','imu_sensor_broadcaster','measurement_hold'):
        args=[name,'--controller-manager','/controller_manager','--controller-manager-timeout','60']
        if name=='measurement_hold':args+=['--inactive']
        nodes.append(Node(package='controller_manager',executable='spawner',arguments=args,output='both'))
    return LaunchDescription(nodes)
