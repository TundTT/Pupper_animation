"""Saved upper-home startup with state broadcasters only; no motion policies."""
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    description = {'robot_description': Command([
        FindExecutable(name='xacro'), ' ', PathJoinSubstitution([
            FindPackageShare('pupper_v3_description'), 'description', 'pupper_v3.urdf.xacro'])])}
    config = ParameterFile(PathJoinSubstitution([
        FindPackageShare('neural_controller'), 'launch', 'config.yaml']), allow_substs=True)
    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[description], output='both'),
        Node(package='controller_manager', executable='ros2_control_node',
             parameters=[config], output='both'),
        *[Node(package='controller_manager', executable='spawner',
               arguments=[name, '--controller-manager-timeout', '60'], output='both')
          for name in ('joint_state_broadcaster', 'imu_sensor_broadcaster')],
    ])
