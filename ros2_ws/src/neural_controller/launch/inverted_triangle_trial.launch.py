"""Hardware startup homes encoders. Obtain physical confirmation before launch."""
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    def path(name):
        return PathJoinSubstitution([FindPackageShare('neural_controller'), 'launch', name])
    common = ParameterFile(path('config.yaml'), allow_substs=True)
    triangle = ParameterFile(path('inverted_triangle_config.yaml'), allow_substs=True)
    description = {'robot_description': Command([
        FindExecutable(name='xacro'), ' ', PathJoinSubstitution([
            FindPackageShare('pupper_v3_description'), 'description', 'pupper_v3.urdf.xacro'])])}
    nodes = [
        Node(package='robot_state_publisher', executable='robot_state_publisher', parameters=[description], output='both'),
        Node(package='controller_manager', executable='ros2_control_node', parameters=[common, triangle], output='both'),
        Node(package='joy_linux', executable='joy_linux_node', name='joy_linux_node',
             parameters=[common, {'autorepeat_rate': 20.0}], output='both'),
        # Only stop is bound. No face button or stop-release can enable a policy.
        Node(package='joy_utils', executable='estop_controller', name='joy_util_node',
             parameters=[common, {'calibration_required': True, 'estop_release_index': -1,
                'controller_names': ['neural_controller_inverted_triangle'], 'switch_button_indices': [-1],
                'default_controller_name': 'neural_controller_inverted_triangle',
                'leg_lift_button_index': -1, 'wheel_align_hybrid_button_index': -1}], output='both'),
    ]
    for name in ['joint_state_broadcaster', 'imu_sensor_broadcaster', 'neural_controller_inverted_triangle']:
        args = [name, '--controller-manager', '/controller_manager', '--controller-manager-timeout', '60']
        if name == 'neural_controller_inverted_triangle':
            # Manager receives the substitution-resolved triangle ParameterFile.
            # Passing its raw YAML here would override model_path with literal $(...).
            args += ['--inactive', '--param-file', path('calibration_hardware.yaml')]
        nodes.append(Node(package='controller_manager', executable='spawner', arguments=args, output='both'))
    return LaunchDescription(nodes)
