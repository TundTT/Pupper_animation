"""X roll, Triangle walk, Circle wheels, R2 lift+align. Startup homes hardware; confirm pose first."""
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    def path(name):
        return PathJoinSubstitution([FindPackageShare('neural_controller'), 'launch', name])
    config = ParameterFile(path('config.yaml'), allow_substs=True)
    roll = ParameterFile(path('triangle_roll_config.yaml'), allow_substs=True)
    # notebook_lift_align_config.yaml already carries the controller_manager type entry,
    # alignment_mode=quadmorph-notebook-lift-align-v1, and the full parameter set;
    # nothing further is needed in combined_motion.yaml for this controller.
    lift = ParameterFile(path('notebook_lift_align_config.yaml'), allow_substs=True)
    # Wheel->Walk/Lift "get ready" bridge; see wheel_to_walk_ready_config.yaml for why.
    ready = ParameterFile(path('wheel_to_walk_ready_config.yaml'), allow_substs=True)
    combined = ParameterFile(path('combined_motion.yaml'), allow_substs=True)
    policies = ['neural_controller_triangle_roll', 'neural_controller_walk_v2', 'neural_controller_wheel',
                'neural_controller_notebook_lift', 'neural_controller_wheel_to_walk_ready']
    description = {'robot_description': Command([
        FindExecutable(name='xacro'), ' ', PathJoinSubstitution([
            FindPackageShare('pupper_v3_description'), 'description', 'pupper_v3.urdf.xacro'])])}
    nodes = [
        Node(package='robot_state_publisher', executable='robot_state_publisher', parameters=[description], output='both'),
        Node(package='controller_manager', executable='ros2_control_node', parameters=[config, roll, lift, ready, combined], output='both'),
        Node(package='joy_linux', executable='joy_linux_node', name='joy_linux_node', parameters=[config, {'autorepeat_rate': 20.}], output='both'),
        # A single dispatcher owns buttons; legacy X alignment/estop-release bindings are absent.
        Node(package='neural_controller', executable='motion_buttons.py', output='both'),
        Node(package='teleop_twist_joy', executable='teleop_node', name='teleop_twist_joy_node',
             parameters=[config, {'scale_linear.x': .20, 'scale_linear.y': .10, 'scale_angular.yaw': .40}],
             remappings=[('cmd_vel', 'teleop_cmd_vel')], output='both'),
        Node(package='cmd_vel_mux', executable='cmd_vel_mux_node', parameters=[config, {
            'inputs': ['/teleop_cmd_vel'], 'timeout_ms': 500,
            'wheel_output_topic': '/wheel_cmd_vel', 'wheel_forward_multiplier': 2.}], output='both'),
    ]
    for name in ['joint_state_broadcaster', 'imu_sensor_broadcaster', *policies]:
        args = [name, '--controller-manager', '/controller_manager', '--controller-manager-timeout', '60']
        if name in policies:
            args += ['--inactive', '--param-file', path('locomotion_wheel_trial.yaml' if name == policies[2]
                                                     else 'calibration_hardware.yaml')]
        nodes.append(Node(package='controller_manager', executable='spawner', arguments=args, output='both'))
    return LaunchDescription(nodes)
