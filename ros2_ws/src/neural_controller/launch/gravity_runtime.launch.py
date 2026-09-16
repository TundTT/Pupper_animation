"""X roll, Triangle walk, Circle wheels, R2 lift+align, Square manual conversion. Startup homes hardware; confirm pose first."""
import os
from pathlib import Path
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Checked before any Node is created. The driver independently consumes it once.
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if os.environ.get('QUADMORPH_GRAVITY_CONFIRMED_BOOT') != boot:
        raise RuntimeError('Use scripts/start_robot.py: fresh operator confirmation required')
    def path(name):
        return PathJoinSubstitution([FindPackageShare('neural_controller'), 'launch', name])
    config = ParameterFile(path('config.yaml'), allow_substs=True)
    roll = ParameterFile(path('triangle_roll_config.yaml'), allow_substs=True)
    manual = ParameterFile(path('leg_to_wheel_config.yaml'), allow_substs=True)
    lift = ParameterFile(path('wheel_lift_config.yaml'), allow_substs=True)
    # Wheel->Walk/Lift "get ready" bridge; see wheel_to_walk_ready_config.yaml for why.
    ready = ParameterFile(path('wheel_to_walk_ready_config.yaml'), allow_substs=True)
    combined = ParameterFile(path('combined_motion.yaml'), allow_substs=True)
    policies = ['neural_controller_triangle_roll', 'neural_controller_walk_v2', 'neural_controller_wheel',
                'neural_controller_wheel_lift', 'neural_controller_wheel_to_walk_ready', 'neural_controller_leg_to_wheel']
    description = {'robot_description': ParameterValue(Command([
        FindExecutable(name='xacro'), ' ', PathJoinSubstitution([
            FindPackageShare('pupper_v3_description'), 'description', 'pupper_v3.urdf.xacro'])]), value_type=str)}
    nodes = [
        Node(package='robot_state_publisher', executable='robot_state_publisher', parameters=[description], output='both'),
        Node(package='controller_manager', executable='ros2_control_node', parameters=[config, roll, lift, manual, ready, combined], output='both'),
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
