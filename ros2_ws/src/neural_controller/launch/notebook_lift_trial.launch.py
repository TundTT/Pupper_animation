"""Explicit lift-only trial; hardware startup requires STARTUP_UPPER_HOME.md."""
from launch import LaunchDescription
from launch.substitutions import Command,FindExecutable,PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    root=FindPackageShare('neural_controller')
    base=ParameterFile(PathJoinSubstitution([root,'launch','config.yaml']),allow_substs=True)
    candidate=ParameterFile(PathJoinSubstitution([root,'launch','notebook_lift_config.yaml']),allow_substs=True)
    name='neural_controller_notebook_lift'
    description={'robot_description':Command([FindExecutable(name='xacro'),' ',PathJoinSubstitution([FindPackageShare('pupper_v3_description'),'description','pupper_v3.urdf.xacro'])])}
    nodes=[Node(package='robot_state_publisher',executable='robot_state_publisher',parameters=[description],output='both'),
           Node(package='controller_manager',executable='ros2_control_node',parameters=[base,candidate],output='both'),
           Node(package='joy_linux',executable='joy_linux_node',name='joy_linux_node',parameters=[base],output='both'),
           Node(package='joy_utils',executable='estop_controller',name='joy_util_node',parameters=[base,{
               'calibration_required':True,'default_controller_name':name,'controller_names':[name],
               'switch_button_indices':[-1],'leg_lift_button_index':-1,
               'wheel_align_hybrid_button_index':0,'wheel_align_hybrid_controller_name':name,
               'wheel_align_hybrid_command_topic':'/notebook_lift_command_index',
               'wheel_align_hybrid_command_states':['stand','front_l','front_r','back_r','back_l'],
               'wheel_align_hybrid_cycle_states':['front_r','stand','front_l','stand','back_r','stand','back_l','stand'],
           }],output='both')]
    for controller in ('joint_state_broadcaster','imu_sensor_broadcaster',name):
        args=[controller,'--controller-manager','/controller_manager','--controller-manager-timeout','60']
        # Model-path substitutions are resolved in the manager's ParameterFile above.
        # Only the calibration override is passed raw to the spawner.
        if controller==name:args+=['--inactive','--param-file',PathJoinSubstitution([root,'launch','calibration_hardware.yaml'])]
        nodes.append(Node(package='controller_manager',executable='spawner',arguments=args,output='both'))
    return LaunchDescription(nodes)
