"""Gap-policy hardware trial. Obtain physical confirmation before encoder homing.

Triangle = walking, Circle = wheels. Both spawn inactive and require calibration.
"""
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = ParameterFile(PathJoinSubstitution([
        FindPackageShare("neural_controller"), "launch", "config.yaml"
    ]), allow_substs=True)
    policies = ["neural_controller_walk_v2", "neural_controller_wheel"]
    description = {"robot_description": Command([
        FindExecutable(name="xacro"), " ", PathJoinSubstitution([
            FindPackageShare("pupper_v3_description"), "description", "pupper_v3.urdf.xacro"
        ])
    ])}
    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[description], output="both"),
        Node(package="controller_manager", executable="ros2_control_node",
             parameters=[config], output="both"),
        Node(package="joy_linux", executable="joy_linux_node", name="joy_linux_node",
             parameters=[config], output="both"),
        Node(package="joy_utils", executable="estop_controller", name="joy_util_node",
             parameters=[config, {
                 "calibration_required": True,
                 "default_controller_name": policies[0],
                 "controller_names": policies,
                 "switch_button_indices": [2, 1],
                 "leg_lift_button_index": -1,
                 "wheel_align_hybrid_button_index": -1,
             }], output="both"),
        Node(package="teleop_twist_joy", executable="teleop_node",
             name="teleop_twist_joy_node", parameters=[config, {
                 "scale_linear.x": 0.20, "scale_linear.y": 0.10,
                 "scale_angular.yaw": 0.40,
             }], remappings=[("cmd_vel", "teleop_cmd_vel")], output="both"),
        Node(package="cmd_vel_mux", executable="cmd_vel_mux_node",
             parameters=[config, {"inputs": ["/teleop_cmd_vel"], "timeout_ms": 500,
                                  "wheel_output_topic": "/wheel_cmd_vel",
                                  "wheel_forward_multiplier": 2.0}],
             output="both"),
    ]
    for name in ["joint_state_broadcaster", "imu_sensor_broadcaster", *policies]:
        args = [name, "--controller-manager", "/controller_manager",
                "--controller-manager-timeout", "60"]
        if name in policies:
            args += ["--inactive", "--param-file", PathJoinSubstitution([
                FindPackageShare("neural_controller"), "launch",
                "locomotion_wheel_trial.yaml" if name == "neural_controller_wheel"
                else "calibration_hardware.yaml"
            ])]
        nodes.append(Node(package="controller_manager", executable="spawner",
                          arguments=args, output="both"))
    return LaunchDescription(nodes)
