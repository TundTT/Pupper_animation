"""Minimal hardware alignment trial; launching performs encoder homing.

Follow STARTUP_CALIBRATION.md before launch. All motion starts inactive.
"""
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    controller = "neural_controller_wheel_align_hybrid"
    config = ParameterFile(PathJoinSubstitution([
        FindPackageShare("neural_controller"), "launch", "config.yaml"
    ]), allow_substs=True)
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
                 "default_controller_name": controller,
                 "controller_names": [controller],
                 "switch_button_indices": [-1],
                 "leg_lift_button_index": -1,
             }], output="both"),
    ]
    for name in ("joint_state_broadcaster", "imu_sensor_broadcaster", controller):
        args = [name, "--controller-manager", "/controller_manager",
                "--controller-manager-timeout", "60"]
        if name == controller:
            args += ["--inactive", "--param-file", PathJoinSubstitution([
                FindPackageShare("neural_controller"), "launch", "calibration_hardware.yaml"
            ])]
        nodes.append(Node(package="controller_manager", executable="spawner",
                          arguments=args, output="both"))
    return LaunchDescription(nodes)
