"""Resolve startup arguments without executing the launch or opening hardware."""
from pathlib import Path
import argparse
import runpy
import sys
from unittest.mock import Mock, patch

from controller_manager import spawner
from controller_manager.controller_manager_services import get_parameter_from_param_files
from launch import LaunchContext
from launch_ros.actions import Node
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions


def parse_controllers(args):
    """Exercise the installed spawner parser without creating a ROS node."""
    args = args[:args.index("--ros-args")] if "--ros-args" in args else args
    if hasattr(spawner, "parse_native_args"):
        return spawner.parse_native_args(args)[1]

    # Older Jazzy embeds its parser in main(). Stop immediately after parsing,
    # before any controller-manager service or other runtime action.
    class Parsed(Exception):
        pass

    parsed = []
    original = argparse.ArgumentParser.parse_args

    def capture(parser, *pos, **kwargs):
        parsed.append(original(parser, *pos, **kwargs))
        raise Parsed

    with patch.object(spawner.rclpy, "init"), \
            patch.object(spawner, "Node", side_effect=AssertionError("Must not create ROS nodes")), \
            patch.object(sys, "argv", ["spawner", *args]), \
            patch.object(argparse.ArgumentParser, "parse_args", capture):
        try:
            spawner.main()
        except Parsed:
            pass
    assert len(parsed) == 1
    result = parsed[0]
    return [{"name": name, "inactive": result.inactive,
             "param_files": result.param_file} for name in result.controller_names]


def assert_calibration_gate(args, expected):
    controllers = parse_controllers(args)
    assert len(controllers) == 1
    controller = controllers[0]
    assert controller["inactive"]
    assert get_parameter_from_param_files(
        Mock(), controller["name"], "/", controller["param_files"], "calibration_required"
    ) is expected


def test_calibration_gate_for_all_policy_spawners():
    launch_file = Path(__file__).parents[1] / "launch" / "launch.py"
    for sim, expected in (("False", True), ("True", False)):
        context = LaunchContext()
        context.launch_configurations["sim"] = sim
        description = runpy.run_path(str(launch_file))["generate_launch_description"]()
        count = 0
        for action in description.entities:
            if not isinstance(action, Node):
                continue
            if perform_substitutions(context, normalize_to_list_of_substitutions(action.node_executable)) != "spawner":
                continue
            args = [perform_substitutions(context, normalize_to_list_of_substitutions(word)) for word in action.cmd][1:]
            # Only policy spawners receive calibration overrides.
            if not args[0].startswith("neural_controller"):
                continue
            assert_calibration_gate(args, expected)
            count += 1
        assert count == 6


def test_alignment_trial_has_only_core_nodes_and_inactive_motion():
    context = LaunchContext()
    path = Path(__file__).parents[1] / "launch" / "alignment_trial.launch.py"
    description = runpy.run_path(str(path))["generate_launch_description"]()
    packages = []
    motion_spawners = 0
    for node in description.entities:
        packages.append(perform_substitutions(context, normalize_to_list_of_substitutions(node.node_package)))
        executable = perform_substitutions(context, normalize_to_list_of_substitutions(node.node_executable))
        if executable != "spawner":
            continue
        args = [perform_substitutions(context, normalize_to_list_of_substitutions(word)) for word in node.cmd][1:]
        if "neural_controller_wheel_align_hybrid" in args:
            assert_calibration_gate(args, True)
            motion_spawners += 1
    assert motion_spawners == 1
    assert set(packages) == {"robot_state_publisher", "controller_manager", "joy_linux", "joy_utils"}


def test_keyframe_trial_spawner_uses_calibration_and_starts_inactive():
    context = LaunchContext()
    path = Path(__file__).parents[1] / 'launch' / 'keyframe_trial.launch.py'
    description = runpy.run_path(str(path))['generate_launch_description']()
    names = []
    for node in description.entities:
        executable = perform_substitutions(context, normalize_to_list_of_substitutions(node.node_executable))
        if executable != 'spawner':
            continue
        args = [perform_substitutions(context, normalize_to_list_of_substitutions(word)) for word in node.cmd][1:]
        names.append(args[0])
        if args[0] == 'neural_controller_keyframe_align':
            assert_calibration_gate(args, True)
    assert names == ['joint_state_broadcaster', 'imu_sensor_broadcaster', 'neural_controller_keyframe_align']


def test_locomotion_trial_has_only_requested_policies_and_command_path():
    context = LaunchContext()
    path = Path(__file__).parents[1] / 'launch' / 'locomotion_trial.launch.py'
    description = runpy.run_path(str(path))['generate_launch_description']()
    names, executables = [], []
    for node in description.entities:
        executable = perform_substitutions(context, normalize_to_list_of_substitutions(node.node_executable))
        executables.append(executable)
        if executable != 'spawner':
            continue
        args = [perform_substitutions(context, normalize_to_list_of_substitutions(word)) for word in node.cmd][1:]
        names.append(args[0])
        if args[0].startswith('neural_controller'):
            assert_calibration_gate(args, True)
            if args[0] == 'neural_controller_wheel':
                controller = parse_controllers(args)[0]
                assert get_parameter_from_param_files(Mock(), controller['name'], '/',
                    controller['param_files'], 'cmd_vel_topic') == '/wheel_cmd_vel'
    assert names == ['joint_state_broadcaster', 'imu_sensor_broadcaster',
                     'neural_controller_walk_v2', 'neural_controller_wheel']
    assert sorted(executables) == sorted([
        'robot_state_publisher', 'ros2_control_node', 'joy_linux_node',
        'estop_controller', 'teleop_node', 'cmd_vel_mux_node', *(['spawner'] * 4)])


def test_inverted_triangle_trial_starts_only_triangle_inactive():
    import yaml
    from launch_ros.parameter_descriptions import ParameterFile
    context = LaunchContext()
    path = Path(__file__).parents[1] / 'launch' / 'inverted_triangle_trial.launch.py'
    description = runpy.run_path(str(path))['generate_launch_description']()
    names = []
    for node in description.entities:
        executable = perform_substitutions(context, normalize_to_list_of_substitutions(node.node_executable))
        if executable != 'spawner':
            continue
        args = [perform_substitutions(context, normalize_to_list_of_substitutions(word)) for word in node.cmd][1:]
        names.append(args[0])
        if args[0] == 'neural_controller_inverted_triangle':
            assert_calibration_gate(args, True)
            assert not any('inverted_triangle_config.yaml' in word for word in args)
    assert names == ['joint_state_broadcaster', 'imu_sensor_broadcaster', 'neural_controller_inverted_triangle']
    parameter_file = ParameterFile(path.with_name('inverted_triangle_config.yaml'), allow_substs=True)
    resolved = parameter_file.evaluate(context)
    params = yaml.safe_load(Path(resolved).read_text())['neural_controller_inverted_triangle']['ros__parameters']
    assert '$(' not in params['model_path']
    assert Path(params['model_path']).is_file()
