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
    args = args[:args.index("--ros-args")] if "--ros-args" in args else args
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
