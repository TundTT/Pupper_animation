"""Resolve startup arguments without executing the launch or opening hardware."""
from pathlib import Path
import runpy

from controller_manager.spawner import parse_native_args
from launch import LaunchContext
from launch_ros.actions import Node
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions


def test_calibration_gate_for_all_policy_spawners():
    launch_file = Path(__file__).parents[1] / "launch" / "launch.py"
    for sim, expected in (("False", "True"), ("True", "False")):
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
            if "--controller-ros-args=-p" not in args:
                continue
            args = args[:args.index("--ros-args")] if "--ros-args" in args else args
            _, controllers = parse_native_args(args)
            assert len(controllers) == 1
            assert controllers[0]["inactive"]
            assert controllers[0]["controller_ros_args"] == ["-p", "calibration_required:=" + expected]
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
            assert "--inactive" in args
            assert "calibration_required:=True" in args
            motion_spawners += 1
    assert motion_spawners == 1
    assert set(packages) == {"robot_state_publisher", "controller_manager", "joy_linux", "joy_utils"}
