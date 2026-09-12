"""Minimal keyframe trial. Launch homes hardware: follow STARTUP_CALIBRATION.md first."""
import importlib.util
from pathlib import Path

def generate_launch_description():
    # Keep the reviewed hardware/broadcaster startup identical to alignment_trial.
    spec=importlib.util.spec_from_file_location('alignment_trial',Path(__file__).with_name('alignment_trial.launch.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # The common factory accepts only an explicit controller name; calibration
    # remains required and the controller is always spawned inactive.
    return module.make_trial('neural_controller_keyframe_align','/keyframe_align_command_index')
