"""Compatibility name for the selected gravity runtime."""
import importlib.util
from pathlib import Path

def generate_launch_description():
    spec = importlib.util.spec_from_file_location('gravity_runtime', Path(__file__).with_name('gravity_runtime.launch.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()
