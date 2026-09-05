"""Pupper training tools; interactive rendering on Windows, EGL on Linux."""
import os
import sys

if sys.platform != 'win32':
    os.environ.setdefault('MUJOCO_GL', 'egl')

# Keep geometry/preflight tools usable without importing the full training stack.
def __getattr__(name):
    if name == 'PupperLegLiftEnv':
        from workspace.leg_lift_env import PupperLegLiftEnv
        return PupperLegLiftEnv
    raise AttributeError(name)
