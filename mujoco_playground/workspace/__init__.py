"""Pupper V3 leg-lift RL training pipeline (built on the playground / brax MJX stack)."""

import os

# MUST run before anything imports `mujoco` (leg_lift_env does, one line below).
# MuJoCo picks its OpenGL backend at import time and caches it, so setting this in
# train.py is TOO LATE: `python -m workspace.train` executes this package __init__
# first, mujoco binds to GLFW/X11, and rollout rendering then aborts the whole
# process on a headless machine (a C++ abort, so no try/except can catch it).
# setdefault, so a caller with a display can still force e.g. MUJOCO_GL=glfw.
os.environ.setdefault("MUJOCO_GL", "egl")

from workspace.leg_lift_env import PupperLegLiftEnv  # noqa: E402

__all__ = ["PupperLegLiftEnv"]
