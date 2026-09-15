"""Leg-lift branch configuration adapted to the pinned wheeled robot."""
from pathlib import Path
import hashlib
import json
import numpy as np
from workspace.configs import *
from workspace.configs import get_config as original_config
PROXIMAL = np.array([0,1,3,4,6,7,9,10])
HUBS = np.array([2,5,8,11])
DEFAULT_MODEL_PATH = Path(__file__).parent / 'wheel_lift_model/model.xml'
DEFAULT_POSE = np.array([1.,0,0,-1,0,0,1,0,0,-1,0,0])
JOINT_LOWER_LIMITS = np.array([-1.12,-.32,-1000,-2.41,-3.04,-1000]*2)
JOINT_UPPER_LIMITS = np.array([2.41,3.04,1000,1.12,.32,1000]*2)
ACTION_SCALE = np.array([.5,1.6,0]*4)

def resolve_model_path(path=None):
    p = Path(path).resolve() if path else DEFAULT_MODEL_PATH.resolve()
    manifest = json.loads((DEFAULT_MODEL_PATH.parent/'provenance.json').read_text())
    if p != DEFAULT_MODEL_PATH.resolve():
        raise ValueError('This experiment requires its pinned wheel/backpack/9 mm model')
    for name, expected in manifest['files'].items():
        if hashlib.sha256((p.parent/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Model asset hash mismatch: '+name)
    return p

def get_config():
    c = original_config()
    c.action_scale = tuple(float(x) for x in ACTION_SCALE)
    # Keep the model's per-joint position PD gains in both MuJoCo and MJX.
    with c.ignore_type():
        c.position_control_kp = (5.,5.,4.)*4
        c.dof_damping = (.25,.25,.15)*4
    c.reward_config.target_lift_height = .02
    c.reward_config.lift_height_mode = "ramp"
    c.reward_config.lift_height_sigma = .015
    c.reward_config.scales.unload = 0.
    c.reward_config.scales.wheel_separation = 0.
    c.reward_floor = 0.
    # The old straight-foot hip-angle proxy is not valid for a wheel.
    c.reward_config.scales.lift_progress = 0.
    c.command_hold_steps_min = 150
    c.command_hold_steps_max = 400
    return c
