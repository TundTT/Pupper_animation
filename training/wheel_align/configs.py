"""Alignment-only profile. Geometry is pinned to align-hybrid bfd74cc; limits
match robot-code 582fd88 components.xacro. No heater or morphing actuation."""
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).with_name('model.xml')
LEGS = ('front_r', 'front_l', 'back_r', 'back_l')
JOINT_NAMES = tuple(f'leg_{leg}_{j}' for leg in LEGS for j in (1, 2, 3))
POSITION_ACTUATOR_ROWS = [0, 1, 3, 4, 6, 7, 9, 10]
WHEEL_ACTUATOR_ROWS = [2, 5, 8, 11]
WHEEL_COLLISION_GEOM_NAMES = [f'leg_{leg}_3_wheel_collision' for leg in LEGS]
WHEEL_DIAMETER_JITTER = .005
DEFAULT_POSE = np.array([1., 0., 0., -1., 0., 0.] * 2)
COMMAND_STATES = ['stand', 'front_l', 'front_r', 'back_r', 'back_l']
CONTROL_DT = 10 / 520
PHYSICS_DT = 1 / 520
OBSERVATION_SIZE = 82
MOTION_VERSION = 2

# Fixed runtime contract: change C++ and parity tests together.
LIFT_SECONDS = 3.
LOWER_SECONDS = 4.
APEX_HIP = .85
ACTIVE_RESIDUAL = np.array([.06, .04])
SUPPORT_RESIDUAL = np.array([.20, .30])
ACTIVE_SPEED = np.array([.4, .7])
SUPPORT_SPEED = np.array([2., 3.])
ACTIVE_ACCEL = np.array([2., 2.])
SUPPORT_ACCEL = np.array([12., 16.])
