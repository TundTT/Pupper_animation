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
OBSERVATION_SIZE = 83
MOTION_VERSION = 5
MOTION_ID = 'quadmorph-align-motion-v5'

# Runtime reference stages: support shift, lift, supported landing, recenter.
SHIFT_SECONDS = 3.5
RISE_SECONDS = 3.
LIFT_SECONDS = SHIFT_SECONDS + RISE_SECONDS
LAND_SECONDS = 6.
RECENTER_SECONDS = 4.
LOWER_SECONDS = LAND_SECONDS + RECENTER_SECONDS
LAND_HIP = .50
APEX_HIP = .95  # legacy diagnostic only; runtime uses per-wheel poses below
ACTIVE_RESIDUAL = np.array([.02, .02])
SUPPORT_RESIDUAL = np.array([.05, .075])
ACTIVE_SPEED = np.array([.30, .45])
SUPPORT_SPEED = np.array([.35, .45])
ACTIVE_ACCEL = np.array([1.5, 2.])
SUPPORT_ACCEL = np.array([1.5, 2.])
LEG_TIMEOUT_STEPS = 48 * 52
SEQUENCE_STEPS = 320 * 52
SINGLE_STEPS = 64 * 52
RECOVERY_SECONDS = 2.
RECOVERY_FLOOR = .012
RECOVERY_WHEEL = .012
RECOVERY_BODY = .007
# Nominal joint targets fitted in the current model; see ALIGN_MOTION_V4.md.
APEX_POSES = np.array([[1.16890458, 1.00215022, -1.26259869, 0.28754656, 1.01678593, -0.16507621, -0.60319068, -0.19620607], [1.26259869, -0.28754656, -1.16890458, -1.00215022, 0.60319068, 0.19620607, -1.01678593, 0.16507621], [1.0, 0.0, -1.0, 0.0, 1.0, 1.05, -1.0, 0.0], [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, -1.05]])
