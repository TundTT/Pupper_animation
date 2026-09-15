"""Shared calibration record; importing this module requires no ROS runtime."""
from .storage import JOINT_NAMES, WHEEL_NAMES, load_current

__all__ = ["JOINT_NAMES", "WHEEL_NAMES", "load_current"]
