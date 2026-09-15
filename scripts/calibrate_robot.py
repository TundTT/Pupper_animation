#!/usr/bin/env python3
"""Repository entry point; source ROS Jazzy and the local overlay before capture."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws/src/robot_calibration"))
from robot_calibration.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
