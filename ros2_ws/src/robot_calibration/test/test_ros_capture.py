"""Exercise the CLI against a local fake manager and encoder publisher; no motor API."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers
from sensor_msgs.msg import JointState
from robot_calibration.storage import JOINT_NAMES, atomic_json, load_current, process_start


class RosCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[])

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"QUADMORPH_CALIBRATION_DIR": self.tmp.name})
        self.env.start()
        atomic_json(Path(self.tmp.name) / "encoder-session.json", {
            "schema_version": 1, "session_id": "ros-test-session", "ready": True,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "owner_pid": os.getpid(), "owner_start_ticks": process_start(os.getpid())})
        self.node = Node("calibration_test_fixture")
        self.moving = False
        self.active = False
        self.q = [1.0, 0.0, 0.3, -1.0, 0.0, -0.4, 1.0, 0.0, 0.5, -1.0, 0.0, -0.6]
        self.publisher = self.node.create_publisher(JointState, "/joint_states", 10)

        def publish():
            msg = JointState()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.name = list(reversed(JOINT_NAMES))
            msg.position = list(reversed(self.q))
            msg.velocity = [0.2 if self.moving else 0.0] * 12
            self.publisher.publish(msg)

        def controllers(request, response):
            broadcaster = ControllerState()
            broadcaster.name, broadcaster.state = "joint_state_broadcaster", "active"
            response.controller = [broadcaster]
            if self.active:
                motor = ControllerState()
                motor.name, motor.state = "fake_policy", "active"
                motor.claimed_interfaces = ["leg_front_r_1/position"]
                response.controller.append(motor)
            return response

        self.node.create_service(ListControllers, "/controller_manager/list_controllers", controllers)
        self.node.create_timer(0.02, publish)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin)
        self.thread.start()

    def tearDown(self):
        self.executor.shutdown()
        self.thread.join(timeout=5)
        self.node.destroy_node()
        self.env.stop()
        self.tmp.cleanup()

    def run_capture(self, *extra):
        return subprocess.run([sys.executable, "-m", "robot_calibration.cli", "capture",
                               "--operator-confirmed", "--timeout", "5", *extra],
                              text=True, capture_output=True, timeout=12)

    def test_capture_named_encoders_and_manual_values(self):
        result = self.run_capture()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = load_current()
        self.assertEqual(record["reference_joint_positions"], self.q)
        result = self.run_capture("--replace", "--wheel-home", "0.3", "-0.4", "0.5", "-0.6")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(load_current()["source"], "manual_encoder_values")

    def test_active_controller_prevents_capture(self):
        self.active = True
        result = self.run_capture()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fake_policy", result.stderr)
        self.assertFalse((Path(self.tmp.name) / "calibration.json").exists())

    def test_moving_encoders_prevent_capture(self):
        self.moving = True
        result = self.run_capture()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stationary", result.stderr)
        self.assertFalse((Path(self.tmp.name) / "calibration.json").exists())
