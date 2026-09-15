#!/usr/bin/env python3
"""Snapshot the robot's current joint positions and hold them with kp/kd position control.

No calibration or homing step is performed by this script: it just reads whatever
/joint_states is currently reporting (the control board hardware interface applies its
own boot-time homing offset before this topic ever sees the data, so the pose read here
is already sane) and commands the forward_position/kp/kd controllers to hold exactly
that pose.

Requires the normal ros2_control stack (hardware or sim) to already be running.

Usage:
    ros2 run <this as a plain script> scripts/hold_current_pose.py
    python3 scripts/hold_current_pose.py --kp 5.0 --kd 0.25
"""
import argparse
import signal
import time

import rclpy
from controller_manager_msgs.srv import SwitchController
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

JOINT_NAMES = [
    "leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
    "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
    "leg_back_r_1", "leg_back_r_2", "leg_back_r_3",
    "leg_back_l_1", "leg_back_l_2", "leg_back_l_3",
]

# Every controller that can claim the same position/kp/kd command interfaces as the
# forward_* controllers below (kept in sync with controller_manager's list in config.yaml).
MOTION_CONTROLLERS = [
    "neural_controller",
    "neural_controller_three_legged",
    "neural_controller_leg_lift",
    "neural_controller_wheel_align_hybrid",
    "neural_controller_keyframe_align",
    "neural_controller_wheel",
    "neural_controller_walk_v2",
]
FORWARD_CONTROLLERS = ["forward_position_controller", "forward_kp_controller", "forward_kd_controller"]


class HoldCurrentPose(Node):
    def __init__(self, kp, kd, ramp_time, rate, max_error, max_velocity):
        super().__init__("hold_current_pose")
        self.kp_target = kp
        self.kd_target = kd
        self.ramp_time = ramp_time
        self.max_error = max_error
        self.max_velocity = max_velocity

        self.target = None  # captured hold pose: joint name -> position (rad)
        self.q = {}
        self.v = {}
        self.last_states_time = 0.0
        self.gain_scale = 0.0
        self.start_time = None
        self.releasing = False
        self.stop = False

        self.position_pub = self.create_publisher(Float64MultiArray, "/forward_position_controller/commands", 10)
        self.kp_pub = self.create_publisher(Float64MultiArray, "/forward_kp_controller/commands", 10)
        self.kd_pub = self.create_publisher(Float64MultiArray, "/forward_kd_controller/commands", 10)
        self.create_subscription(JointState, "/joint_states", self._joint_states_cb, 10)
        self.switch_client = self.create_client(SwitchController, "/controller_manager/switch_controller")

        self.timer = self.create_timer(1.0 / rate, self._control_loop)

    def _joint_states_cb(self, msg):
        if len(msg.name) != len(msg.position):
            return
        self.q = dict(zip(msg.name, msg.position))
        self.v = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        self.last_states_time = time.monotonic()

    def wait_for_pose(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if all(n in self.q for n in JOINT_NAMES) and time.monotonic() - self.last_states_time < 0.2:
                self.target = {n: self.q[n] for n in JOINT_NAMES}
                return True
        return False

    def activate_forward_controllers(self, timeout=5.0):
        if not self.switch_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("controller_manager not available (/controller_manager/switch_controller)")
        request = SwitchController.Request()
        request.activate_controllers = FORWARD_CONTROLLERS
        request.deactivate_controllers = MOTION_CONTROLLERS
        request.strictness = SwitchController.Request.BEST_EFFORT
        request.activate_asap = True
        request.timeout = rclpy.duration.Duration(seconds=timeout).to_msg()
        future = self.switch_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None or not future.result().ok:
            raise RuntimeError("Failed to activate forward_position/kp/kd controllers")

    def begin_release(self):
        self.releasing = True
        self.release_start = time.monotonic()
        self.release_from = self.gain_scale

    def _publish(self, positions, kp, kd):
        pos_msg = Float64MultiArray()
        pos_msg.data = positions
        kp_msg = Float64MultiArray()
        kp_msg.data = kp
        kd_msg = Float64MultiArray()
        kd_msg.data = kd
        self.position_pub.publish(pos_msg)
        self.kp_pub.publish(kp_msg)
        self.kd_pub.publish(kd_msg)

    def _control_loop(self):
        if self.target is None:
            return
        now = time.monotonic()
        if self.start_time is None:
            self.start_time = now

        if self.releasing:
            elapsed = now - self.release_start
            alpha = min(elapsed / max(self.ramp_time, 1e-3), 1.0)
            self.gain_scale = self.release_from * (1.0 - alpha)
            if alpha >= 1.0:
                self.stop = True
        else:
            elapsed = now - self.start_time
            alpha = min(elapsed / max(self.ramp_time, 1e-3), 1.0)
            self.gain_scale = alpha * alpha * (3 - 2 * alpha)  # cubic smoothstep ramp-in

            # Safety trip: measured pose drifted too far from the held target, or a
            # joint is moving too fast (e.g. the robot got bumped) -> release rather
            # than fight it indefinitely.
            if self.q and now - self.last_states_time < 0.5:
                error = max(abs(self.target[n] - self.q.get(n, self.target[n])) for n in JOINT_NAMES)
                velocity = max((abs(self.v.get(n, 0.0)) for n in JOINT_NAMES), default=0.0)
                if error > self.max_error or velocity > self.max_velocity:
                    self.get_logger().error(
                        f"Hold safety trip (error={error:.3f} rad, velocity={velocity:.3f} rad/s) -- releasing"
                    )
                    self.begin_release()

        kp = [self.kp_target * self.gain_scale] * len(JOINT_NAMES)
        kd = [self.kd_target * self.gain_scale] * len(JOINT_NAMES)
        positions = [self.target[n] for n in JOINT_NAMES]
        self._publish(positions, kp, kd)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kp", type=float, default=5.0, help="Hold position gain (default 5.0)")
    parser.add_argument("--kd", type=float, default=0.25, help="Hold damping gain (default 0.25)")
    parser.add_argument("--ramp-time", type=float, default=2.0, help="Seconds to ramp gains in/out (default 2.0)")
    parser.add_argument("--rate", type=float, default=100.0, help="Command publish rate in Hz (default 100)")
    parser.add_argument("--max-error", type=float, default=0.35,
                         help="Max allowed position error in rad before releasing (default 0.35)")
    parser.add_argument("--max-velocity", type=float, default=2.0,
                         help="Max allowed joint velocity in rad/s before releasing (default 2.0)")
    args = parser.parse_args()

    rclpy.init()
    node = HoldCurrentPose(args.kp, args.kd, args.ramp_time, args.rate, args.max_error, args.max_velocity)

    def handle_signal(signum, frame):
        if node.target is not None and not node.releasing:
            node.get_logger().info("Stop requested -- releasing gains smoothly")
            node.begin_release()
        else:
            node.stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        node.get_logger().info("Waiting for /joint_states...")
        if not node.wait_for_pose():
            raise RuntimeError("No fresh /joint_states received -- is the robot hardware/sim stack running?")
        node.get_logger().info(f"Captured pose: {node.target}")

        node.get_logger().info("Activating forward_position/kp/kd controllers...")
        node.activate_forward_controllers()

        node.get_logger().info(
            f"Holding current pose at kp={args.kp} kd={args.kd} (ramping over {args.ramp_time}s). "
            "Ctrl+C to release and exit."
        )
        while rclpy.ok() and not node.stop:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
