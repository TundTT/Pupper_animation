#!/usr/bin/env python3

import rclpy
import serial
from rclpy.node import Node
from sensor_msgs.msg import Joy

# Serial command letters, one per D-pad direction. Must match the pin mapping in
# robot/arduino/dpad_pin_driver/dpad_pin_driver.ino.
DIRECTIONS = ("U", "R", "D", "L")


class DpadSerialBridgeNode(Node):
    def __init__(self):
        super().__init__("dpad_serial_bridge_node")

        # This Nano uses a CH340 USB-serial chip, which enumerates as /dev/ttyUSB0, not
        # /dev/ttyACM0 (confirmed via lsusb/dmesg on the physical robot 2026-09-04).
        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("baud_rate", 115200)
        # "axes" (two hat-style axes, values -1/0/1) or "buttons" (four discrete
        # buttons[] indices). Verify which this hardware uses via `ros2 topic echo
        # /joy` before trusting the defaults below -- see D_pad.md Step 0.
        self.declare_parameter("dpad_mode", "axes")

        # Used when dpad_mode == "axes".
        self.declare_parameter("dpad_axis_horizontal", 6)  # -1 = left, +1 = right
        self.declare_parameter("dpad_axis_vertical", 7)  # -1 = down, +1 = up
        self.declare_parameter("dpad_axis_horizontal_sign", 1)
        self.declare_parameter("dpad_axis_vertical_sign", 1)

        # Used when dpad_mode == "buttons".
        self.declare_parameter("dpad_button_up", -1)
        self.declare_parameter("dpad_button_right", -1)
        self.declare_parameter("dpad_button_down", -1)
        self.declare_parameter("dpad_button_left", -1)

        self.declare_parameter("reconnect_period_sec", 2.0)
        self.declare_parameter("reset_settle_sec", 2.0)

        self.serial_port_name = self.get_parameter("serial_port").get_parameter_value().string_value
        self.baud_rate = self.get_parameter("baud_rate").get_parameter_value().integer_value
        self.dpad_mode = self.get_parameter("dpad_mode").get_parameter_value().string_value

        self.axis_h = self.get_parameter("dpad_axis_horizontal").get_parameter_value().integer_value
        self.axis_v = self.get_parameter("dpad_axis_vertical").get_parameter_value().integer_value
        self.axis_h_sign = self.get_parameter("dpad_axis_horizontal_sign").get_parameter_value().integer_value
        self.axis_v_sign = self.get_parameter("dpad_axis_vertical_sign").get_parameter_value().integer_value

        self.button_up = self.get_parameter("dpad_button_up").get_parameter_value().integer_value
        self.button_right = self.get_parameter("dpad_button_right").get_parameter_value().integer_value
        self.button_down = self.get_parameter("dpad_button_down").get_parameter_value().integer_value
        self.button_left = self.get_parameter("dpad_button_left").get_parameter_value().integer_value

        self.reconnect_period_sec = self.get_parameter("reconnect_period_sec").get_parameter_value().double_value
        self.reset_settle_sec = self.get_parameter("reset_settle_sec").get_parameter_value().double_value

        # Last state sent to the Nano for each direction, keyed by DIRECTIONS letter.
        self.last_sent_state = {d: False for d in DIRECTIONS}
        # Current state as read from /joy, used to re-sync all directions after a reconnect.
        self.current_state = {d: False for d in DIRECTIONS}

        self.serial_conn = None
        self.resync_timer = None

        self.try_open_serial()
        self.reconnect_timer = self.create_timer(self.reconnect_period_sec, self.try_open_serial)

        self.joy_subscription = self.create_subscription(Joy, "/joy", self.joy_callback, 10)

        self.get_logger().info("D-pad serial bridge node initialized")
        self.get_logger().info(f"Serial port: {self.serial_port_name} @ {self.baud_rate} baud")
        self.get_logger().info(f"D-pad mode: {self.dpad_mode}")

    def try_open_serial(self):
        """(Re)open the serial connection to the Nano. Never raises -- the rest of the
        robot must keep working even if the Arduino is unplugged."""
        if self.serial_conn is not None and self.serial_conn.is_open:
            return

        try:
            self.serial_conn = serial.Serial(self.serial_port_name, self.baud_rate, timeout=0)
            self.get_logger().info(f"Opened serial port {self.serial_port_name}")
            # Nano resets on DTR toggle when the port opens; give it time to boot
            # before sending anything, then re-sync all four directions.
            if self.resync_timer is not None:
                self.resync_timer.cancel()
            self.resync_timer = self.create_timer(self.reset_settle_sec, self.resync_after_open)
        except serial.SerialException as e:
            if self.serial_conn is not None:
                self.get_logger().warn(f"Serial port {self.serial_port_name} unavailable: {e}")
            self.serial_conn = None

    def resync_after_open(self):
        """One-shot (self-cancels): re-send current state of all directions after the
        Nano finishes resetting."""
        self.resync_timer.cancel()
        self.resync_timer = None
        if self.serial_conn is None or not self.serial_conn.is_open:
            return
        self.get_logger().info("Re-syncing D-pad state to Nano after (re)connect")
        for direction in DIRECTIONS:
            self.send_state(direction, self.current_state[direction], force=True)

    def send_state(self, direction, pressed, force=False):
        if not force and self.last_sent_state[direction] == pressed:
            return
        self.last_sent_state[direction] = pressed
        if self.serial_conn is None or not self.serial_conn.is_open:
            return
        line = f"{direction}{1 if pressed else 0}\n"
        try:
            self.serial_conn.write(line.encode("ascii"))
        except serial.SerialException as e:
            self.get_logger().warn(f"Lost serial connection while writing: {e}")
            self.close_serial()

    def close_serial(self):
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except serial.SerialException:
                pass
        self.serial_conn = None

    def read_dpad_axes(self, msg):
        if len(msg.axes) <= max(self.axis_h, self.axis_v):
            return None
        h = msg.axes[self.axis_h] * self.axis_h_sign
        v = msg.axes[self.axis_v] * self.axis_v_sign
        return {
            "U": v > 0.5,
            "D": v < -0.5,
            "R": h > 0.5,
            "L": h < -0.5,
        }

    def read_dpad_buttons(self, msg):
        indices = {"U": self.button_up, "R": self.button_right, "D": self.button_down, "L": self.button_left}
        if min(indices.values()) < 0:
            self.get_logger().error(
                "dpad_mode is 'buttons' but one or more dpad_button_* params is unset (-1); "
                "set them per Step 0's findings",
                throttle_duration_sec=5.0,
            )
            return None
        if len(msg.buttons) <= max(indices.values()):
            return None
        return {d: msg.buttons[i] == 1 for d, i in indices.items()}

    def joy_callback(self, msg):
        if self.dpad_mode == "axes":
            state = self.read_dpad_axes(msg)
        elif self.dpad_mode == "buttons":
            state = self.read_dpad_buttons(msg)
        else:
            self.get_logger().error(f"Unknown dpad_mode '{self.dpad_mode}', expected 'axes' or 'buttons'")
            return

        if state is None:
            return

        for direction in DIRECTIONS:
            self.current_state[direction] = state[direction]
            self.send_state(direction, state[direction])

    def destroy_node(self):
        self.close_serial()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = DpadSerialBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
