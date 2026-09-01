#!/usr/bin/env python3
"""Keyboard teleop for testing the wheeled policy without a physical joystick.

Unlike spacebar_joy_sim.py (a single edge-triggered tap for the O-button leg-lift
cycle), the wheel controller is velocity-commanded and needs a sustained signal, not
a tap. To sidestep X11 key-autorepeat ambiguity (repeated KeyPress/KeyRelease while a
key is held look identical to independent taps without XkbSetDetectableAutoRepeat),
this uses "set and hold" semantics instead of true press-and-hold: each key sets a
persistent axis value that is republished continuously (at 20 Hz, matching
teleop_twist_joy's require_enable_button:false expectation of a steady stream) until
a different key changes it. Space zeroes everything.

Grabs keys globally on the X server (DISPLAY, default :1), same trick as
spacebar_joy_sim.py, so it works regardless of window focus -- including while the
MuJoCo viewer window is focused.

Keys:
  T       tap Triangle (button 2) -- activate neural_controller_wheel
  X       tap X (button 0) -- back to locomotion, also zeroes drive axes
  W / S   forward / backward (axis 1, teleop_twist_joy's axis_linear.x)
  A / D   turn left / right (axis 3, teleop_twist_joy's axis_angular.yaw)
  Q / E   strafe left / right (axis 0, teleop_twist_joy's axis_linear.y)
  Space   stop (zero all drive axes, does not change controller/e-stop)

Requires: pip install python-xlib
Run inside the same ROS2 env / sourced workspace as the launch you're testing.
"""
import os
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from Xlib import X, XK, display

NUM_BUTTONS = 16
NUM_AXES = 8
TRIANGLE_BUTTON_INDEX = 2  # matches config.yaml's switch_button_indices entry for the wheel controller
X_BUTTON_INDEX = 0  # back to locomotion

DRIVE_SPEED = 0.6  # axis units, pre-scale (teleop_twist_joy applies scale_linear/scale_angular)

# keysym -> (axis index, value to hold)
DRIVE_BINDINGS = {
    XK.XK_w: (1, DRIVE_SPEED),
    XK.XK_s: (1, -DRIVE_SPEED),
    XK.XK_a: (3, DRIVE_SPEED),
    XK.XK_d: (3, -DRIVE_SPEED),
    XK.XK_q: (0, DRIVE_SPEED),
    XK.XK_e: (0, -DRIVE_SPEED),
}
TAP_BINDINGS = {
    XK.XK_t: TRIANGLE_BUTTON_INDEX,
    XK.XK_x: X_BUTTON_INDEX,
}


class WheelKeyboardTeleop(Node):
    def __init__(self):
        super().__init__("wheel_keyboard_teleop")
        self.pub = self.create_publisher(Joy, "/joy", 10)
        self.lock = threading.Lock()
        self.axes = [0.0] * NUM_AXES
        self.create_timer(1.0 / 20.0, self._publish_axes)

    def _publish_axes(self):
        with self.lock:
            axes = list(self.axes)
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = axes
        msg.buttons = [0] * NUM_BUTTONS
        self.pub.publish(msg)

    def set_axis(self, axis, value):
        with self.lock:
            self.axes[axis] = value

    def stop(self):
        with self.lock:
            self.axes = [0.0] * NUM_AXES

    def tap_button(self, button_index):
        for pressed in (1, 0):
            msg = Joy()
            msg.header.stamp = self.get_clock().now().to_msg()
            with self.lock:
                msg.axes = list(self.axes)
            msg.buttons = [0] * NUM_BUTTONS
            msg.buttons[button_index] = pressed
            self.pub.publish(msg)
            time.sleep(0.08)


def main():
    rclpy.init()
    node = WheelKeyboardTeleop()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    disp_name = os.environ.get("DISPLAY", ":1")
    d = display.Display(disp_name)
    root = d.screen().root

    keycode_to_drive = {}
    for keysym, binding in DRIVE_BINDINGS.items():
        kc = d.keysym_to_keycode(keysym)
        keycode_to_drive[kc] = binding
        root.grab_key(kc, X.AnyModifier, True, X.GrabModeAsync, X.GrabModeAsync)

    keycode_to_tap = {}
    for keysym, button_index in TAP_BINDINGS.items():
        kc = d.keysym_to_keycode(keysym)
        keycode_to_tap[kc] = button_index
        root.grab_key(kc, X.AnyModifier, True, X.GrabModeAsync, X.GrabModeAsync)

    space_kc = d.keysym_to_keycode(XK.XK_space)
    root.grab_key(space_kc, X.AnyModifier, True, X.GrabModeAsync, X.GrabModeAsync)
    d.sync()

    print(
        "Grabbed W/A/S/D/Q/E (set drive speed), Space (stop), T (tap Triangle -> "
        "activate wheel), X (tap X -> back to locomotion). Ctrl+C here to stop and "
        "release the grab.",
        flush=True,
    )

    try:
        while True:
            event = d.next_event()
            if event.type != X.KeyPress:
                continue
            kc = event.detail
            if kc in keycode_to_drive:
                axis, value = keycode_to_drive[kc]
                node.set_axis(axis, value)
                print(f"axis[{axis}] = {value}", flush=True)
            elif kc == space_kc:
                node.stop()
                print("stop", flush=True)
            elif kc in keycode_to_tap:
                button_index = keycode_to_tap[kc]
                if button_index == X_BUTTON_INDEX:
                    node.stop()
                print(f"tap button {button_index}", flush=True)
                node.tap_button(button_index)
    except KeyboardInterrupt:
        pass
    finally:
        for kc in list(keycode_to_drive) + list(keycode_to_tap) + [space_kc]:
            root.ungrab_key(kc, X.AnyModifier)
        d.sync()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
