#!/usr/bin/env python3
"""Global spacebar -> fake /joy 'O button' press, for exercising the leg-lift
controller without a physical PS5 gamepad connected to the machine running
this script. Grabs the Space key on the X server (DISPLAY, default :1) so it
works regardless of window focus -- including while the MuJoCo viewer window
is focused, even though MuJoCo itself also binds Space (to its "Mode"
toggle); the X grab means our listener gets the event instead of whichever
window has focus.

Each spacebar tap publishes a single /joy press+release pair on button
index 1 (O / leg-lift-cycle in joy_util_node's config), matching a real
joystick press for edge-detection in estop_controller.cpp.

Only useful on a machine with an X session (e.g. sim testing on the
workstation). On the real robot, just plug a controller into the robot
directly -- joy_linux_node reads local hardware, not anything over the
network.

Requires: pip install python-xlib   (pure Python, no compilation needed)
Run inside the same ROS2 env / sourced workspace as the launch you're testing.
"""
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from Xlib import X, XK, display


NUM_BUTTONS = 16
NUM_AXES = 8
O_BUTTON_INDEX = 1  # matches joy_util_node's leg_lift_button_index


class SpacebarJoySim(Node):
    def __init__(self):
        super().__init__("spacebar_joy_sim")
        self.pub = self.create_publisher(Joy, "/joy", 10)

    def send_tap(self):
        for pressed in (1, 0):
            msg = Joy()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.axes = [0.0] * NUM_AXES
            msg.buttons = [0] * NUM_BUTTONS
            msg.buttons[O_BUTTON_INDEX] = pressed
            self.pub.publish(msg)
            time.sleep(0.08)


def main():
    rclpy.init()
    node = SpacebarJoySim()

    disp_name = os.environ.get("DISPLAY", ":1")
    d = display.Display(disp_name)
    root = d.screen().root
    space_keycode = d.keysym_to_keycode(XK.XK_space)

    # X.AnyModifier so it fires regardless of Caps/Num Lock state.
    root.grab_key(space_keycode, X.AnyModifier, True, X.GrabModeAsync, X.GrabModeAsync)
    d.sync()
    print(f"Grabbed Space on {disp_name}. Tap spacebar to step the O-button leg-lift "
          f"cycle. Ctrl+C here to stop and release the grab.", flush=True)

    try:
        while True:
            event = d.next_event()
            if event.type == X.KeyPress and event.detail == space_keycode:
                print("Space tapped -> publishing O-button press", flush=True)
                node.send_tap()
    except KeyboardInterrupt:
        pass
    finally:
        root.ungrab_key(space_keycode, X.AnyModifier)
        d.sync()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
