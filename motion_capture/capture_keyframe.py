#!/usr/bin/env python3
"""Capture a single labeled keyframe (one /joint_states snapshot) into keyframes.json.

Run this ON THE ROBOT (needs rclpy) after the stack is launched and homed, while
the joints are backdrivable (no active position controller) and posed by hand:

    source /opt/ros/jazzy/setup.bash
    python3 capture_keyframe.py --label flip_start

Appends one entry to ./keyframes.json (created if missing) in the current directory.
Run it again with a new --label each time you want to mark another pose.
"""
import argparse
import json
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class _Once(Node):
    def __init__(self):
        super().__init__("capture_keyframe")
        self.msg = None
        self.create_subscription(JointState, "/joint_states", self._cb, 10)

    def _cb(self, msg):
        self.msg = msg


def capture_once(timeout=5.0):
    rclpy.init()
    node = _Once()
    start = time.time()
    while node.msg is None and time.time() - start < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    msg = node.msg
    node.destroy_node()
    rclpy.shutdown()
    if msg is None:
        raise RuntimeError(
            "No /joint_states message received within timeout -- is the stack launched?"
        )
    return msg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True, help="Name for this pose, e.g. flip_start")
    ap.add_argument("--file", default="keyframes.json", help="Output JSON file (default: keyframes.json)")
    args = ap.parse_args()

    msg = capture_once()
    entry = {
        "label": args.label,
        "captured_at": time.time(),
        "joint_names": list(msg.name),
        "position": list(msg.position),
        "velocity": list(msg.velocity) if msg.velocity else [],
    }

    data = []
    if os.path.exists(args.file):
        with open(args.file) as f:
            data = json.load(f)
    data.append(entry)
    with open(args.file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Captured keyframe {args.label!r} ({len(data)} total in {args.file})")


if __name__ == "__main__":
    main()
