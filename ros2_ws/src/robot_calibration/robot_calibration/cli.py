"""Read-only ROS sampling plus explicit operator-confirmed calibration storage."""
import argparse
import json
import subprocess
import sys
import time
from .storage import (StationarySample, capture_lock, current_session, directory,
                      load_current, make_record, save_record)


def sample_robot(timeout, expected_session):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from controller_manager_msgs.srv import ListControllers
    from sensor_msgs.msg import JointState

    rclpy.init(args=[])
    node = Node("startup_calibration_capture")
    sampler = StationarySample()
    ready = False

    def observe(msg):
        nonlocal ready
        ready = sampler.observe(list(msg.name), list(msg.position), list(msg.velocity),
                                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                                node.get_clock().now().nanoseconds * 1e-9, time.monotonic())

    node.create_subscription(JointState, "/joint_states", observe, qos_profile_sensor_data)
    client = node.create_client(ListControllers, "/controller_manager/list_controllers")
    deadline = time.monotonic() + timeout

    def check_controllers():
        if not client.wait_for_service(timeout_sec=max(0.0, deadline - time.monotonic())):
            raise ValueError("Controller manager unavailable; cannot verify inactive motion controllers")
        future = client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=max(0.0, deadline-time.monotonic()))
        if not future.done() or future.result() is None:
            raise ValueError("Controller state query timed out")
        controllers = future.result().controller
        if not controllers:
            raise ValueError("No controllers reported; wait for stack startup to finish")
        owners = [c.name for c in controllers if c.state == "active" and c.claimed_interfaces]
        if owners:
            raise ValueError("Motion controllers must be inactive before calibration: " + ", ".join(owners))

    try:
        check_controllers()
        sampler.reset()  # Only samples after the initial ownership check count.
        ready = False
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if ready and sampler.last_receipt is not None and time.monotonic()-sampler.last_receipt < 0.2:
                check_controllers()
                if not ready or time.monotonic()-sampler.last_receipt >= 0.2:
                    continue
                if current_session() != expected_session:
                    raise ValueError("Hardware encoder session changed during capture")
                return sampler.positions
        raise ValueError("No stationary sample: need 1 second of fresh data for all 12 joints, "
                         "<=0.002 rad position excursion, and bounded velocity outliers "
                         "(<=20 ms consecutive, <=50 ms total, <=0.2 rad/s)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Validate the live session and saved calibration; never enables motion")
    status.add_argument("--json", action="store_true")
    capture = commands.add_parser("capture", help="Prompt, read stationary encoders, and save; never commands motors")
    capture.add_argument("--operator-confirmed", action="store_true",
                         help="Agent use ONLY after the user explicitly confirms physical setup for this startup")
    capture.add_argument("--wheel-home", type=float, nargs=4, metavar=("FR", "FL", "BR", "BL"),
                         help="Optional current encoder readings in radians; checked against live capture")
    capture.add_argument("--pose-note", default="", help="Description of the physical mark and proximal reference pose")
    capture.add_argument("--replace", action="store_true", help="Explicitly replace this session's home while controllers are inactive")
    capture.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            record = load_current()
        else:
            if not 1.0 <= args.timeout <= 120.0:
                raise ValueError("Timeout must be between 1 and 120 seconds")
            session = current_session()
            if not args.operator_confirmed:
                if not sys.stdin.isatty():
                    raise ValueError("User confirmation required: run interactively or obtain the user's confirmation before --operator-confirmed")
                print("Position the marked wheel rings in the agreed home pose, with the robot supported and motion controllers inactive.")
                print("Keep all joints stationary. This records angles; it does not move or physically home the wheels.")
                if input("Type CALIBRATE when the physical setup is ready: ").strip() != "CALIBRATE":
                    raise ValueError("Calibration cancelled; nothing saved")
            with capture_lock():
                if current_session() != session:
                    raise ValueError("Stack restarted after confirmation; confirm the new startup before capturing")
                if not args.replace:
                    try:
                        load_current()
                    except (OSError, ValueError, KeyError):
                        pass
                    else:
                        raise ValueError("Already calibrated for this startup; use status, or explicit --replace")
                positions = sample_robot(args.timeout, session)
                try:
                    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
                except (OSError, subprocess.CalledProcessError):
                    commit = "unknown"
                record = make_record(session, positions, args.wheel_home, args.pose_note, commit)
                record["stationarity_check"] = StationarySample.POLICY
                save_record(record, replace=args.replace)
        if getattr(args, "json", False):
            print(json.dumps(record, indent=2))
        else:
            print(f"VALID calibration {record['calibration_id']} for encoder session {record['encoder_session_id']}")
            print("Wheel home FR/FL/BR/BL (rad): " + ", ".join(f"{q:.8f}" for q in record["wheel_home"]))
            print(f"Saved record: {directory() / 'calibration.json'}")
            print("No controller was activated.")
        return 0
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"CALIBRATION NOT READY: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
