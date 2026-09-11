#!/usr/bin/env python3
"""Read-only alignment deployment checks. Never starts ROS or commands motors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY = "policy_wheel_align_motion_v5.json"
EXPORT_SHA = "05f62e90101597ffbbd9d6a26dce443b3f1af831ba5a5879607f1709b19860a9"
CHECKPOINT_SHA = "83a3e4caa8481d22fcbff5ad39947057707c87c4fa394343c8abd0ddf6212cc3"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(controller_share, description_share):
    path = controller_share / "launch" / POLICY
    require(hashlib.sha256(path.read_bytes()).hexdigest() == EXPORT_SHA,
            "Policy is not the reviewed export (or Git LFS files are unresolved)")
    p = json.loads(path.read_text())
    config = yaml.safe_load((controller_share / "launch/config.yaml").read_text())
    c = config["neural_controller_wheel_align_hybrid"]["ros__parameters"]
    require(p["checkpoint_sha256"] == CHECKPOINT_SHA, "Wrong checkpoint")
    require(p["motion_contract_id"] == "quadmorph-align-motion-v5", "Wrong motion contract")
    require(p["in_shape"] == [None, 83] and p["policy_action_size"] == 8, "Wrong network shape")
    require(config["controller_manager"]["ros__parameters"]["update_rate"] == 520
            and c["repeat_action"] == 10 and abs(p["ctrl_dt"] - 10/520) < 1e-12,
            "Manager/policy timing differs from training")
    require(c["model_path"].endswith("/" + POLICY), "YAML selects another policy")
    require(c["gain_multiplier"] == 1, "Gain multiplier changed")
    for key in ("joint_names", "action_types", "default_joint_pos", "kps", "kds"):
        require(c[key] == p[key], f"YAML/export mismatch: {key}")
    require(c["init_kps"] == p["kps"] and c["init_kds"] == p["kds"], "Initialization gains mismatch")
    tree = ET.parse(description_share / "description/components.xacro")
    joints = {j.attrib["name"]: j for j in tree.findall(".//joint")}
    for i, name in enumerate(p["joint_names"]):
        joint = joints[name]
        hw = {x.attrib["name"]: float(x.text) for x in joint.findall("param")}
        require(hw["can_channel"] == [2, 1, 4, 3][i//3] and hw["can_id"] == i%3+1,
                f"CAN mapping mismatch: {name}")
        require({x.attrib["name"] for x in joint.findall("command_interface")} >=
                {"position", "velocity", "effort", "kp", "kd"}, f"Missing motor command interfaces: {name}")
        require({x.attrib["name"] for x in joint.findall("state_interface")} >=
                {"position", "velocity"}, f"Missing encoder interfaces: {name}")
        require(p["kps"][i] <= hw["kp_max"] and p["kds"][i] <= hw["kd_max"]
                and c["estop_kd"] <= hw["kd_max"], f"Gain would be clamped: {name}")
        if i % 3 == 2:
            require(p["action_types"][i] == "velocity" and p["kps"][i] == 0,
                    f"Hub is not in velocity mode: {name}")
            require(hw["position_min"] <= -1000 and hw["position_max"] >= 1000
                    and "hard_limit_min" not in hw and "hard_limit_max" not in hw
                    and hw["velocity_max"] >= 2, f"Hub profile would limit alignment: {name}")
        else:
            require(p["action_types"][i] == "position", f"Proximal mode mismatch: {name}")
            require(hw["position_min"] <= p["joint_lower_limits"][i]
                    < p["joint_upper_limits"][i] <= hw["position_max"], f"Position limits mismatch: {name}")
    geometry = json.loads((ROOT / "hardware_testing/align_v5_2026-09-11/training-geometry.json").read_text())
    urdf = ET.parse(description_share / "description/urdf/pupper_v3.edited.fixed.urdf")
    urdf_joints = {j.attrib["name"]: j for j in urdf.findall(".//joint")}
    for i, name in enumerate(p["joint_names"]):
        joint = urdf_joints[name]
        require([float(x) for x in joint.find("axis").attrib["xyz"].split()] == [0, 0, 1],
                f"Joint axis differs from training: {name}")
        origin = joint.find("origin").attrib
        xyz = [float(x) for x in origin["xyz"].split()]
        expected = geometry["p1" if i%3 == 0 else "p3"][i//3] if i%3 != 1 else [0, 0, 0]
        require(max(abs(a-b) for a, b in zip(xyz, expected)) < 1e-6, f"Joint origin differs: {name}")
        roll, pitch, yaw = (float(x) for x in origin["rpy"].split())
        cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        rotation = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                    [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
        expected_rotation = geometry[f"r{i%3+1}"][i//3]
        require(max(abs(rotation[a][b]-expected_rotation[a][b]) for a in range(3) for b in range(3)) < 1e-4,
                f"Joint frame differs from training: {name}")
    print(f"PASS: exact v5 checkpoint/export, 83/8 contract, 52 Hz inference, CAN order, modes, gains and limits: {path}")
    print("PASS: description joint axes and mounting transforms match training geometry within source rounding")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true", help="Also verify sourced ROS overlay packages")
    args = parser.parse_args()
    check(ROOT / "ros2_ws/src/neural_controller", ROOT / "ros2_ws/src/pupper_v3_description")
    if args.installed:
        shares = {}
        for package in ("robot_calibration", "control_board_hardware_interface", "neural_controller",
                        "joy_utils", "pupper_v3_description"):
            prefix = Path(subprocess.check_output(["ros2", "pkg", "prefix", package], text=True).strip())
            require(prefix.is_relative_to(ROOT / "ros2_ws/install"),
                    f"Wrong overlay for {package}: {prefix}; source this checkout's install/local_setup.bash")
            shares[package] = prefix / "share" / package
        check(shares["neural_controller"], shares["pupper_v3_description"])
        for package in ("controller_manager", "robot_state_publisher", "joy_linux"):
            subprocess.run(["ros2", "pkg", "prefix", package], check=True, stdout=subprocess.DEVNULL)
        require(Path("/dev/input/js0").exists(), "Joystick /dev/input/js0 is missing")
        print("PASS: selected overlay and core launch packages; joystick device exists")
    print("Simulation acceptance: nominal 64/64; randomized 45/64 (FAILED); interrupted not passed.")
    print("Software compatibility only. Physical pose, clearance, tracking and timing still need the lab trial.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"FAIL: {error}") from error
