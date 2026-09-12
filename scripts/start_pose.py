#!/usr/bin/env python3
"""Manage the desired startup reference; never zero encoders or command motors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws/src/robot_calibration"))
from robot_calibration.storage import JOINT_NAMES, atomic_json, directory, validate

DEFAULT = ROOT / "hardware_testing/start_pose/start_pose.json"


def read_profile(path):
    profile = json.loads(path.read_text(encoding="utf-8"))
    if (profile.get("schema_version") != 1 or
            profile.get("purpose") != "desired_start_pose" or
            profile.get("angle_units") != "radians" or
            set(profile.get("joint_positions", {})) != set(JOINT_NAMES)):
        raise ValueError("Invalid start-pose schema or joint names")
    values = [profile["joint_positions"][name] for name in JOINT_NAMES]
    if any(type(q) not in (int, float) or not math.isfinite(q) for q in values):
        raise ValueError("All 12 joint positions must be finite numbers")
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    save = commands.add_parser("save", help="Save an explicitly selected calibration as the desired pose")
    save.add_argument("--from-calibration", type=Path, default=directory() / "calibration.json")
    save.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "save":
            raw = args.from_calibration.read_bytes()
            record = json.loads(raw)
            # Validate the historical record's structure, not its current-session validity.
            # Selecting it is explicit; it is never installed as a live calibration.
            validate(record, record["encoder_session_id"])
            profile = {
                "schema_version": 1,
                "purpose": "desired_start_pose",
                "angle_units": "radians",
                "coordinate_frame": "source_calibration_joint_frame",
                "joint_positions": dict(zip(JOINT_NAMES, record["reference_joint_positions"])),
                "source_calibration_id": record["calibration_id"],
                "source_calibration_sha256": hashlib.sha256(raw).hexdigest(),
                "source_captured_at_utc": record["captured_at_utc"],
                "source_commit": record.get("source_commit", "unknown"),
                "note": "Desired physical reference. Cross-boot encoder mapping is not yet established; this file does not enable automatic homing.",
            }
            if args.file.exists():
                if not args.replace:
                    raise ValueError("Start pose already exists; use --replace to update it")
                previous = read_profile(args.file)
                digest = hashlib.sha256(args.file.read_bytes()).hexdigest()[:16]
                atomic_json(args.file.parent / "history" / (digest + ".json"), previous)
            atomic_json(args.file, profile)
        profile = read_profile(args.file)
        print(f"Desired start pose: {args.file}")
        print(f"Source calibration: {profile['source_calibration_id']}")
        for name in JOINT_NAMES:
            print(f"  {name}: {profile['joint_positions'][name]:.8f} rad")
        print("Reference saved only. Automatic return across power cycles is not implemented.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"START POSE ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
