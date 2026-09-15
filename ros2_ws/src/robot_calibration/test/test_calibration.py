import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robot_calibration.storage import (JOINT_NAMES, StationarySample, atomic_json,
    capture_lock, current_session, directory, load_current, make_record, process_start, save_record)
from robot_calibration.cli import main


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"QUADMORPH_CALIBRATION_DIR": self.tmp.name})
        self.env.start()
        self.session = {"schema_version": 1, "session_id": "test-session", "ready": True,
                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                        "owner_pid": os.getpid(), "owner_start_ticks": process_start(os.getpid())}
        self.write_session()
        self.q = [1, 0, 0.3, -1, 0, -0.4, 1, 0, 0.5, -1, 0, -0.6]

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def write_session(self):
        atomic_json(directory() / "encoder-session.json", self.session)

    def record(self):
        return make_record("test-session", self.q, pose_note="test fixture only")

    def test_save_load_and_reuse_without_recalibrating(self):
        record = self.record()
        with capture_lock():
            save_record(record)
        self.assertEqual(load_current(), record)
        self.assertTrue((directory() / "history" / (record["calibration_id"] + ".json")).is_file())
        with self.assertRaisesRegex(ValueError, "already calibrated"):
            save_record(self.record())
        self.assertEqual(load_current(), record)
        replacement = self.record()
        with capture_lock():
            save_record(replacement, replace=True)
        self.assertEqual(load_current(), replacement)
        self.assertEqual(len(list((directory() / "history").glob("*.json"))), 2)

    def test_new_capture_archives_old_motion_references(self):
        record = self.record()
        for name in ("triangle-roll-map.json", "walking-handoff.json"):
            atomic_json(directory() / name, {"calibration_id": "old"})
        with capture_lock():
            save_record(record)
        for name in ("triangle-roll-map.json", "walking-handoff.json"):
            self.assertFalse((directory() / name).exists())
            archive = directory() / "history" / (record["calibration_id"] + "-superseded") / name
            self.assertEqual(json.loads(archive.read_text()), {"calibration_id": "old"})

    def test_restart_rehome_reboot_dead_owner_and_pending_homing_rejected(self):
        save_record(self.record())
        cases = {"session_id": "new-hardware-activation", "boot_id": "previous-boot",
                 "owner_start_ticks": "0", "owner_pid": 999999999, "ready": False}
        for key, value in cases.items():
            original = self.session[key]
            self.session[key] = value
            self.write_session()
            with self.assertRaises((OSError, ValueError)):
                load_current()
            self.session[key] = original
            self.write_session()

    def test_boost_session_serialization(self):
        # Boost property_tree writes scalar fields as JSON strings.
        self.session.update(schema_version="1", ready="true", owner_pid=str(os.getpid()))
        self.write_session()
        self.assertEqual(current_session(), "test-session")
        self.session["ready"] = "false"
        self.write_session()
        with self.assertRaises(ValueError):
            current_session()

    def test_invalid_values_and_manual_entry(self):
        home = [v + 2*math.pi for v in self.q[2::3]]
        record = make_record("test-session", self.q, home)
        self.assertEqual(record["source"], "manual_encoder_values")
        for value in [float("nan"), float("inf"), True, "0"]:
            bad = self.q.copy()
            bad[0] = value
            with self.assertRaises(ValueError):
                make_record("test-session", bad)
        with self.assertRaisesRegex(ValueError, "differs from current"):
            make_record("test-session", self.q, [0, 0, 0, 0])
        with self.assertRaises(ValueError):
            make_record("test-session", self.q, [0, 0, 0])

    def test_incompatible_record_rejected_without_overwriting_history(self):
        for key, value in [("operator_confirmed", False), ("encoder_session_id", "old"),
                           ("joint_names", list(reversed(JOINT_NAMES))),
                           ("angle_units", "degrees"), ("wheel_base_target", [0, 0, 0, 0]),
                           ("wheel_home", [0, 1]), ("calibration_id", "")]:
            record = self.record()
            record[key] = value
            atomic_json(directory() / "calibration.json", record)
            with self.assertRaises(ValueError):
                load_current()

    def test_failed_atomic_write_keeps_previous_record(self):
        save_record(self.record())
        previous = (directory() / "calibration.json").read_bytes()
        with patch("robot_calibration.storage.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                atomic_json(directory() / "calibration.json", self.record())
        self.assertEqual((directory() / "calibration.json").read_bytes(), previous)
        self.assertFalse(list(directory().glob("*.tmp")))

    def test_capture_lock_excludes_other_capture_or_activation(self):
        with capture_lock():
            with self.assertRaisesRegex(ValueError, "already in progress"):
                with capture_lock():
                    pass

    def test_cli_requires_explicit_confirmation_before_ros_sampling(self):
        with patch("robot_calibration.cli.sys.stdin.isatty", return_value=False), patch("robot_calibration.cli.sample_robot") as sample:
            self.assertEqual(main(["capture"]), 1)
            sample.assert_not_called()
        with patch("robot_calibration.cli.sample_robot", return_value=self.q):
            self.assertEqual(main(["capture", "--operator-confirmed"]), 0)
        self.assertEqual(load_current()["wheel_home"], self.q[2::3])

    def test_cli_rejects_stack_restart_after_confirmation(self):
        def confirm(_):
            self.session["session_id"] = "new"
            self.write_session()
            return "CALIBRATE"
        with patch("robot_calibration.cli.sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=confirm), patch("robot_calibration.cli.sample_robot") as sample:
            self.assertEqual(main(["capture"]), 1)
            sample.assert_not_called()


class StationaryTest(unittest.TestCase):
    def sample(self, sampler, t, q=None, v=None, names=None, age=0.01, stamp=None):
        return sampler.observe(names or JOINT_NAMES, q if q is not None else [0.0]*12,
                               v if v is not None else [0.0]*12,
                               stamp if stamp is not None else 100+t, 100+t+age, t)

    def test_operator_confirmed_mode_ignores_speed_but_rejects_drift(self):
        stationary = StationarySample(check_velocity=False)
        moving = StationarySample(check_velocity=False)
        for n in range(80):
            ready = self.sample(stationary, n*0.02, v=[0.5]*12)
            self.assertFalse(self.sample(moving, n*0.02, q=[n*0.002]*12, v=[0.0]*12))
        self.assertTrue(ready)
        self.assertEqual(stationary.POLICY, "operator_confirmed_position_stability_v1")
        self.assertFalse(self.sample(stationary, 1.6, v=[float('nan')]*12))

    def test_stationary_named_samples_and_wrap_boundary(self):
        sampler = StationarySample()
        names = list(reversed(JOINT_NAMES))
        for n in range(60):
            q = [0.0]*12
            q[2] = math.pi-0.001 if n % 2 else -math.pi+0.001
            ready = self.sample(sampler, n*0.02, list(reversed(q)), names=names)
        self.assertTrue(ready)
        self.assertEqual(sampler.positions, q)

    def test_moving_stale_missing_duplicate_future_or_nonfinite_cannot_capture(self):
        cases = [dict(v=[0.1]*12), dict(age=1.0), dict(age=-0.1),
                 dict(q=[float("nan")]*12), dict(names=["other"]*12), dict(q=[0.0]*11),
                 dict(v=[]), dict(stamp=0)]
        for case in cases:
            sampler = StationarySample()
            for n in range(80):
                self.assertFalse(self.sample(sampler, n*0.02, **case))

    def test_low_reported_velocity_does_not_hide_position_drift(self):
        sampler = StationarySample()
        for n in range(80):
            self.assertFalse(self.sample(sampler, n*0.02, q=[n*0.002]*12))

    def test_dropped_or_repeated_messages_restart_stationary_dwell(self):
        for repeated in [False, True]:
            sampler = StationarySample()
            for n in range(40):
                self.assertFalse(self.sample(sampler, n*0.02))
            if repeated:
                self.assertFalse(self.sample(sampler, 0.80, stamp=100.78))
            else:
                self.assertFalse(self.sample(sampler, 1.1))

    def test_brief_quantized_velocity_outliers_with_tight_positions(self):
        sampler = StationarySample()
        accepted = []
        for n in range(601):
            q = [0.0]*12
            q[5] = 0.00114 if n % 2 else 0.0
            v = [0.015877]*12
            if n % 100 == 40:
                v[5] = 0.14286  # Isolated 5 ms report, conservatively charged 10 ms.
            if self.sample(sampler, n*0.005, q=q, v=v):
                accepted.append(n)
        self.assertTrue(accepted)
        self.assertGreaterEqual(min(accepted), 200)
        self.assertFalse(self.sample(sampler, 3.005, v=[0.1]*12))

    def test_sustained_bursty_or_frequent_speed_is_not_a_glitch(self):
        patterns = [lambda n: True, lambda n: n % 100 < 8,
                    lambda n: n % 10 == 0]
        for pattern in patterns:
            sampler = StationarySample()
            for n in range(601):
                v = [0.1 if pattern(n) else 0.015877]*12
                self.assertFalse(self.sample(sampler, n*0.005, v=v))

    def test_tight_position_excursion_catches_drift_and_oscillation(self):
        for position in [lambda n: n*0.000015, lambda n: 0.003 if n % 2 else 0.0]:
            sampler = StationarySample()
            for n in range(601):
                self.assertFalse(self.sample(sampler, n*0.005, q=[position(n)]*12))

    def test_large_speed_spike_and_bad_data_require_a_new_full_window(self):
        for bad in [dict(v=[0.21]*12), dict(v=[float('nan')]*12), dict(age=0.3)]:
            sampler = StationarySample()
            for n in range(220):
                self.sample(sampler, n*0.005)
            self.assertFalse(self.sample(sampler, 1.1, **bad))
            for n in range(221, 420):
                self.assertFalse(self.sample(sampler, n*0.005))

    def test_backward_receipt_clock_resets_capture(self):
        sampler = StationarySample()
        for n in range(220):
            self.sample(sampler, n*0.005)
        self.assertFalse(self.sample(sampler, 1.0, stamp=101.1))


if __name__ == "__main__":
    unittest.main()
