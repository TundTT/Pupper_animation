"""Versioned, local calibration storage. No ROS or motor operations in this module."""
from contextlib import contextmanager
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

JOINT_NAMES = [f"leg_{leg}_{joint}" for leg in ("front_r", "front_l", "back_r", "back_l")
               for joint in (1, 2, 3)]
WHEEL_NAMES = JOINT_NAMES[2::3]


def directory():
    if "QUADMORPH_CALIBRATION_DIR" in os.environ:
        path = Path(os.environ["QUADMORPH_CALIBRATION_DIR"])
    else:
        state = os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))
        path = Path(state) / "quadmorph"
    if not path.is_absolute():
        raise ValueError("Calibration directory must be absolute")
    return path


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def process_start(pid):
    text = Path(f"/proc/{int(pid)}/stat").read_text()
    return text[text.rindex(")") + 2:].split()[19]


def current_session():
    """Reject records left by a dead process, reboot, pending homing or new activation."""
    session = json.loads((directory() / "encoder-session.json").read_text())
    if (not isinstance(session, dict) or int(session["schema_version"]) != 1 or session["ready"] not in (True, "true") or
            session["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip() or
            str(session["owner_start_ticks"]) != process_start(session["owner_pid"]) or
            not isinstance(session["session_id"], str) or not session["session_id"]):
        raise ValueError("No live, homed encoder session; start the updated robot stack first")
    return session["session_id"]


def finite_vector(values, size):
    if (not isinstance(values, list) or len(values) != size or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in values)):
        raise ValueError(f"Expected {size} finite numeric angles")
    return [float(x) for x in values]


def validate(record, session):
    if (not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("encoder_session_id") != session or
            record.get("operator_confirmed") is not True or record.get("angle_units") != "radians" or
            record.get("reference_convention") != "marked_point_ring_home" or
            record.get("joint_names") != JOINT_NAMES or
            not isinstance(record.get("calibration_id"), str) or not record["calibration_id"]):
        raise ValueError("Missing, incompatible or stale calibration; capture for this startup")
    finite_vector(record.get("reference_joint_positions"), 12)
    home = finite_vector(record.get("wheel_home"), 4)
    target = finite_vector(record.get("wheel_base_target"), 4)
    if any(abs(wrap(t - h - math.pi)) > 1e-9 for h, t in zip(home, target)):
        raise ValueError("Base targets disagree with home + pi")
    return record


def load_current():
    session = current_session()
    record = validate(json.loads((directory() / "calibration.json").read_text()), session)
    if current_session() != session:
        raise ValueError("Encoder session changed while reading calibration")
    return record


@contextmanager
def capture_lock():
    import fcntl  # Robot is Linux; status/storage inspection remains importable elsewhere.
    directory().mkdir(parents=True, exist_ok=True)
    with (directory() / "capture.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Calibration/startup is already in progress") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def atomic_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def make_record(session, positions, wheel_home=None, pose_note="", source_commit="unknown"):
    positions = finite_vector(positions, 12)
    home = finite_vector(wheel_home, 4) if wheel_home is not None else positions[2::3]
    # Manual entry means readings in the CURRENT frame, not arbitrary desired zero values.
    if any(abs(wrap(h - q)) > 0.03 for h, q in zip(home, positions[2::3])):
        raise ValueError("Entered home differs from current encoders by more than 0.03 rad")
    return {
        "schema_version": 1, "calibration_id": uuid.uuid4().hex,
        "encoder_session_id": session, "operator_confirmed": True,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "angle_units": "radians", "reference_convention": "marked_point_ring_home",
        "joint_names": JOINT_NAMES, "reference_joint_positions": positions,
        "wheel_home": [wrap(h) for h in home],
        "wheel_base_target": [wrap(h + math.pi) for h in home],
        "source": "manual_encoder_values" if wheel_home is not None else "joint_states",
        "pose_note": pose_note, "source_commit": source_commit,
    }


def save_record(record, replace=False):
    """Caller holds capture_lock from before checking controller ownership through save."""
    session = current_session()
    validate(record, session)
    try:
        previous = load_current()
    except (OSError, ValueError, KeyError):
        previous = None
    if previous is not None and not replace:
        raise ValueError("This startup is already calibrated; use status or explicit --replace")
    atomic_json(directory() / "history" / (record["calibration_id"] + ".json"), record)
    if current_session() != session:
        raise ValueError("Encoder session changed; calibration was not activated")
    # These coordinate snapshots belong to the previous capture. Retain the
    # evidence, but do not let it block walking with the new marked-hub frame.
    for name in ("triangle-roll-map.json", "walking-handoff.json"):
        path = directory() / name
        if path.exists():
            archive = directory() / "history" / (record["calibration_id"] + "-superseded")
            archive.mkdir(parents=True, exist_ok=True)
            path.rename(archive / name)
    atomic_json(directory() / "calibration.json", record)


class StationarySample:
    """Bound encoder outliers within a fresh, tightly position-stable window.

    Operator-confirmed stationary hardware recordings contain quantized velocity
    spikes. Tolerate at most 50 ms total / 20 ms consecutive overspeed in a
    one-second window, only with <= 0.002 rad peak-to-peak position excursion.
    These are capture tolerances, not proof that the robot is physically still.
    """
    POLICY = "bounded_velocity_outliers_v1"

    def __init__(self, duration=1.0, max_velocity=0.02, max_drift=0.002, check_velocity=True):
        self.check_velocity = check_velocity
        self.POLICY = "bounded_velocity_outliers_v1" if check_velocity else "operator_confirmed_position_stability_v1"
        self.duration, self.max_velocity, self.max_drift = duration, max_velocity, max_drift
        self.reset()

    def reset(self):
        self.start = self.last_receipt = self.last_stamp = self.anchor = self.positions = None
        self.count = 0
        self.samples = deque()

    def observe(self, names, positions, velocities, stamp, ros_now, monotonic_now):
        if (len(names) != len(set(names)) or len(positions) != len(names) or
                len(velocities) != len(names) or not set(JOINT_NAMES).issubset(names) or
                not all(math.isfinite(x) for x in (stamp, ros_now, monotonic_now)) or
                stamp <= 0 or not 0 <= ros_now - stamp <= 0.2):
            self.reset()
            return False
        order = [names.index(name) for name in JOINT_NAMES]
        try:
            q = finite_vector([positions[i] for i in order], 12)
            v = finite_vector([velocities[i] for i in order], 12)
        except ValueError:
            self.reset()
            return False
        speed = max(abs(x) for x in v)
        if ((self.check_velocity and speed > 0.2) or
                (self.last_stamp is not None and stamp <= self.last_stamp) or
                (self.last_receipt is not None and monotonic_now <= self.last_receipt)):
            self.reset()
            return False
        if (self.last_receipt is not None and
                (monotonic_now - self.last_receipt > 0.2 or stamp - self.last_stamp > 0.2)):
            self.reset()
        self.last_receipt, self.last_stamp, self.positions = monotonic_now, stamp, q
        self.samples.append((stamp, q, speed > self.max_velocity, monotonic_now))
        cutoff = stamp - self.duration
        # Retain the sample bracketing the start of the full window.
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        self.count += 1
        if (len(self.samples) < 10 or self.samples[0][0] > cutoff or
                monotonic_now - self.samples[0][3] < self.duration or (self.check_velocity and speed > self.max_velocity)):
            return False
        anchor = self.samples[0][1]
        for joint in range(12):
            offsets = [wrap(row[1][joint] - anchor[joint]) for row in self.samples]
            if max(offsets) - min(offsets) > self.max_drift + 1e-12:
                return False
        if not self.check_velocity:
            return True
        total = burst = 0.0
        previous = self.samples[0]
        for row in list(self.samples)[1:]:
            dt = row[0] - max(previous[0], cutoff)
            # Charge both edges of an outlier: never assume an unseen interval
            # between a high and low speed sample was stationary.
            if row[2] or previous[2]:
                total += dt
                burst += dt
                if total > 0.05 + 1e-9 or burst > 0.02 + 1e-9:
                    return False
            else:
                burst = 0.0
            previous = row
        return True
