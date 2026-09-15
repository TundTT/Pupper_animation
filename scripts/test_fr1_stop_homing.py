#!/usr/bin/env python3
"""Experimental, near-stop FR1 homing trial. Never launches ROS or saves calibration.

Only for the operator-confirmed, supported robot at a freshly measured FR1
negative stop. Uses host PD converted to bounded feed-forward effort; firmware
kp/kd stay zero for ALL motors. A stationary reading alone is not accepted:
the trial must first demonstrate outward travel, then return to the measured
endpoint. This validates a short stop search, not arbitrary-pose homing.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import signal
import statistics
import struct
import subprocess
import time

from read_disabled_spi import decode

EFFORT_CAP = 0.4
DT = 0.01


def command(effort, enabled=True):
    if not math.isfinite(effort) or abs(effort) > EFFORT_CAP + 1e-9:
        raise ValueError('Effort outside trial limit')
    values = [0.0] * 30
    # spine_cmd_t: 6 q, 6 qd, 6 kp, 6 kd, 6 tau; FR1 slot 1.
    values[25] = -effort if enabled else 0.0
    raw = bytearray(struct.pack('<30f2I', *values, 0, int(enabled)))
    checksum = 0
    for word in struct.unpack('<32I', raw):
        checksum ^= word
    raw.extend(struct.pack('<I', checksum))
    for i in range(0, 132, 2):
        raw[i], raw[i + 1] = raw[i + 1], raw[i]
    return list(raw)


class Trial:
    def __init__(self, stop):
        self.stop = stop
        self.phase = 'backoff'
        self.phase_start = 0.0
        self.history = collections.deque()
        self.last_t = None
        self.last_q = None
        self.peak = stop
        self.endpoint = None

    def step(self, t, q):
        if not math.isfinite(q) or not math.isfinite(t):
            raise ValueError('Nonfinite sample')
        if self.last_t is not None and not 0 < t - self.last_t <= 0.06:
            raise RuntimeError('Feedback/control interval exceeded')
        if q < self.stop - 0.035 or q > self.stop + 0.18:
            raise RuntimeError('Travel envelope exceeded')
        velocity = 0 if self.last_t is None else (q - self.last_q) / (t - self.last_t)
        if abs(velocity) > 0.8:
            raise RuntimeError('Position-derived speed exceeded')
        self.last_t, self.last_q = t, q
        self.peak = max(self.peak, q)
        if t - self.phase_start > 5.0:
            raise RuntimeError(self.phase + ' timed out; no reference accepted')
        if self.phase == 'backoff':
            target = self.stop + min(0.06, 0.04 * t)
            if q >= self.stop + 0.045:
                self.phase, self.phase_start = 'seek', t
                self.seek_start = q
                self.history.clear()
        else:
            target = self.seek_start - 0.04 * (t - self.phase_start)
            self.history.append((t, q))
            while self.history and self.history[0][0] < t - 0.4:
                self.history.popleft()
            positions = [p for _, p in self.history]
            stable = (len(positions) >= 30 and
                      self.history[-1][0] - self.history[0][0] >= 0.35 and
                      max(positions) - min(positions) <= 0.002)
            if stable and target < q - 0.045 and abs(q - self.stop) <= 0.012:
                self.endpoint = statistics.median(positions)
                return 0.0, True
        # No accumulated firmware position error: kp=kd=0 on the wire.
        effort = max(-EFFORT_CAP, min(EFFORT_CAP, 6.0 * (target - q) - 0.15 * velocity))
        return effort, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--operator-confirmed-supported', action='store_true')
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding='utf-8-sig'))
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if reference['boot_id'] != boot or reference['joint'] != 'leg_front_r_1':
        raise ValueError('Wrong boot or joint')
    stop = float(reference['raw_reference_rad'])
    if not math.isfinite(stop) or reference['model_reference_rad'] != -1.22:
        raise ValueError('Invalid negative-stop reference')
    if not args.execute:
        print(json.dumps({'preview': True, 'joint': 'leg_front_r_1',
                          'stop': stop, 'effort_cap': EFFORT_CAP,
                          'backoff_rad': 0.06, 'per_phase_timeout_s': 5}))
        return
    if not args.operator_confirmed_supported:
        raise ValueError('Operator support confirmation required')
    if args.output.exists():
        raise ValueError('Preserve prior evidence')
    import fcntl
    import spidev
    lock = open('/tmp/control_board_hardware.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    busy = subprocess.run(['fuser', '/dev/spidev0.0', '/dev/spidev0.1'], capture_output=True)
    if busy.returncode != 1:
        raise RuntimeError('SPI ownership check failed')
    def abort(signum, frame):
        raise RuntimeError('Trial interrupted or deadline reached')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, abort)
    buses, rows = [], []
    result = {'boot_id': boot, 'joint': 'leg_front_r_1', 'success': False,
              'runtime_calibration_applied': False, 'effort_cap_nm_commanded': EFFORT_CAP}
    try:
        signal.setitimer(signal.ITIMER_REAL, 14)
        for channel in (0, 1):
            bus = spidev.SpiDev()
            bus.open(0, channel)
            bus.mode, bus.max_speed_hz, bus.bits_per_word = 0, 6000000, 8
            buses.append(bus)
        def exchange(effort=0, enabled=False):
            boards = [decode(buses[0].xfer2(command(effort, enabled))),
                      decode(buses[1].xfer2([0] * 132))]
            if any(b['all_zero_packet'] for b in boards):
                raise RuntimeError('Zero feedback packet')
            return [-q for b in boards for q in b['q']]
        # Enable the FR leg with zero gains/effort to request updated motor feedback.
        initial = []
        for _ in range(60):
            initial.append(exchange(0, True))
            time.sleep(DT)
        baseline = [statistics.median(r[i] for r in initial[-30:]) for i in range(12)]
        result['initial_samples'] = initial
        result['baseline_raw_rad'] = baseline
        result['expected_stop_raw_rad'] = stop
        if abs(baseline[1] - stop) > 0.012:
            raise RuntimeError('Current position differs from measured stop')
        if max(r[1] for r in initial[-30:]) - min(r[1] for r in initial[-30:]) > 0.002:
            raise RuntimeError('Initial position unstable')
        trial = Trial(stop)
        start, effort = time.monotonic(), 0.0
        while True:
            q = exchange(effort, True)
            t = time.monotonic() - start
            # Unpowered joints may move with gravity; stop this isolated trial if large.
            if any(abs(q[i] - baseline[i]) > 0.10 for i in range(12) if i != 1):
                raise RuntimeError('Another joint moved beyond trial allowance')
            next_effort, done = trial.step(t, q[1])
            rows.append({'t': t, 'q': q, 'phase': trial.phase, 'sent_effort': effort})
            effort = next_effort
            if done:
                result.update(success=True, endpoint_rad=trial.endpoint,
                              endpoint_error_rad=trial.endpoint - stop,
                              demonstrated_backoff_rad=trial.peak - stop,
                              candidate_offset_rad=trial.endpoint + 1.22)
                break
            time.sleep(DT)
    except Exception as exc:
        result['error'] = str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        disable_errors = []
        for _ in range(10):
            for bus in buses:
                try:
                    bus.xfer2([0] * 132)
                except Exception as exc:
                    disable_errors.append(str(exc))
            time.sleep(DT)
        for bus in buses:
            bus.close()
        result['disable_packets_sent_without_error'] = len(buses) == 2 and not disable_errors
        result['disable_errors'] = disable_errors
        result['rows'] = rows
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            json.dump(result, f, indent=2)
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
