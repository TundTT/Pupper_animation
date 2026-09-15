"""Offline checks: packet isolation, effort bound, and failed/successful stop searches."""
import math
import struct
import unittest

from test_fr1_stop_homing import Trial, command, EFFORT_CAP


class Checks(unittest.TestCase):
    def test_packet_isolation_and_checksum(self):
        for effort in (-EFFORT_CAP, 0, EFFORT_CAP):
            packet = bytearray(command(effort))
            for i in range(0, 132, 2):
                packet[i], packet[i+1] = packet[i+1], packet[i]
            values = struct.unpack('<30f3I', packet)
            self.assertAlmostEqual(values[25], -effort)
            self.assertTrue(all(v == 0 for i, v in enumerate(values[:30]) if i != 25))
            self.assertEqual(values[30:32], (0, 1))
            checksum = 0
            for word in struct.unpack('<33I', packet):
                checksum ^= word
            self.assertEqual(checksum, 0)
        self.assertEqual(command(.3, False), [0] * 132)
        for invalid in (float('nan'), float('inf'), .401, -.401):
            with self.assertRaises(ValueError):
                command(invalid)

    def test_stale_or_stuck_does_not_home(self):
        trial = Trial(-2.13)
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            for i in range(510):
                effort, done = trial.step(i * .01, -2.13)
                self.assertLessEqual(abs(effort), EFFORT_CAP)
                self.assertFalse(done)

    def test_timing_speed_and_envelope(self):
        for t, q, message in ((.1, 0, 'interval'), (.01, .02, 'speed'),
                              (.01, -.04, 'envelope'), (.01, .19, 'envelope')):
            trial = Trial(0)
            trial.step(0, 0)
            with self.assertRaisesRegex(RuntimeError, message):
                trial.step(t, q)

    def test_synthetic_mechanical_stop(self):
        trial, q, velocity, effort = Trial(0), 0., 0., 0.
        done = False
        for i in range(1000):
            # Damped synthetic joint with unilateral hard stop at zero.
            velocity += (effort - 1.5 * velocity) / .1 * .01
            q = max(0, q + velocity * .01)
            if q == 0 and velocity < 0:
                velocity = 0
            effort, done = trial.step(i * .01, q)
            self.assertLessEqual(abs(effort), EFFORT_CAP)
            if done:
                break
        self.assertTrue(done)
        self.assertGreaterEqual(trial.peak, .045)
        self.assertLessEqual(abs(trial.endpoint), .012)

    def test_false_stop_away_from_reference_rejected(self):
        trial = Trial(0)
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            for i in range(900):
                q = min(.05, i * .0004)
                _, done = trial.step(i * .01, q)
                self.assertFalse(done)


if __name__ == '__main__':
    unittest.main()
