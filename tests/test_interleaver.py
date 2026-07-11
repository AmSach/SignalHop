"""Tests for the SignalHop burst-error interleaver."""

import os
import sys
import random
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.interleaver import (  # noqa: E402
    interleave,
    deinterleave,
    interleave_bytes,
    deinterleave_bytes,
    burst_tolerance,
)


class TestInterleaveRoundTrip(unittest.TestCase):
    def test_round_trip_exact_block(self):
        bits = [random.randint(0, 1) for _ in range(64)]
        out = interleave(bits, 4, 16)
        self.assertEqual(len(out), 64)
        self.assertEqual(deinterleave(out, 4, 16), bits)

    def test_round_trip_with_padding(self):
        bits = [1, 0, 1, 1, 0, 0, 1, 0, 1]  # 9 bits, block size 16
        out = interleave(bits, 4, 4)
        self.assertEqual(len(out), 16)
        self.assertEqual(deinterleave(out, 4, 4)[:9], bits)

    def test_empty_input(self):
        self.assertEqual(interleave([], 4, 4), [])
        self.assertEqual(deinterleave([], 4, 4), [])

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            interleave([0, 1, 0], 0, 4)
        with self.assertRaises(ValueError):
            interleave([0, 1, 0], 4, 0)

    def test_burst_tolerance(self):
        max_burst, per_cw = burst_tolerance(8, 16)
        self.assertEqual((max_burst, per_cw), (16, 1))


class TestBurstSpreading(unittest.TestCase):
    def test_burst_spreads_across_codewords(self):
        """A 5-bit burst should hit 5 different codewords, not kill one."""
        bits = [1] * 64  # all-1 codeword
        interleaved = interleave(bits, 4, 16)
        # inject a burst of 5 errors starting at index 20
        for i in range(20, 25):
            interleaved[i] ^= 1
        recovered = deinterleave(interleaved, 4, 16)
        # count error rows
        error_rows = sum(1 for i in range(64) if recovered[i] != bits[i])
        self.assertEqual(error_rows, 5, "burst should be spread, not concentrated")


class TestByteRoundTrip(unittest.TestCase):
    def test_bytes_round_trip(self):
        msg = b"SignalHop is a low-bandwidth, long-range acoustic mesh network."
        for rows, cols in [(4, 8), (8, 8), (2, 16)]:
            with self.subTest(rows=rows, cols=cols):
                out = interleave_bytes(msg, rows, cols)
                back = deinterleave_bytes(out, rows, cols)
                self.assertEqual(back[: len(msg)], msg)


class TestInterleaverProperties(unittest.TestCase):
    def test_permutation_is_bijection(self):
        bits = list(range(64))  # distinct values so any swap is detectable
        out = interleave(bits, 4, 16)
        self.assertEqual(sorted(out), sorted(bits))

    def test_reordering_changes_order(self):
        bits = [int(b) for b in "0000" + "1111" + "0000" + "1111"]
        out = interleave(bits, 4, 4)
        # should not equal input
        self.assertNotEqual(out, bits)


if __name__ == "__main__":
    unittest.main()
