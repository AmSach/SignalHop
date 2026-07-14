"""Tests for sim_absorption.py — Thorp / Francois-Garrison underwater absorption."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim_absorption import (
    absorption_db_per_km,
    total_transmission_loss_db,
    max_range_for_loss,
    band_summary,
)


class TestAbsorptionCoefficients(unittest.TestCase):
    def test_alpha_25khz_in_underwater_band(self):
        """25 kHz is a common underwater band. Alpha should be 4-8 dB/km."""
        a = absorption_db_per_km(25_000.0)
        self.assertGreater(a, 4.0)
        self.assertLess(a, 8.0)

    def test_alpha_increases_with_frequency_overall(self):
        """Absorption grows with frequency overall (Thorp model)."""
        # Compare low-band (100 Hz) to mid-band (10 kHz) to high-band (100 kHz).
        a_lo = absorption_db_per_km(100.0, model="thorp")
        a_md = absorption_db_per_km(10_000.0, model="thorp")
        a_hi = absorption_db_per_km(100_000.0, model="thorp")
        self.assertLess(a_lo, a_md)
        self.assertLess(a_md, a_hi)

    def test_thorp_vs_francois_at_25khz_same_order(self):
        """Both models should give comparable-magnitude values at 25 kHz."""
        a_t = absorption_db_per_km(25_000.0, model="thorp")
        a_f = absorption_db_per_km(25_000.0, model="francois")
        # Both should be in the 1-50 dB/km ballpark for 25 kHz underwater.
        self.assertGreater(a_t, 1.0)
        self.assertGreater(a_f, 0.01)

    def test_very_low_freq_uses_thorp(self):
        """At 50 Hz, Thorp's low-frequency limit applies; alpha ~ 0.003 / 0.9144 dB/km."""
        a = absorption_db_per_km(50.0, model="auto")
        # ~ 0.003 dB / 0.9144 km = 0.0033 dB/km
        self.assertLess(a, 0.05)

    def test_zero_or_negative_frequency_rejected(self):
        with self.assertRaises(ValueError):
            absorption_db_per_km(0.0)
        with self.assertRaises(ValueError):
            absorption_db_per_km(-100.0)

    def test_invalid_model_raises(self):
        with self.assertRaises(ValueError):
            absorption_db_per_km(1_000.0, model="bogus")


class TestTransmissionLoss(unittest.TestCase):
    def test_tl_at_1km_spherical_is_60db(self):
        """TL = 2 * 10*log10(1) + alpha*1 = 0 + alpha. For alpha~0.04 at 1 kHz, ~ 0.04 dB."""
        tl = total_transmission_loss_db(1000.0, 1_000.0, spreading=2.0)
        self.assertLess(tl, 1.0)
        self.assertGreater(tl, 0.0)

    def test_tl_at_10km_higher_than_1km(self):
        """TL must increase monotonically with distance."""
        tl1 = total_transmission_loss_db(1_000.0, 25_000.0)
        tl10 = total_transmission_loss_db(10_000.0, 25_000.0)
        self.assertLess(tl1, tl10)

    def test_spherical_grows_faster_than_cylindrical(self):
        """At any distance, spherical TL >= cylindrical TL (geom term)."""
        d = 5_000.0
        tl_cyl = total_transmission_loss_db(d, 25_000.0, spreading=1.0)
        tl_sph = total_transmission_loss_db(d, 25_000.0, spreading=2.0)
        self.assertGreater(tl_sph, tl_cyl)

    def test_invalid_distance_rejected(self):
        with self.assertRaises(ValueError):
            total_transmission_loss_db(0.0, 1_000.0)
        with self.assertRaises(ValueError):
            total_transmission_loss_db(-1.0, 1_000.0)


class TestMaxRange(unittest.TestCase):
    def test_max_range_decreases_with_frequency(self):
        """Higher frequency -> shorter range for the same link budget."""
        r_low = max_range_for_loss(80.0, 5_000.0)
        r_hi = max_range_for_loss(80.0, 50_000.0)
        self.assertGreater(r_low, r_hi)

    def test_max_range_increases_with_link_budget(self):
        r60 = max_range_for_loss(60.0, 25_000.0)
        r90 = max_range_for_loss(90.0, 25_000.0)
        self.assertGreater(r90, r60)

    def test_max_range_at_least_1km_for_low_freq_low_budget(self):
        """1 kHz @ 70 dB should easily reach > 1 km."""
        r = max_range_for_loss(70.0, 1_000.0)
        self.assertGreater(r, 1_000.0)


class TestBandSummary(unittest.TestCase):
    def test_band_summary_returns_n_points(self):
        rows = band_summary(n_points=32)
        self.assertEqual(len(rows), 32)

    def test_band_summary_log_spacing(self):
        """Frequencies should be log-uniformly spaced."""
        rows = band_summary(f_low_hz=100.0, f_high_hz=10_000.0, n_points=11)
        freqs = [r[0] for r in rows]
        # check log-uniform: successive ratios should be constant
        ratios = [freqs[i + 1] / freqs[i] for i in range(len(freqs) - 1)]
        for r in ratios:
            self.assertAlmostEqual(r, ratios[0], places=4)

    def test_band_summary_invalid_band(self):
        with self.assertRaises(ValueError):
            band_summary(f_low_hz=100.0, f_high_hz=50.0)
        with self.assertRaises(ValueError):
            band_summary(f_low_hz=0.0, f_high_hz=100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
