#!/usr/bin/env python3
"""Tests for sim_ranging_error.py."""

import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import sim_ranging_error as sre  # noqa: E402


def test_speed_of_sound_at_20c():
    # 331.3 + 0.606 * 20 = 343.42
    c = sre.speed_of_sound(20.0)
    assert 343.0 < c < 344.0, f"c(20C) = {c}"


def test_speed_of_sound_increases_with_temp():
    assert sre.speed_of_sound(30.0) > sre.speed_of_sound(10.0)


def test_clock_skew_zero_ppm():
    assert sre.clock_skew_ppm_to_phase_error_m(0.0, 10.0, 0.0) == 0.0


def test_clock_skew_sign_and_scale():
    # 100 ppm on 10 m: 100e-6 * 2 * 10 = 2 mm
    e = sre.clock_skew_ppm_to_phase_error_m(100.0, 10.0, 0.0)
    assert abs(e - 0.002) < 1e-9, f"clock skew bias = {e}"


def test_detection_jitter_decreases_with_snr():
    s_low = sre.detection_jitter_std_m(5.0, 2000.0)
    s_hi = sre.detection_jitter_std_m(25.0, 2000.0)
    assert s_low > s_hi, "higher SNR must give tighter detection"
    assert s_hi > 0


def test_detection_jitter_below_zero_db_is_huge():
    s = sre.detection_jitter_std_m(-5.0, 2000.0)
    assert math.isinf(s), "below 0 dB SNR the estimator variance diverges"


def test_multipath_bias_zero_when_reflection_very_weak():
    # 30 dB weaker reflection - bias should be tiny
    b = sre.multipath_bias_m(6.0, 1.5, 30.0)
    assert b < 0.01, f"weak reflection bias = {b}"


def test_multipath_bias_approaches_extra_path_for_dominant_reflection():
    # -6 dB means reflection is LOUDER than direct. bias -> extra path.
    b = sre.multipath_bias_m(6.0, 1.5, -6.0)
    assert b > 1.0, f"strong reflection should give large bias, got {b}"


def test_nlos_flag_threshold():
    assert sre.is_likely_nlos(2.0) is True
    assert sre.is_likely_nlos(2.0, threshold_db=3.0) is True
    assert sre.is_likely_nlos(15.0) is False


def test_simulate_one_no_errors_zero_bias():
    cfg = sre.RangingConfig(
        true_range_m=5.0,
        temp_c=20.0,        # no temp error
        assumed_temp_c=20.0,
        clock_ppm=0.0,      # no clock error
        snr_db=40.0,        # very high SNR, tiny jitter
        chirp_bandwidth_hz=5000.0,
        nlos=False,
        reflection_strength_db=60.0,  # very weak reflection
    )
    # Run a few times - mean error should be near zero
    import random
    rng = random.Random(0)
    errs = [sre.simulate_one_measurement(cfg, rng).error_m for _ in range(50)]
    mean = sum(errs) / len(errs)
    assert abs(mean) < 0.005, f"near-ideal mean error = {mean}"


def test_simulate_los_pure_bias():
    cfg = sre.RangingConfig(
        true_range_m=10.0,
        temp_c=20.0,
        assumed_temp_c=20.0,
        clock_ppm=0.0,
        snr_db=40.0,
        chirp_bandwidth_hz=10_000.0,
        nlos=False,
        reflection_extra_path_m=2.0,
        reflection_strength_db=0.0,  # equal strength - half the bias
    )
    # bias = 2.0 * 1/(1+1) = 1.0 m
    # with no other errors, error = 1.0 m exactly
    import random
    rng = random.Random(0)
    r = sre.simulate_one_measurement(cfg, rng)
    assert abs(r.error_m - 1.0) < 0.05, f"LOS bias case error = {r.error_m}"


def test_simulate_nlos_bias_equals_extra_path():
    cfg = sre.RangingConfig(
        true_range_m=8.0,
        temp_c=20.0,
        assumed_temp_c=20.0,
        clock_ppm=0.0,
        snr_db=40.0,
        chirp_bandwidth_hz=10_000.0,
        nlos=True,
        reflection_extra_path_m=2.5,
    )
    import random
    rng = random.Random(0)
    r = sre.simulate_one_measurement(cfg, rng)
    # In pure NLOS, measured = direct + extra
    assert abs(r.error_m - 2.5) < 0.05, f"NLOS bias = {r.error_m}"


def test_campaign_summary_keys():
    cfg = sre.RangingConfig(true_range_m=6.0, seed=1)
    _, summary = sre.simulate_campaign(cfg, n_measurements=200)
    for k in ("n", "true_range_m", "mean_error_m", "median_error_m",
              "std_error_m", "rmse_m", "min_error_m", "max_error_m", "nlos_fraction"):
        assert k in summary, f"missing key {k}"
    assert summary["n"] == 200
    assert summary["rmse_m"] >= abs(summary["mean_error_m"]) - 1e-9


def test_campaign_high_snr_low_error():
    cfg = sre.RangingConfig(
        true_range_m=4.0,
        snr_db=40.0,
        chirp_bandwidth_hz=10_000.0,
        clock_ppm=0.0,
        assumed_temp_c=22.0,  # matches default
        nlos=False,
        reflection_strength_db=60.0,  # suppress multipath
        seed=7,
    )
    _, summary = sre.simulate_campaign(cfg, n_measurements=500)
    # RMSE should be small (sub-cm) for this ideal-ish config
    assert summary["rmse_m"] < 0.05, f"high-SNR RMSE = {summary['rmse_m']}"


def test_cli_runs():
    """Smoke test: CLI produces a summary line."""
    result = subprocess.run(
        [sys.executable, os.path.join(PARENT, "sim_ranging_error.py"),
         "--range", "5", "--n", "50", "--snr-db", "20", "--seed", "1"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Mean error" in result.stdout
    assert "RMSE" in result.stdout
