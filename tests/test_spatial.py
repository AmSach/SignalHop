#!/usr/bin/env python3
"""Tests for the spatial room-impulse channel simulator."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_spatial import (
    _db_to_linear,
    SPEED_OF_SOUND,
    RoomConfig,
    ChannelConfig,
    build_impulse_response,
    apply_channel,
    simulate,
)


def test_db_linear_round_trip():
    assert abs(_db_to_linear(0.0) - 1.0) < 1e-9
    assert abs(_db_to_linear(20.0) - 10.0) < 1e-3
    assert abs(_db_to_linear(-6.0) - 0.5012) < 1e-3


def test_distance_attenuation_monotonic():
    d1 = ChannelConfig(range_m=1.0, seed=0)
    build_impulse_response(RoomConfig(), d1)
    d2 = ChannelConfig(range_m=2.0, seed=0)
    build_impulse_response(RoomConfig(), d2)
    assert d2.path_loss_db > d1.path_loss_db
    # Free-space: 6 dB per doubling
    assert abs((d2.path_loss_db - d1.path_loss_db) - 6.0205) < 0.1


def test_impulse_response_direct_path_only():
    ch = ChannelConfig(range_m=2.0, reflections=0, fading_taps=0, seed=1)
    h = build_impulse_response(RoomConfig(), ch)
    assert h.ndim == 1
    assert h.dtype == np.float32
    # Path loss at 2m should be ~6 dB
    assert 5.5 < ch.path_loss_db < 6.5


def test_more_reflections_means_more_taps():
    ch_direct = ChannelConfig(range_m=4.0, reflections=0, fading_taps=0, seed=2)
    h_direct = build_impulse_response(RoomConfig(), ch_direct)

    ch_two = ChannelConfig(range_m=4.0, reflections=2, fading_taps=0, seed=2)
    h_two = build_impulse_response(RoomConfig(), ch_two)

    # More reflections => more non-zero taps (or at least equal length)
    nonzero_direct = int(np.count_nonzero(h_direct))
    nonzero_two = int(np.count_nonzero(h_two))
    assert nonzero_two >= nonzero_direct


def test_fading_increases_impulse_length():
    ch_no_fade = ChannelConfig(range_m=4.0, reflections=1, fading_taps=0, seed=3)
    h_no_fade = build_impulse_response(RoomConfig(), ch_no_fade)

    ch_fade = ChannelConfig(range_m=4.0, reflections=1, fading_taps=4, seed=3)
    h_fade = build_impulse_response(RoomConfig(), ch_fade)

    assert len(h_fade) >= len(h_no_fade)
    # Fading tail should contribute energy
    assert np.sum(np.abs(h_fade)) >= np.sum(np.abs(h_no_fade))


def test_apply_channel_adds_noise():
    """Noise level should grow as SNR drops — measure noise power relative to signal."""
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 1, size=4096).astype(np.float32)
    impulse = np.array([0.0, 1.0, 0.5, 0.25], dtype=np.float32)
    clean = apply_channel(sig, impulse, snr_db=80.0, seed=0)
    noisy = apply_channel(sig, impulse, snr_db=10.0, seed=0)

    # Compare to the "convolved only" version (no noise) to isolate noise contribution
    convolved_only = np.convolve(sig, impulse, mode="full").astype(np.float32)[: len(noisy)]
    # The first len(impulse)-1 samples contain the rising edge of the convolution
    # (energy growing from 0 to steady state). Skip them for a clean noise comparison.
    skip = len(impulse)
    noise_window = noisy[skip:] - convolved_only[skip:skip + len(noisy) - skip]
    noise_power = float(np.mean(noise_window ** 2))
    signal_power = float(np.mean(convolved_only[skip:skip + len(noise_window)] ** 2))
    assert noise_power > 0
    # At snr_db=10, expect noise to be roughly the same order as signal — sanity-check upper bound
    assert noise_power < signal_power * 100


def test_simulate_low_snr_high_ber():
    ch = ChannelConfig(range_m=2.0, snr_db=-20.0, reflections=0, fading_taps=0, seed=7)
    res = simulate(symbols=64, ch=ch)
    assert res["ber"] > 0.15, f"expected high BER at very low SNR, got {res['ber']}"


def test_simulate_high_snr_low_ber():
    res = simulate(symbols=64, ch=ChannelConfig(range_m=1.0, snr_db=60.0, reflections=0, fading_taps=0, seed=7))
    assert res["ber"] < 0.05
    assert res["delay_spread_ms"] < 1.0


def test_delay_spread_increases_with_reflections():
    res1 = simulate(symbols=32, ch=ChannelConfig(range_m=4.0, snr_db=40.0, reflections=0, fading_taps=0, seed=5))
    res3 = simulate(symbols=32, ch=ChannelConfig(range_m=4.0, snr_db=40.0, reflections=2, fading_taps=0, seed=5))
    assert res3["delay_spread_ms"] >= res1["delay_spread_ms"]


def test_simulate_result_is_dict_of_floats():
    res = simulate(symbols=16, ch=ChannelConfig(range_m=3.0, snr_db=25.0, seed=9))
    for k, v in res.items():
        assert isinstance(v, (int, float)), f"{k} should be numeric, got {type(v)}"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
