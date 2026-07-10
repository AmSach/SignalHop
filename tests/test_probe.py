#!/usr/bin/env python3
"""Tests for core/probe.py (SignalHop Link Probe)."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.modem import AcousticModem, SAMPLE_RATE
from core.probe import LinkProbe, ProbeResult, _estimate_snr_db


def test_probe_init_defaults():
    p = LinkProbe()
    assert p.c == 343.0
    assert isinstance(p.modem, AcousticModem)
    print("  ✅ default init")


def test_probe_custom_speed_of_sound():
    # Underwater ~1500 m/s
    p = LinkProbe(speed_of_sound=1500.0)
    assert p.c == 1500.0
    print("  ✅ custom speed of sound")


def test_ping_pair_loopback_zero_distance():
    """When RX == TX (self loopback with no delay), distance should be ~0."""
    m = AcousticModem()
    p = LinkProbe(m)
    msg = b"hello"
    tx = m.tx(msg)
    rx = tx.copy()
    res = p.ping_pair(tx, rx)
    # No delay inserted, so RTT should be very small -> distance near 0
    assert res.distance_m < 5.0, f"expected near-zero distance, got {res.distance_m}"
    print(f"  ✅ loopback distance ~0 ({res.distance_m:.2f} m)")


def test_ping_pair_with_simulated_delay():
    """Insert 50ms of silence between our TX and the remote's echo.

    Realistic scenario: we TX, the remote re-modulates, and we hear it
    50ms later.  rx = [tx, 50ms_silence, echo_of_tx].
    """
    m = AcousticModem()
    p = LinkProbe(m)
    msg = b"delay test"
    tx = m.tx(msg)
    # 50ms delay @ 48kHz = 2400 samples
    delay_samples = int(0.05 * SAMPLE_RATE)
    silence = np.zeros(delay_samples, dtype=np.float32)
    echo = tx * 0.9  # remote re-emits, slightly attenuated
    rx = np.concatenate([tx, silence, echo])
    res = p.ping_pair(tx, rx)
    # Expected distance: 0.05s * 343 m/s / 2 = 8.575 m
    expected = (0.05 * 343.0) / 2.0
    assert abs(res.distance_m - expected) < 3.0, (
        f"expected ~{expected:.2f} m, got {res.distance_m:.2f} m"
    )
    print(f"  ✅ delay-based distance ({res.distance_m:.2f} m, expected ~{expected:.2f} m)")


def test_ping_pair_returns_probe_result():
    m = AcousticModem()
    p = LinkProbe(m)
    tx = m.tx(b"x")
    res = p.ping_pair(tx, tx.copy())
    assert isinstance(res, ProbeResult)
    assert res.rtt_s >= 0
    assert isinstance(res.snr_db, float)
    print("  ✅ returns ProbeResult dataclass")


def test_ping_local_zero_delay():
    m = AcousticModem()
    p = LinkProbe(m)
    samples = m.tx(b"self test")
    res = p.ping_local(samples)
    assert res.distance_m < 5.0
    print("  ✅ ping_local alias")


def test_probe_invalid_inputs():
    m = AcousticModem()
    p = LinkProbe(m)
    try:
        p.ping_pair(np.array([]), m.tx(b"x"))
        assert False, "should have raised"
    except ValueError:
        pass
    try:
        p.ping_pair(m.tx(b"x"), np.array([]))
        assert False, "should have raised"
    except ValueError:
        pass
    print("  ✅ invalid input validation")


def test_probe_result_to_dict():
    r = ProbeResult(distance_m=10.0, rtt_s=0.06, snr_db=12.5,
                    chirp_corr=0.8, peer_id=b"NODE0001")
    d = r.to_dict()
    assert d["distance_m"] == 10.0
    assert d["peer_id"] == "4e4f444530303031"  # hex of NODE0001
    # No peer_id case
    r2 = ProbeResult(distance_m=1.0, rtt_s=0.01, snr_db=10.0, chirp_corr=0.5)
    assert r2.to_dict()["peer_id"] is None
    print("  ✅ ProbeResult.to_dict")


def test_is_usable_thresholds():
    good = ProbeResult(distance_m=5.0, rtt_s=0.03, snr_db=15.0, chirp_corr=0.5)
    bad_snr = ProbeResult(distance_m=5.0, rtt_s=0.03, snr_db=2.0, chirp_corr=0.5)
    bad_corr = ProbeResult(distance_m=5.0, rtt_s=0.03, snr_db=15.0, chirp_corr=0.01)
    assert good.is_usable() is True
    assert bad_snr.is_usable() is False
    assert bad_corr.is_usable() is False
    # Custom thresholds
    assert good.is_usable(snr_min_db=20.0) is False
    print("  ✅ is_usable thresholds")


def test_snr_estimation_clean_vs_noisy():
    """A signal with low-amplitude noise first, then a strong tone, should
    have clearly positive SNR. Pure random noise should be ~0 dB."""
    sr = 48000
    t = np.arange(sr) / sr
    # Quiet noise first, then a strong tone
    noise_head = np.random.randn(sr // 4).astype(np.float32) * 0.01
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.5
    mixed = np.concatenate([noise_head, tone])
    snr_clean = _estimate_snr_db(mixed, noise_floor_samples=sr // 4)
    assert snr_clean > 15.0, f"tone-after-quiet should have high SNR, got {snr_clean:.1f} dB"
    # Pure random noise
    noise = np.random.randn(sr).astype(np.float32) * 0.1
    snr_noise = _estimate_snr_db(noise, noise_floor_samples=sr // 4)
    # Should be close to 0 dB (within ±3 dB)
    assert abs(snr_noise) < 3.0, f"pure noise should be ~0 dB, got {snr_noise}"
    print(f"  ✅ SNR estimation (clean={snr_clean:.1f} dB, noise={snr_noise:.1f} dB)")


def test_snr_short_signal_fallback():
    """Very short signals should not crash — return the fallback value."""
    short = np.array([0.1, 0.2, -0.1], dtype=np.float32)
    snr = _estimate_snr_db(short, noise_floor_samples=1000)
    assert snr == 3.0
    print("  ✅ SNR short-signal fallback")


if __name__ == "__main__":
    tests = [
        test_probe_init_defaults,
        test_probe_custom_speed_of_sound,
        test_ping_pair_loopback_zero_distance,
        test_ping_pair_with_simulated_delay,
        test_ping_pair_returns_probe_result,
        test_ping_local_zero_delay,
        test_probe_invalid_inputs,
        test_probe_result_to_dict,
        test_is_usable_thresholds,
        test_snr_estimation_clean_vs_noisy,
        test_snr_short_signal_fallback,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ❌ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} probe tests passed")
    sys.exit(0 if failed == 0 else 1)
