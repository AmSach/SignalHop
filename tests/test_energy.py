#!/usr/bin/env python3
"""Tests for SignalHop energy-aware duty-cycle scheduler."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_energy import (
    DutyCycleSchedule,
    SimResult,
    simulate,
    sweep_duty_cycles,
    LISTEN_CURRENT_MA,
    SLEEP_CURRENT_MA,
)


def test_listen_window_detection():
    s = DutyCycleSchedule(cycle_ms=100.0, listen_ms=20.0)
    # window is centered at t=0, half=10ms wide on each side
    assert s.is_listening(0.0) is True
    assert s.is_listening(5.0) is True
    assert s.is_listening(9.99) is True
    assert s.is_listening(15.0) is False
    assert s.is_listening(50.0) is False
    assert s.is_listening(95.0) is True  # wrap-around
    assert s.is_listening(99.99) is True
    print("  ✓ listen window detection")


def test_low_duty_yields_low_current():
    s = DutyCycleSchedule(cycle_ms=1000.0, listen_ms=10.0)
    r = simulate(s, duration_s=5.0, n_nodes=3, traffic_per_s=1.0, clock_drift_ppm=0.0, rng_seed=1)
    # 1% duty -> ~0.28mA, dominated by sleep
    assert r.mean_current_ma < 1.0
    assert r.mean_current_ma > SLEEP_CURRENT_MA
    print(f"  ✓ low duty -> near-sleep current ({r.mean_current_ma:.3f} mA)")


def test_high_duty_yields_high_pdr():
    s = DutyCycleSchedule(cycle_ms=100.0, listen_ms=80.0)
    r = simulate(s, duration_s=30.0, n_nodes=4, traffic_per_s=2.0, clock_drift_ppm=0.0, rng_seed=2)
    assert r.pdr > 0.5, f"high duty should give PDR > 0.5, got {r.pdr}"
    print(f"  ✓ high duty -> high PDR ({r.pdr*100:.1f}%)")


def test_low_duty_yields_low_pdr():
    s = DutyCycleSchedule(cycle_ms=1000.0, listen_ms=5.0)
    r = simulate(s, duration_s=30.0, n_nodes=4, traffic_per_s=2.0, clock_drift_ppm=0.0, rng_seed=3)
    assert r.pdr < 0.3, f"low duty should give PDR < 0.3, got {r.pdr}"
    print(f"  ✓ low duty -> low PDR ({r.pdr*100:.1f}%)")


def test_latency_bounded_by_cycle():
    cycle_ms = 200.0
    s = DutyCycleSchedule(cycle_ms=cycle_ms, listen_ms=50.0)
    r = simulate(s, duration_s=10.0, n_nodes=3, traffic_per_s=2.0, clock_drift_ppm=0.0, rng_seed=4)
    assert 0 <= r.mean_latency_ms <= cycle_ms * 1.5
    print(f"  ✓ latency within one cycle ({r.mean_latency_ms:.0f}ms / {cycle_ms}ms)")


def test_clock_drift_degrades_pdr():
    s = DutyCycleSchedule(cycle_ms=100.0, listen_ms=20.0)
    r_low = simulate(s, duration_s=60.0, n_nodes=5, traffic_per_s=2.0, clock_drift_ppm=10.0, rng_seed=5)
    r_high = simulate(s, duration_s=60.0, n_nodes=5, traffic_per_s=2.0, clock_drift_ppm=200.0, rng_seed=5)
    # High drift should not help; should be same or worse
    assert r_high.pdr <= r_low.pdr + 0.05
    print(f"  ✓ high drift doesn't help PDR ({r_low.pdr*100:.0f}% vs {r_high.pdr*100:.0f}%)")


def test_battery_life_scales_inversely_with_current():
    battery_mah = 2000.0  # a 2000 mAh Li-ion
    s_low = DutyCycleSchedule(cycle_ms=2000.0, listen_ms=5.0)
    s_high = DutyCycleSchedule(cycle_ms=100.0, listen_ms=80.0)
    r_low = simulate(s_low, duration_s=10.0, n_nodes=3, traffic_per_s=1.0, clock_drift_ppm=0.0, rng_seed=6)
    r_high = simulate(s_high, duration_s=10.0, n_nodes=3, traffic_per_s=1.0, clock_drift_ppm=0.0, rng_seed=6)
    life_low = battery_mah / max(r_low.mean_current_ma, 1e-6)
    life_high = battery_mah / max(r_high.mean_current_ma, 1e-6)
    assert life_low > life_high * 5
    print(f"  ✓ low duty -> {life_low:.0f}h life vs {life_high:.0f}h at high duty")


def test_sweep_returns_one_row_per_duty():
    rows = sweep_duty_cycles(cycles=[100.0, 200.0, 500.0, 1000.0], listen_ms=20.0, duration_s=5.0, clock_drift_ppm=0.0)
    assert len(rows) == 4
    assert all(isinstance(r, SimResult) for r in rows)
    assert all(r.cycle_ms in {100.0, 200.0, 500.0, 1000.0} for r in rows)
    print(f"  ✓ sweep returns {len(rows)} rows")


def test_deterministic_with_seed():
    s = DutyCycleSchedule(cycle_ms=200.0, listen_ms=30.0)
    r1 = simulate(s, duration_s=10.0, n_nodes=4, traffic_per_s=2.0, clock_drift_ppm=50.0, rng_seed=42)
    r2 = simulate(s, duration_s=10.0, n_nodes=4, traffic_per_s=2.0, clock_drift_ppm=50.0, rng_seed=42)
    assert r1.pdr == r2.pdr
    assert r1.frames_received == r2.frames_received
    print("  ✓ deterministic with seed")


def main() -> int:
    tests = [
        test_low_duty_yields_low_current,
        test_high_duty_yields_high_pdr,
        test_low_duty_yields_low_pdr,
        test_latency_bounded_by_cycle,
        test_clock_drift_degrades_pdr,
        test_battery_life_scales_inversely_with_current,
        test_listen_window_detection,
        test_sweep_returns_one_row_per_duty,
        test_deterministic_with_seed,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    if failed:
        print(f"\nFAILED: {failed}/{len(tests)}")
        return 1
    print(f"\nAll {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
