"""Tests for sim_battery.py"""
import math
import pytest
from sim_battery import (
    DutyCycle, NodePowerModel, battery_lifetime_hours,
    lifetime_summary, typical_chat_duty, BATTERY_PRESETS,
)


def test_duty_cycle_rejects_bad_sum():
    with pytest.raises(ValueError):
        DutyCycle(0.5, 0.5, 0.5, 0.0)


def test_duty_cycle_accepts_valid_sum():
    d = DutyCycle(0.7, 0.2, 0.05, 0.05)
    assert d.sleep_pct == 0.7


def test_avg_current_weighted():
    p = NodePowerModel(sleep_ma=1.0, rx_ma=10.0, tx_ma=100.0, cpu_active_ma=5.0)
    d = DutyCycle(0.5, 0.3, 0.1, 0.1)
    expected = 0.5*1.0 + 0.3*10.0 + 0.1*100.0 + 0.1*5.0
    assert abs(p.avg_current_ma(d) - expected) < 1e-9


def test_lifetime_inverse_proportional():
    assert battery_lifetime_hours(1000, 1.0) == 1000.0
    assert battery_lifetime_hours(1000, 10.0) == 100.0


def test_lifetime_zero_current_is_infinite():
    assert battery_lifetime_hours(1000, 0.0) == float("inf")


def test_summary_keys():
    s = lifetime_summary(500, 0.5, DutyCycle(0.8, 0.1, 0.05, 0.05), 10)
    assert "lifetime_hours" in s
    assert "lifetime_days" in s
    assert s["total_packets_in_lifetime"] == pytest.approx(10 * 1000.0)


def test_typical_chat_duty_sums_to_one():
    d = typical_chat_duty(beacon_interval_s=5.0, chat_bursts_per_hour=12)
    total = d.sleep_pct + d.rx_pct + d.tx_pct + d.cpu_active_pct
    assert abs(total - 1.0) < 0.01


def test_typical_chat_duty_mostly_sleeping():
    d = typical_chat_duty(beacon_interval_s=5.0, chat_bursts_per_hour=12)
    assert d.sleep_pct > 0.8


def test_battery_presets_have_capacity():
    for name, preset in BATTERY_PRESETS.items():
        assert preset["capacity_mah"] > 0
        assert preset["nominal_v"] > 0


def test_higher_tx_power_shorter_lifetime():
    low = NodePowerModel(0.02, 40.0, 60.0, 30.0)
    high = NodePowerModel(0.02, 40.0, 200.0, 30.0)
    d = DutyCycle(0.85, 0.10, 0.03, 0.02)
    assert low.avg_current_ma(d) < high.avg_current_ma(d)


def test_main_runs(capsys):
    import sys
    from sim_battery import main
    old = sys.argv
    try:
        sys.argv = ["sim_battery", "--battery", "cr2032"]
        assert main() == 0
    finally:
        sys.argv = old
    out = capsys.readouterr().out
    assert "Lifetime:" in out
    assert "Avg current" in out
