#!/usr/bin/env python3
"""Tests for sim_link_budget.py."""

import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import sim_link_budget as slb  # noqa: E402


def test_iso_absorption_in_known_range():
    # At 20 kHz, 20 C, 50% RH, ISO 9613-1 gives ~0.51 dB/m.
    # Our simplified model should land within 0.1-1.5 dB/m for the 18-22 kHz band.
    a = slb._absorp_coeff_db_per_m(20_000, 20, 50)
    assert 0.1 < a < 1.5, f"alpha at 20kHz/20C/50RH = {a}"


def test_absorption_increases_with_frequency():
    a18 = slb._absorp_coeff_db_per_m(18_000, 22, 50)
    a22 = slb._absorp_coeff_db_per_m(22_000, 22, 50)
    assert a22 > a18, "absorption must increase with frequency"


def test_absorption_humidity_dependence():
    # At 20 kHz, 20 C, going from 10% to 90% RH should not be monotonic
    # identically but it MUST change — and should be positive at all values.
    a_dry = slb._absorp_coeff_db_per_m(20_000, 20, 10)
    a_wet = slb._absorp_coeff_db_per_m(20_000, 20, 90)
    assert a_dry > 0 and a_wet > 0
    assert abs(a_dry - a_wet) > 1e-6, "humidity must affect absorption"


def test_link_budget_default_compute():
    env = slb.ENVIRONMENTS[3]  # office
    lb = slb.LinkBudget(range_m=5.0, temp_c=22, rel_humidity=50, env=env)
    lb.compute()
    # 5 m office: should be usable with comfortable margin
    assert lb.path_loss_db == 20 * math.log10(5.0)
    assert lb.absorption_db > 0
    assert lb.rx_spl_db < 100.0  # below TX level at 1 m
    assert lb.ber_estimate >= 0
    assert lb.is_usable(), f"5 m office should be usable, got BER={lb.ber_estimate}"


def test_link_budget_long_range_breaks():
    env = slb.ENVIRONMENTS[3]  # office
    lb = slb.LinkBudget(range_m=200.0, temp_c=22, rel_humidity=50, env=env)
    lb.compute()
    assert not lb.is_usable(), "200 m office should be unusable"
    assert lb.ber_estimate > 1e-2


def test_quiet_environment_outperforms_loud():
    quiet = slb.ENVIRONMENTS[1]  # quiet_office
    loud = slb.ENVIRONMENTS[5]   # industrial
    q = slb.LinkBudget(range_m=20, temp_c=22, rel_humidity=50, env=quiet)
    l = slb.LinkBudget(range_m=20, temp_c=22, rel_humidity=50, env=loud)
    q.compute(); l.compute()
    assert q.rx_snr_db > l.rx_snr_db
    assert q.ber_estimate < l.ber_estimate
    assert q.is_usable()
    assert not l.is_usable()


def test_temperature_extremes_still_compute():
    # ISO model should not blow up at temperature extremes.
    for T in [-5, 0, 25, 40, 50]:
        a = slb._absorp_coeff_db_per_m(19_000, T, 50)
        assert a > 0 and a < 10, f"alpha at T={T} unreasonable: {a}"


def test_minimum_range_does_not_crash():
    env = slb.ENVIRONMENTS[0]  # anechoic
    lb = slb.LinkBudget(range_m=0.1, temp_c=22, rel_humidity=50, env=env)
    lb.compute()
    assert lb.path_loss_db < 0  # 0.1 m is "closer" than 1 m ref
    assert lb.ber_estimate < 1e-2


def test_compare_includes_all_environments():
    # Make sure every env in the list produces a valid max range
    for env in slb.ENVIRONMENTS:
        lo, hi = 0.5, 100.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            lb = slb.LinkBudget(range_m=mid, temp_c=22, rel_humidity=50, env=env)
            lb.compute()
            if lb.is_usable():
                lo = mid
            else:
                hi = mid
        assert 0 < lo <= 100


def test_cli_sweep_runs():
    out = subprocess.run(
        [sys.executable, os.path.join(PARENT, "sim_link_budget.py"),
         "--sweep", "--environment", "office"],
        capture_output=True, text=True, timeout=15,
    )
    assert out.returncode == 0, f"stderr: {out.stderr}"
    assert "operational envelope" in out.stdout
    assert "rx_spl" in out.stdout


def test_cli_compare_runs():
    out = subprocess.run(
        [sys.executable, os.path.join(PARENT, "sim_link_budget.py"),
         "--compare-environments", "--temp", "20"],
        capture_output=True, text=True, timeout=20,
    )
    assert out.returncode == 0, f"stderr: {out.stderr}"
    assert "max_range_m" in out.stdout
    assert "industrial" in out.stdout


def test_cli_single_runs():
    out = subprocess.run(
        [sys.executable, os.path.join(PARENT, "sim_link_budget.py"),
         "--range", "3", "--environment", "home"],
        capture_output=True, text=True, timeout=10,
    )
    assert out.returncode == 0, f"stderr: {out.stderr}"
    assert "Link Budget" in out.stdout
    assert "BER est." in out.stdout


if __name__ == "__main__":
    print("running 12 tests...")
    failures = []
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {t.__name__}: {e}")
            failures.append(t.__name__)
    if failures:
        print(f"\n{len(failures)} failure(s): {failures}")
        sys.exit(1)
    print(f"\n{len(tests)} tests passed.")
