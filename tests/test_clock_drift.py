"""Tests for SignalHop clock-drift TDMA collision sim."""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Make sim modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_clock_drift import NodeClock, simulate, sweep  # noqa: E402


def test_zero_drift_no_collisions():
    """Zero drift, frequent sync -> essentially no collisions over short run."""
    rep = simulate(
        num_nodes=3,
        duration_s=10.0,
        sync_interval_s=1.0,
        drift_ppm_std=0.0,
        seed=1,
    )
    # With perfect clocks and frequent sync, collision count is tiny/zero.
    assert rep.collisions <= 2, f"expected <=2 collisions, got {rep.collisions}"
    assert rep.delivery_rate >= 0.9


def test_high_drift_produces_collisions():
    """High drift with infrequent sync should produce collisions."""
    rep = simulate(
        num_nodes=4,
        duration_s=600.0,
        sync_interval_s=600.0,  # no resync during the run
        drift_ppm_std=2000.0,  # large std so tail nodes drift past guard
        seed=2,
    )
    assert rep.collisions > 10, f"expected many collisions, got {rep.collisions}"
    assert rep.delivery_rate < 1.0


def test_frequent_sync_outperforms_infrequent():
    """More frequent sync should give higher delivery rate."""
    rep_often = simulate(
        num_nodes=5,
        duration_s=300.0,
        sync_interval_s=10.0,
        drift_ppm_std=50.0,
        seed=3,
    )
    rep_rare = simulate(
        num_nodes=5,
        duration_s=300.0,
        sync_interval_s=120.0,
        drift_ppm_std=50.0,
        seed=3,
    )
    assert rep_often.delivery_rate >= rep_rare.delivery_rate


def test_node_clock_drift_accumulates():
    """A 1000 ppm clock drifts ~1 ms per second."""
    clk = NodeClock(node_id=1, drift_ppm=1000.0)
    clk.resync(0.0)
    # After 1 second of master time, local time should be ~1 ms ahead of master.
    local = clk.local_t(1000.0)
    assert 1000.9 <= local <= 1001.1, f"expected ~1001.0 ms local, got {local}"


def test_node_clock_resync_zeros_offset():
    """Resync should bring local back to master."""
    clk = NodeClock(node_id=1, drift_ppm=100.0)
    clk.resync(0.0)
    # After 5s, drift accumulates
    drifted = clk.local_t(5000.0)
    assert drifted != 5000.0
    # Resync
    clk.resync(5000.0)
    assert abs(clk.local_t(5000.0) - 5000.0) < 0.01


def test_sweep_returns_all_intervals():
    """sweep() should return one report per requested interval."""
    intervals = [1.0, 5.0, 30.0]
    results = sweep(intervals, num_nodes=3, duration_s=10.0, drift_ppm_std=20.0)
    assert len(results) == len(intervals)
    seen = {si for si, _ in results}
    assert seen == set(intervals)


def test_empty_simulation_has_no_collisions():
    """Zero-duration simulation should produce no events."""
    rep = simulate(
        num_nodes=4,
        duration_s=0.0,
        sync_interval_s=1.0,
        drift_ppm_std=30.0,
        seed=4,
    )
    assert rep.collisions == 0
    assert rep.total_transmissions == 0
    assert rep.delivery_rate == 1.0  # vacuously true


def test_seed_reproducibility():
    """Same seed should produce identical reports."""
    rep_a = simulate(num_nodes=4, duration_s=30.0, sync_interval_s=10.0, seed=99)
    rep_b = simulate(num_nodes=4, duration_s=30.0, sync_interval_s=10.0, seed=99)
    assert rep_a.collisions == rep_b.collisions
    assert rep_a.delivery_rate == rep_b.delivery_rate
    assert rep_a.worst_overlap_ms == rep_b.worst_overlap_ms
