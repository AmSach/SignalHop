"""Tests for SignalHop mesh resilience simulator."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sim_resilience import FailureProfile, ResilientSimulator, default_failure_schedule
from sim_demo import SimConfig


def test_failure_profile_alive_default():
    p = FailureProfile({})
    assert p.is_alive(0, 0.0) is True
    assert p.is_alive(99, 1000.0) is True


def test_failure_profile_window():
    p = FailureProfile({1: (10.0, 20.0)})
    assert p.is_alive(1, 5.0) is True
    assert p.is_alive(1, 15.0) is False
    assert p.is_alive(1, 25.0) is True


def test_failure_profile_infinite_recovery():
    p = FailureProfile({2: (5.0, float('inf'))})
    assert p.is_alive(2, 6.0) is False
    assert p.is_alive(2, 9999.0) is False


def test_noise_db_increases_with_failure():
    p = FailureProfile({1: (10.0, 20.0)})
    assert p.noise_db(0, 5.0) == 0.0
    assert p.noise_db(1, 15.0) == 200.0  # saturated


def test_resilient_sim_runs():
    cfg = SimConfig(num_nodes=6, tx_range=20.0, simulation_time=10.0)
    profile = FailureProfile({1: (5.0, float('inf'))})
    sim = ResilientSimulator(cfg, profile)
    sim._seed = 42
    stats = sim.run_with_failures()
    assert 'delivery_rate_baseline' in stats
    assert 'delivery_rate_after_failures' in stats
    assert 0.0 <= stats['delivery_rate_baseline'] <= 1.0
    assert 0.0 <= stats['delivery_rate_after_failures'] <= 1.0


def test_default_schedule_has_failures():
    p = default_failure_schedule(8)
    schedule = p.fail_at
    assert len(schedule) > 0
    for nid, (fail_t, _) in schedule.items():
        assert fail_t > 0
