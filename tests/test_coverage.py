"""Tests for sim_coverage.py coverage estimator."""
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_coverage import estimate_coverage, sweep_radius, _hop_reachable, _dist2d


def test_dist2d():
    assert _dist2d((0, 0), (3, 4)) == 5.0
    assert _dist2d((1.5, 2.5), (1.5, 2.5)) == 0.0
    assert math.isclose(_dist2d((0, 0), (1, 1)), math.sqrt(2))


def test_hop_reachable_zero_hops():
    reached = _hop_reachable([(0, 0), (10, 0), (20, 0)], src=0, radius=5, hops=0)
    assert reached == {0}


def test_hop_reachable_one_hop_isolated():
    # only self is reachable when nothing is within radius
    positions = [(0, 0), (100, 100), (200, 200)]
    reached = _hop_reachable(positions, src=0, radius=10, hops=3)
    assert reached == {0}


def test_hop_reachable_two_hops_chain():
    # chain: 0 -- 5 -- 10 -- 15, radius 6 should reach all in 3 hops
    positions = [(float(i * 5), 0.0) for i in range(4)]
    reached = _hop_reachable(positions, src=0, radius=6.0, hops=3)
    assert reached == {0, 1, 2, 3}


def test_hop_reachable_two_hops_too_far():
    # chain with gap that's too big for 2 hops
    positions = [(0.0, 0.0), (5.0, 0.0), (15.0, 0.0), (20.0, 0.0)]
    reached = _hop_reachable(positions, src=0, radius=6.0, hops=2)
    # 0 -> 1 (dist 5, ok), 1 -> 2 (dist 10, NOT ok with radius 6)
    assert reached == {0, 1}


def test_estimate_dense_area_covers_everything():
    # tight cluster: every node is within radius of at least one neighbor
    res = estimate_coverage(area_m=10, n_nodes=20, radius_m=10, hops=3,
                            trials=30, seed=1)
    # in a 10x10 with radius 10, one hop should reach almost everyone
    assert res.reach_mean >= 18.0
    assert 0.0 <= res.reach_fraction <= 1.0
    assert res.reach_ci95_lo <= res.reach_mean <= res.reach_ci95_hi


def test_estimate_sparse_area_partial_coverage():
    # sparse: large area, small radius -> only a fraction covered
    res = estimate_coverage(area_m=500, n_nodes=30, radius_m=20, hops=2,
                            trials=50, seed=7)
    assert 1 < res.reach_mean < 30
    assert res.reach_stdev >= 0.0


def test_estimate_more_hops_helps():
    r1 = estimate_coverage(area_m=300, n_nodes=40, radius_m=30, hops=1,
                           trials=100, seed=42)
    r3 = estimate_coverage(area_m=300, n_nodes=40, radius_m=30, hops=3,
                           trials=100, seed=42)
    assert r3.reach_mean > r1.reach_mean


def test_estimate_larger_radius_helps():
    r_small = estimate_coverage(area_m=200, n_nodes=30, radius_m=10, hops=2,
                                trials=80, seed=5)
    r_big = estimate_coverage(area_m=200, n_nodes=30, radius_m=50, hops=2,
                              trials=80, seed=5)
    assert r_big.reach_mean > r_small.reach_mean


def test_estimate_seed_reproducible():
    a = estimate_coverage(100, 25, 20, 2, trials=50, seed=99)
    b = estimate_coverage(100, 25, 20, 2, trials=50, seed=99)
    assert a.reach_mean == b.reach_mean
    assert a.reach_stdev == b.reach_stdev


def test_estimate_invalid_inputs():
    import pytest
    try:
        estimate_coverage(100, 1, 10, 1, trials=10)
    except ValueError:
        pass
    else:
        assert False, "expected ValueError for n_nodes < 2"

    try:
        estimate_coverage(0, 10, 10, 1, trials=10)
    except ValueError:
        pass
    else:
        assert False, "expected ValueError for area_m <= 0"

    try:
        estimate_coverage(100, 10, 10, 0, trials=10)
    except ValueError:
        pass
    else:
        assert False, "expected ValueError for hops < 1"


def test_sweep_radius():
    results = sweep_radius(area_m=200, n_nodes=30, radii=[10, 20, 40],
                           hops=2, trials=40, seed=3)
    assert len(results) == 3
    # monotonically increasing with radius
    assert results[0].reach_mean <= results[1].reach_mean <= results[2].reach_mean
