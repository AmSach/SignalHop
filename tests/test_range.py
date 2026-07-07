#!/usr/bin/env python3
"""Tests for sim_range.py — physical layer range sweep."""
import math
import random
import sys

import sim_range


def test_preamble_length_constant():
    assert sim_range.PREAMBLE_LEN > 0
    assert sim_range.PREAMBLE_LEN == sim_range.SAMPLES_PER_SYMBOL * 4


def test_geometric_spreading_at_reference_is_zero():
    assert sim_range.geometric_spreading_db(0.1) == 0.0
    # 10x reference distance = 20 dB
    assert math.isclose(sim_range.geometric_spreading_db(1.0), 20.0, rel_tol=1e-6)


def test_atmospheric_absorption_grows_with_distance():
    a1 = sim_range.atmospheric_absorption_db(1.0, 18_000.0)
    a10 = sim_range.atmospheric_absorption_db(10.0, 18_000.0)
    assert a10 > a1 > 0


def test_snr_decreases_with_distance():
    s1 = sim_range.received_snr_db(1.0, ambient_noise_db_spl=40.0)
    s10 = sim_range.received_snr_db(10.0, ambient_noise_db_spl=40.0)
    assert s1 > s10


def test_snr_decreases_with_noise():
    s_quiet = sim_range.received_snr_db(5.0, ambient_noise_db_spl=20.0)
    s_loud = sim_range.received_snr_db(5.0, ambient_noise_db_spl=80.0)
    assert s_quiet > s_loud


def test_encode_decode_no_noise_roundtrip():
    """With infinite SNR the frame must decode perfectly."""
    tx = sim_range.encode_frame(b"hello world")
    ok, payload = sim_range.decode_frame(tx)
    assert ok, "no-noise roundtrip must succeed"
    assert payload == b"hello world"


def test_close_range_high_snr_round_trip():
    """At very short range and high SNR, frames should round-trip reliably."""
    rng = random.Random(7)
    res = sim_range.sweep_one(
        distance_m=0.5,
        noise_db_spl=20.0,
        trials=10,
        payload=b"SIGNALHOP PKT",
        tx_spl=100.0,
        rng=rng,
    )
    assert res.frame_success_rate >= 0.9, (
        f"close-range high-SNR should be near-perfect, got {res.frame_success_rate}"
    )


def test_far_range_low_snr_mostly_fails():
    """At 50m with loud noise, we expect the link to fail most of the time."""
    rng = random.Random(11)
    res = sim_range.sweep_one(
        distance_m=50.0,
        noise_db_spl=80.0,
        trials=10,
        payload=b"x" * 16,
        tx_spl=80.0,
        rng=rng,
    )
    assert res.frame_success_rate <= 0.2, (
        f"far/loud link should be unreliable, got {res.frame_success_rate}"
    )


def test_run_sweep_returns_full_grid():
    distances = [1.0, 2.0, 5.0]
    noises = [20.0, 40.0]
    results = sim_range.run_sweep(
        distances=distances,
        noise_levels=noises,
        trials=2,
        payload=b"x" * 4,
        tx_spl=90.0,
        seed=99,
    )
    assert len(results) == len(distances) * len(noises)
    seen = {(r.distance_m, r.noise_db_spl) for r in results}
    expected = {(d, n) for d in distances for n in noises}
    assert seen == expected


def test_render_table_includes_every_cell():
    results = [
        sim_range.SweepResult(1.0, 20.0, 30.0, 0.0, 1.0),
        sim_range.SweepResult(1.0, 40.0, 10.0, 0.5, 0.5),
        sim_range.SweepResult(2.0, 20.0, 25.0, 0.1, 0.8),
        sim_range.SweepResult(2.0, 40.0, 5.0, 0.9, 0.0),
    ]
    table = sim_range.render_table(results)
    assert "d=" in table
    assert "n=" in table
    # header has two noise levels -> two columns of "n=..."
    assert table.count("n=") == 2


def test_render_envelope_groups_by_noise():
    results = [
        sim_range.SweepResult(1.0, 20.0, 30.0, 0.0, 1.0),
        sim_range.SweepResult(5.0, 20.0, 20.0, 0.1, 0.99),
        sim_range.SweepResult(10.0, 20.0, 15.0, 0.4, 0.0),
        sim_range.SweepResult(1.0, 80.0, -20.0, 1.0, 0.0),
    ]
    env = sim_range.render_envelope(results, threshold=0.9)
    assert "noise=20" in env
    assert "noise=80" in env
    # the loudest noise band should report "<no range>"
    assert "<no range>" in env


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{failed} test(s) failed")
        sys.exit(1)
    print(f"\nAll {sum(1 for n in globals() if n.startswith('test_'))} tests passed")
