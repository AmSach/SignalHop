"""Smoke tests for the capacity sweep."""
import contextlib
import io
import sys

import pytest

sys.path.insert(0, ".")
from sim_capacity import run_once  # noqa: E402


def test_run_once_returns_routing_data():
    out = run_once(num_nodes=4, area=40.0, tx_range=15.0, seed=1)
    assert "delivered_rate" in out
    assert "avg_routes" in out
    assert 0.0 <= out["delivered_rate"] <= 100.0
    assert out["avg_routes"] >= 0.0


def test_run_once_is_quiet(capsys):
    """sim.run() prints a lot — make sure we suppress it."""
    run_once(num_nodes=4, area=40.0, tx_range=15.0, seed=2)
    captured = capsys.readouterr()
    assert "Simulation complete" not in captured.out
    assert "Packets sent" not in captured.out


def test_avg_routes_grows_with_node_count():
    small = run_once(num_nodes=4, area=80.0, tx_range=20.0, seed=3)
    large = run_once(num_nodes=24, area=80.0, tx_range=20.0, seed=4)
    assert large["avg_routes"] >= small["avg_routes"]


def test_routing_keys_present():
    out = run_once(num_nodes=4, area=40.0, tx_range=15.0, seed=5)
    for key in ("delivered", "sent"):
        assert key in out
        assert isinstance(out[key], int)
        assert out[key] >= 0
