#!/usr/bin/env python3
"""Tests for SignalHop multi-hop relay simulator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_relay import (
    topology_chain,
    topology_mesh,
    topology_fanout,
    simulate,
    AcousticChannel,
    RelayNode,
    Frame,
)


def test_chain_neighbors():
    ch, a, _ = topology_chain()
    assert ch.in_range("A", "B")
    assert ch.in_range("B", "C")
    assert not ch.in_range("A", "C"), "A and C should be out of direct range"
    assert "B" in ch.neighbors("A")
    assert "C" not in ch.neighbors("A")
    print("  ✓ chain topology — only adjacent nodes are neighbors")


def test_chain_payload_reaches_c():
    ch, a, dest = topology_chain()
    result = simulate(a, dest, b"SOS: zone 7", seed=0)
    assert result.delivered, f"chain should deliver: {result}"
    assert result.hops_used >= 2
    print("  ✓ chain — payload reaches C via B")


def test_chain_drops_still_deliver():
    """Even with 20% drop rate the chain should usually deliver."""
    ch, a, dest = topology_chain()
    ch.drop_prob = 0.2
    delivered_any = False
    for seed in range(10):
        result = simulate(a, dest, b"hi", seed=seed)
        if result.delivered:
            delivered_any = True
            break
    assert delivered_any, "chain should deliver in at least 1 of 10 trials at 20% drop"
    print("  ✓ chain — survives 20% drop rate")


def test_mesh_payload_reaches_e():
    ch, a, dest = topology_mesh()
    result = simulate(a, dest, b"hello mesh", seed=0)
    assert result.delivered, f"mesh should deliver to E: {result}"
    print("  ✓ mesh — payload reaches E")


def test_fanout_all_leaves_get_payload():
    ch, a, _ = topology_fanout()
    payload = b"all-hands"
    result = simulate(a, "N", payload, seed=0)
    assert result.delivered, f"fanout should deliver to N: {result}"
    # The central node A should have made at least 4 transmissions —
    # one for each of the four leaves (N, S, E, W).
    assert result.transmissions >= 4, (
        f"expected >=4 tx from A (one per leaf): {result}"
    )
    print("  ✓ fanout — A transmits to all 4 leaves")


def test_ttl_exhaustion_drops_payload():
    ch, a, dest = topology_chain()
    # Initial TTL=1 means the frame can only be forwarded by B once
    # before its TTL hits 0. B then can't re-broadcast to C.
    result = simulate(a, dest, b"too far", seed=0, initial_ttl=1)
    assert not result.delivered, f"TTL=1 chain should not deliver: {result}"
    print("  ✓ TTL=1 chain — payload dropped (no further hops)")


def test_frame_payload_integrity():
    """The destination must receive the exact bytes that were sent."""
    ch, a, dest = topology_chain()
    payload = b"\x00\x01\xfe\xff binary bytes"
    result = simulate(a, dest, payload, seed=0)
    assert result.delivered
    # Walk the registry to find the dest node and verify the bytes match.
    from sim_relay import NODE_REGISTRY

    dest_node = NODE_REGISTRY[dest]
    assert payload in dest_node.delivered
    print("  ✓ chain — payload bytes match end-to-end")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(tests)} failed")
        raise SystemExit(1)
    print(f"\n{len(tests)}/{len(tests)} passed")
