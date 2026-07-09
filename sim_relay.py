#!/usr/bin/env python3
"""
SignalHop — Multi-hop relay simulator.

Walks a payload through a chain or partial-mesh of acoustic nodes where
some pairs are out of direct range. The simulation is deterministic
(seeded), zero-dependency, and runs without any audio hardware.

This file is the counterpart to sim_range.py / sim_tdma.py. Those stress
the physical layer (SNR, range) and the MAC layer (slot scheduling).
This one stresses the network layer — TTL decrement, hop counting,
out-of-range fall-back routing.

The model is intentionally simple:

  * Each node sits at an (x, y) coordinate in arbitrary "units".
  * The AcousticChannel decides which node pairs can hear each other
    using a hard range cutoff (no diffraction, no multipath).
  * `step()` does one round of broadcasts: every node that has a frame
    to send announces it once; every in-range neighbor either accepts
    (and re-broadcasts) or ignores it (already seen / not the dest).

Run from the repo root:

    python3 sim_relay.py --payload "SOS: zone 7" --topology chain
    python3 sim_relay.py --payload "hello"      --topology mesh
    python3 sim_relay.py --payload "alpha"      --topology 5node
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Channel model
# ---------------------------------------------------------------------------
@dataclass
class AcousticChannel:
    """Symmetric acoustic link with a hard range limit.

    Two nodes can exchange frames if and only if they are within
    `max_range_units` of each other. Distance is interpreted in arbitrary
    "units" — a stand-in for meters in the field.
    """

    max_range_units: float = 10.0
    drop_prob: float = 0.05
    positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def add(self, node_id: str, position: Tuple[float, float]) -> None:
        self.positions[node_id] = position

    def distance(self, a: str, b: str) -> float:
        ax, ay = self.positions[a]
        bx, by = self.positions[b]
        dx, dy = ax - bx, ay - by
        return (dx * dx + dy * dy) ** 0.5

    def neighbors(self, node: str) -> List[str]:
        return [
            other
            for other in self.positions
            if other != node and self.distance(node, other) <= self.max_range_units
        ]

    def in_range(self, a: str, b: str) -> bool:
        if a not in self.positions or b not in self.positions:
            return False
        return self.distance(a, b) <= self.max_range_units


# ---------------------------------------------------------------------------
# Frame and node
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    """A single hop-by-hop frame: source, dest, ttl, payload."""

    source: str
    dest: str
    ttl: int
    payload: bytes

    def key(self) -> Tuple[str, int, int]:
        """Identity used to dedupe so we never re-broadcast the same frame."""
        return (self.source, id(self.payload), self.ttl)


@dataclass
class RelayNode:
    """A node that can hold a pending frame and broadcast it once per step."""

    node_id: str
    position: Tuple[float, float]
    channel: AcousticChannel
    delivered: List[bytes] = field(default_factory=list)
    sent_frames: List[Frame] = field(default_factory=list)
    received_frames: List[Frame] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.channel.add(self.node_id, self.position)

    def is_dest(self, frame: Frame) -> bool:
        return frame.dest == self.node_id

    def accept(self, frame: Frame) -> None:
        """Receive a frame. If it's for us, deliver. Otherwise, queue for re-broadcast."""
        self.received_frames.append(frame)
        if self.is_dest(frame):
            self.delivered.append(frame.payload)
            return
        if frame.ttl <= 0:
            return
        # Decrement TTL and queue for the next step.
        self.sent_frames.append(
            Frame(
                source=frame.source,
                dest=frame.dest,
                ttl=frame.ttl - 1,
                payload=frame.payload,
            )
        )


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------
@dataclass
class RelayResult:
    """Outcome of a relay run: did the destination receive the payload?"""

    delivered: bool
    hops_used: int
    final_ttl: int
    transmissions: int
    nodes_visited: List[str]


def simulate(
    source: RelayNode,
    dest_id: str,
    payload: bytes,
    max_steps: int = 16,
    initial_ttl: int = 8,
    seed: int = 0,
) -> RelayResult:
    """Walk a payload from `source` to `dest_id` through the network.

    Each step advances one round: every node with a pending frame
    broadcasts it once. The simulation terminates as soon as the
    destination accepts the payload, or when no node has a frame to
    send, or after `max_steps`.
    """
    rng = random.Random(seed)
    seen_keys: Set[Tuple[str, int, int]] = set()
    transmissions = 0
    nodes_visited: List[str] = []

    # Seed: source puts the frame in its own queue.
    source.sent_frames.append(
        Frame(source=source.node_id, dest=dest_id, ttl=initial_ttl, payload=payload)
    )

    # Track delivery as soon as it happens, but keep running the loop
    # for `max_steps` so other nodes (in fanout topologies, etc.) still
    # get a chance to hear the broadcast. This matches the model
    # described in the module docstring: every node with a frame
    # broadcasts it once per step.
    delivered = False
    for step in range(max_steps):
        any_broadcast = False
        for node in list(NODE_REGISTRY.values()):
            if not node.sent_frames:
                continue
            for frame in list(node.sent_frames):
                # Drop frames whose TTL has already expired — they must
                # not be forwarded or counted as transmissions.
                if frame.ttl <= 0:
                    continue
                key = frame.key()
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                if rng.random() < node.channel.drop_prob:
                    continue  # lossy link: frame disappears
                transmissions += 1
                nodes_visited.append(node.node_id)
                any_broadcast = True
                # Every in-range neighbor hears the frame.
                for neighbor_id in node.channel.neighbors(node.node_id):
                    neighbor = NODE_REGISTRY[neighbor_id]
                    neighbor.accept(frame)
                    if neighbor.is_dest(frame) and not delivered:
                        delivered = True
            # Clear the broadcast queue; any forwarded frames are now
            # sitting in neighbors' sent_frames for the next round.
            node.sent_frames.clear()
        if not any_broadcast:
            break
        if delivered:
            # Let the rest of the network hear the broadcast for one
            # more round so fanout/mesh side-effects are visible.
            if step >= 1:
                break

    dest_node = NODE_REGISTRY.get(dest_id)
    delivered = dest_node is not None and payload in dest_node.delivered
    # hops_used counts the number of forwarding broadcasts it took to
    # reach the destination. One broadcast per hop, so hops_used
    # equals transmissions along the path.
    hops_used = transmissions if delivered else 0
    final_ttl = 0
    if delivered and dest_node is not None:
        ttls = [f.ttl for f in dest_node.received_frames if f.dest == dest_id]
        if ttls:
            final_ttl = max(ttls)
    return RelayResult(
        delivered=delivered,
        hops_used=hops_used,
        final_ttl=final_ttl,
        transmissions=transmissions,
        nodes_visited=nodes_visited,
    )


# A simple global registry so the simulator can find nodes by id.
NODE_REGISTRY: Dict[str, RelayNode] = {}


def make_node(node_id: str, position: Tuple[float, float], channel: AcousticChannel) -> RelayNode:
    node = RelayNode(node_id=node_id, position=position, channel=channel)
    NODE_REGISTRY[node_id] = node
    return node


def reset_registry() -> None:
    NODE_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Topologies
# ---------------------------------------------------------------------------
def topology_chain(r: float = 10.0) -> Tuple[AcousticChannel, RelayNode, str]:
    """Three nodes in a line: A -- B -- C. A and C are out of range."""
    reset_registry()
    ch = AcousticChannel(max_range_units=r)
    a = make_node("A", (0.0, 0.0), ch)
    make_node("B", (r * 0.9, 0.0), ch)
    make_node("C", (r * 1.8, 0.0), ch)
    return ch, a, "C"


def topology_mesh(r: float = 10.0) -> Tuple[AcousticChannel, RelayNode, str]:
    """Five-node partial mesh: A--B--C, A--D, D--E (E is the target).

    Layout (with default r=10):
        A=(0,0)   B=(8,0)  C=(16,0)
        D=(0,7)   E=(7,7)

    A is in range of B (8) and D (7). D is in range of A (7) and E
    (sqrt(49+49) ~= 9.9, within 10). B is in range of A and C. E is in
    range of D only. So E is reachable from A in two hops via A-D-E.
    """
    reset_registry()
    ch = AcousticChannel(max_range_units=r)
    a = make_node("A", (0.0, 0.0), ch)
    make_node("B", (r * 0.8, 0.0), ch)
    make_node("C", (r * 1.6, 0.0), ch)
    make_node("D", (0.0, r * 0.7), ch)
    make_node("E", (r * 0.7, r * 0.7), ch)
    return ch, a, "E"


def topology_fanout(r: float = 10.0) -> Tuple[AcousticChannel, RelayNode, str]:
    """A central node A broadcasts to four leaves arranged in a + shape."""
    reset_registry()
    ch = AcousticChannel(max_range_units=r)
    a = make_node("A", (0.0, 0.0), ch)
    make_node("N", (0.0, r * 0.9), ch)
    make_node("S", (0.0, -r * 0.9), ch)
    make_node("E", (r * 0.9, 0.0), ch)
    make_node("W", (-r * 0.9, 0.0), ch)
    return ch, a, "N"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
TOPOLOGIES = {
    "chain": topology_chain,
    "mesh": topology_mesh,
    "5node": topology_mesh,  # alias
    "fanout": topology_fanout,
}


def main() -> int:
    p = argparse.ArgumentParser(description="SignalHop multi-hop relay simulator")
    p.add_argument("--payload", default="SOS: zone 7", help="message bytes to send")
    p.add_argument(
        "--topology",
        choices=sorted(TOPOLOGIES.keys()),
        default="chain",
        help="which network shape to simulate",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for link drops")
    p.add_argument("--ttl", type=int, default=8, help="initial time-to-live")
    p.add_argument("--max-steps", type=int, default=16, help="hard cap on rounds")
    args = p.parse_args()

    ch, source, dest_id = TOPOLOGIES[args.topology]()
    payload = args.payload.encode("utf-8")
    result = simulate(
        source=source,
        dest_id=dest_id,
        payload=payload,
        max_steps=args.max_steps,
        initial_ttl=args.ttl,
        seed=args.seed,
    )
    print(f"topology:  {args.topology}")
    print(f"source:    {source.node_id}")
    print(f"dest:      {dest_id}")
    print(f"payload:   {payload!r}")
    print(f"delivered: {result.delivered}")
    print(f"hops_used: {result.hops_used}")
    print(f"final_ttl: {result.final_ttl}")
    print(f"tx_count:  {result.transmissions}")
    print(f"path:      {' -> '.join(result.nodes_visited) or '(none)'}")
    return 0 if result.delivered else 1


if __name__ == "__main__":
    sys.exit(main())
