#!/usr/bin/env python3
"""
SignalHop — Mesh Networking Layer
Peer discovery via chirp beacons, hop-by-hop routing, simple distance-vector protocol.
"""

import time
import struct
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import defaultdict
import hashlib


MAX_HOPS = 8
BEACON_INTERVAL = 5.0      # seconds between peer discovery broadcasts
PEER_TIMEOUT = 30.0         # seconds before peer is considered gone
MAX_PEERS = 32


@dataclass
class Peer:
    id: bytes
    last_seen: float
    signal_strength: float = -60.0  # dBm (placeholder)
    position: Optional[tuple] = None  # (x, y) if known
    is_gateway: bool = False          # has internet access
    battery_pct: int = 100


@dataclass
class Route:
    next_hop: bytes
    metric: int       # hop count
    last_updated: float
    seq: int


@dataclass
class MeshFrame:
    src: bytes
    dst: bytes
    payload: bytes
    ttl: int
    seq: int
    path: List[bytes] = field(default_factory=list)


class MeshNode:
    """A node in the acoustic mesh network."""

    def __init__(self, node_id: Optional[bytes] = None):
        self.node_id = node_id or self._generate_node_id()
        self.peers: Dict[bytes, Peer] = {}
        self.routes: Dict[bytes, Route] = {}
        self.pending_acks: Dict[int, float] = {}
        self.message_log: Dict[int, bytes] = {}  # seq → payload
        self._last_beacon = 0.0

    # ─── Peer Discovery ────────────────────────────────────────────

    def should_broadcast_beacon(self) -> bool:
        return (time.time() - self._last_beacon) >= BEACON_INTERVAL

    def generate_beacon(self) -> bytes:
        """Generate a peer discovery beacon frame."""
        beacon = bytearray()
        beacon += b"BEACON_V1"
        beacon += self.node_id
        beacon += struct.pack("!I", int(time.time()))
        beacon += struct.pack("!H", self._battery_estimate())
        beacon += struct.pack("!B", 1 if self._has_internet() else 0)
        return bytes(beacon)

    def process_beacon(self, beacon_data: bytes) -> Optional[Peer]:
        """Parse a received beacon and update peer table."""
        if not beacon_data.startswith(b"BEACON_V1"):
            return None

        peer_id = beacon_data[8:16]
        ts = struct.unpack("!I", beacon_data[16:20])[0]
        battery = struct.unpack("!H", beacon_data[20:22])[0]
        is_gateway = beacon_data[22] == 1

        peer = Peer(
            id=peer_id,
            last_seen=ts,
            battery_pct=min(100, max(0, battery)),
            is_gateway=is_gateway,
            signal_strength=self._estimate_signal_strength()
        )
        self.peers[peer_id] = peer
        self._update_routes(peer_id)
        return peer

    def _update_routes(self, peer_id: bytes):
        """Update routing table when a new peer is discovered."""
        self.routes[peer_id] = Route(
            next_hop=peer_id,
            metric=1,
            last_updated=time.time(),
            seq=random.randint(0, 65535)
        )

    # ─── Routing ───────────────────────────────────────────────────

    def route_message(self, dst: bytes, payload: bytes) -> Optional[MeshFrame]:
        """Route a message toward destination."""
        if dst == self.node_id:
            return None  # local delivery

        if dst in self.peers:
            # Direct delivery
            next_hop = dst
        elif dst in self.routes:
            next_hop = self.routes[dst].next_hop
        else:
            # Flood to all peers (limited)
            next_hop = None

        seq = self._next_seq()
        frame = MeshFrame(
            src=self.node_id,
            dst=dst,
            payload=payload,
            ttl=MAX_HOPS,
            seq=seq,
            path=[self.node_id]
        )
        self.message_log[seq] = payload

        return frame

    def forward_frame(self, frame: MeshFrame) -> Optional[MeshFrame]:
        """Process a received frame (forward or deliver)."""
        if frame.dst == self.node_id:
            return frame  # deliver to this node

        if frame.ttl <= 0:
            return None  # expired

        # Check if we've already seen this frame (loop detection)
        if frame.seq in self.message_log and frame.src in [p.id for p in self.peers.values()]:
            return None  # duplicate, skip

        self.message_log[frame.seq] = frame.payload

        # Decrement TTL and forward
        frame.ttl -= 1
        frame.path.append(self.node_id)

        return frame

    # ─── Seq Counter ──────────────────────────────────────────────

    def _next_seq(self) -> int:
        if not hasattr(self, '_seq'):
            self._seq = random.randint(1, 65535)
        self._seq = (self._seq + 1) & 0xFFFF
        return self._seq

    # ─── Placeholder Hardware Abstraction ─────────────────────────

    def _generate_node_id(self) -> bytes:
        import uuid
        return hashlib.sha256(str(uuid.getnode()).encode()).digest()[:8]

    def _battery_estimate(self) -> int:
        return 85  # TODO: integrate with hardware

    def _has_internet(self) -> bool:
        return False  # TODO: check connectivity

    def _estimate_signal_strength(self) -> float:
        return random.uniform(-70, -45)

    # ─── Status ────────────────────────────────────────────────────

    def get_peer_count(self) -> int:
        self._purge_stale_peers()
        return len(self.peers)

    def get_routes(self) -> Dict[bytes, Route]:
        return dict(self.routes)

    def _purge_stale_peers(self):
        now = time.time()
        stale = [pid for pid, p in self.peers.items() if now - p.last_seen > PEER_TIMEOUT]
        for pid in stale:
            del self.peers[pid]
            if pid in self.routes:
                del self.routes[pid]


if __name__ == "__main__":
    node = MeshNode()
    print(f"Node ID: {node.node_id.hex()}")

    # Simulate peer discovery
    peer_beacon = node.generate_beacon()
    node.process_beacon(peer_beacon)
    print(f"Peers after own beacon: {node.get_peer_count()}")

    # Simulate a second peer
    node2 = MeshNode()
    peer2_beacon = node2.generate_beacon()
    node.process_beacon(peer2_beacon)
    print(f"Peers after second beacon: {node.get_peer_count()}")

    # Route a message
    frame = node.route_message(node2.node_id, b"Test message")
    if frame:
        print(f"Routed: {frame.src.hex()} → {frame.dst.hex()}, seq={frame.seq}")