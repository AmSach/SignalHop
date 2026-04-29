#!/usr/bin/env python3
"""
SignalHop — Mesh Networking Layer
Peer discovery via chirp beacons, hop-by-hop routing with TTL.
"""

import struct, time, threading
from dataclasses import dataclass
from typing import Optional, Dict, List
from collections import deque


NETWORK_ID = b"SIGNALHOP_V1"
BEACON_INTERVAL = 5.0  # seconds between chirp beacons
PEER_TIMEOUT = 30.0     # seconds before a peer is considered lost


@dataclass
class Peer:
    node_id: bytes
    last_seen: float
    signal_strength: float  # correlation peak
    hops: int = 1


class MeshNode:
    """Participate in the SignalHop mesh network."""

    def __init__(self, node_id: bytes, modem=None):
        self.node_id = node_id[:8].ljust(8, b'\x00')
        self.modem = modem
        self.peers: Dict[bytes, Peer] = {}
        self.message_queue: deque = deque(maxlen=100)
        self.routing_table: Dict[bytes, tuple] = {}  # dest -> (next_hop, hops)
        self.running = False
        self._lock = threading.Lock()

    def start(self):
        """Start beacon broadcasts and peer discovery."""
        self.running = True
        threading.Thread(target=self._beacon_loop, daemon=True).start()
        threading.Thread(target=self._peer_cleanup, daemon=True).start()

    def stop(self):
        self.running = False

    def _beacon_loop(self):
        """Broadcast chirp beacons periodically."""
        while self.running:
            self._send_beacon()
            time.sleep(BEACON_INTERVAL)

    def _send_beacon(self):
        """Send a peer announcement chirp."""
        if not self.modem:
            return
        # Beacon = short preamble only (no payload)
        preamble = self.modem.generate_preamble()
        # Embed our node ID in the preamble via small frequency offset
        self.modem.preamble_override = self.node_id
        # In a real impl, we'd modulate the chirp's start frequency with node_id
        # For now, just broadcast sync

    def _peer_cleanup(self):
        """Remove stale peers."""
        while self.running:
            time.sleep(PEER_TIMEOUT)
            with self._lock:
                now = time.time()
                stale = [k for k, v in self.peers.items() if now - v.last_seen > PEER_TIMEOUT]
                for k in stale:
                    del self.peers[k]

    def receive_beacon(self, signal, peer_node_id, signal_strength):
        """Called when a chirp beacon is detected."""
        with self._lock:
            self.peers[peer_node_id] = Peer(
                node_id=peer_node_id,
                last_seen=time.time(),
                signal_strength=signal_strength,
                hops=1
            )

    def route(self, payload: bytes, ttl: int = 8) -> Optional[bytes]:
        """Route a payload to its destination, hopping through peers."""
        if not payload:
            return None

        # Check if we have a direct path to destination
        for peer_id, peer in self.peers.items():
            if peer.hops <= ttl:
                return self._forward(peer_id, payload, ttl)

        # No route found — flood to all peers
        return self._flood(payload, ttl)

    def _forward(self, next_hop: bytes, payload: bytes, ttl: int) -> bytes:
        """Forward a payload to a specific peer."""
        if ttl <= 0:
            return None
        # Build frame with decremented TTL
        frame = self.modem.build_frame(payload, ttl=ttl - 1, sender_id=self.node_id)
        # In real impl, transmit via modem
        return frame

    def _flood(self, payload: bytes, ttl: int) -> bytes:
        """Flood payload to all reachable peers."""
        return self._forward(b'\xff\xff\xff\xff\xff\xff\xff\xff', payload, ttl)


class RoutingTable:
    """Simple routing table with shortest-path logic."""

    def __init__(self):
        self.routes: Dict[bytes, tuple] = {}  # dest_id -> (next_hop_id, hops)

    def add_route(self, dest: bytes, next_hop: bytes, hops: int):
        """Add or update a route."""
        if dest not in self.routes or hops < self.routes[dest][1]:
            self.routes[dest] = (next_hop, hops)

    def best_route(self, dest: bytes) -> Optional[tuple]:
        return self.routes.get(dest)

    def prune_invalid(self, alive_peers: set):
        """Remove routes through dead peers."""
        self.routes = {
            d: v for d, v in self.routes.items()
            if v[0] in alive_peers
        }
