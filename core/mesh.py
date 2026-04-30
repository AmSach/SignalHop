#!/usr/bin/env python3
"""
SignalHop — Mesh Networking Layer
Peer discovery via chirp beacons, hop-by-hop routing with TTL.
"""

import struct, time, threading, numpy as np
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
        preamble = self.modem.generate_preamble()
        # Encode node_id into last chirp via small frequency offset (±200Hz encoding)
        node_offset = int.from_bytes(self.node_id[:4], 'big') % 401 - 200  # -200..+200 Hz
        chirp = self.modem.generate_chirp(up=True)
        freq_start = self.modem.cfg.carrier_low - 2000 + node_offset
        freq_end = self.modem.cfg.carrier_high + 2000
        t = np.linspace(0, 0.05, len(chirp), False)
        phase = 2 * np.pi * np.cumsum(np.linspace(freq_start, freq_end, len(t))) / self.modem.cfg.sample_rate
        chirp = np.sin(phase).astype(np.float32)
        tx_signal = np.concatenate([preamble, chirp])
        # In production: write to audio device or transmit via modem
        return tx_signal

    def _peer_cleanup(self):
        """Remove stale peers."""
        while self.running:
            time.sleep(PEER_TIMEOUT / 2)
            with self._lock:
                now = time.time()
                stale = [k for k, v in self.peers.items() if now - v.last_seen > PEER_TIMEOUT]
                for k in stale:
                    del self.peers[k]
                    # Prune routing table entries via this peer
                    self.routing_table = {d: v for d, v in self.routing_table.items() if v[0] != k}

    def discover_peers(self, signals: List[tuple]) -> List[Peer]:
        """Process incoming signals and update peer list.
        
        Args:
            signals: List of (signal_array, peer_node_id, signal_strength) tuples
                    from recent acoustic activity.
        
        Returns:
            List of newly discovered or updated peers.
        """
        discovered = []
        for signal, peer_id, strength in signals:
            if not self.modem:
                continue
            if self.modem.detect_chirp(signal):
                with self._lock:
                    if peer_id not in self.peers:
                        discovered.append(peer_id)
                    self.peers[peer_id] = Peer(
                        node_id=peer_id,
                        last_seen=time.time(),
                        signal_strength=strength,
                        hops=1
                    )
                    # Update routing table
                    self.routing_table[peer_id] = (peer_id, 1)
        return discovered

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
        # Check routing table for direct or multi-hop path
        dest = payload[:8] if len(payload) >= 8 else payload
        if dest in self.routing_table:
            next_hop, hops = self.routing_table[dest]
            if hops <= ttl:
                return self._forward(next_hop, payload, ttl - 1)
        # Flood to all known peers within TTL
        return self._flood(payload, ttl)

    def _forward(self, next_hop: bytes, payload: bytes, ttl: int) -> Optional[bytes]:
        """Forward a payload to a specific peer."""
        if ttl < 0 or not self.modem:
            return None
        # Build frame manually: preamble + encoded payload
        frame_data = bytes(self.node_id) + struct.pack('!B', ttl) + payload
        return self.modem.build_frame(frame_data)

    def _flood(self, payload: bytes, ttl: int) -> Optional[bytes]:
        """Flood payload to all reachable peers."""
        if ttl < 0 or not self.modem:
            return None
        return self.modem.build_frame(b'\xff' * 8 + struct.pack('!B', ttl) + payload)


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


# Mesh protocol enhancements added
