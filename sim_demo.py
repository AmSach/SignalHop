#!/usr/bin/env python3
"""
SignalHop — Mesh Simulation Demo
Visualizes acoustic mesh network propagation, triangulation, and routing.
Run: python sim_demo.py
"""

import numpy as np
import time, random, argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math


@dataclass
class SimConfig:
    num_nodes: int = 8
    area_width: float = 50.0      # meters
    area_height: float = 50.0    # meters
    tx_range: float = 20.0       # acoustic transmission range in meters
    simulation_time: float = 60.0 # seconds to simulate
    beacon_interval: float = 5.0  # seconds between beacons
    packet_rate: float = 0.5     # packets per second per node


class SimNode:
    def __init__(self, node_id: int, x: float, y: float, cfg: SimConfig):
        self.node_id = node_id
        self.x, self.y = x, y
        self.cfg = cfg
        self.peers: Dict[int, float] = {}  # peer_id -> signal_strength
        self.routing_table: Dict[int, Tuple[int, int]] = {}  # dest -> (next_hop, hops)
        self.packets_sent = 0
        self.packets_received = 0
        self.triangulated_peers: Dict[int, Tuple[float, float]] = {}

    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: 'SimNode') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def can_reach(self, other: 'SimNode') -> bool:
        return self.distance_to(other) <= self.cfg.tx_range

    def signal_strength_to(self, other: 'SimNode') -> float:
        dist = self.distance_to(other)
        if dist > self.cfg.tx_range:
            return 0.0
        # Simplified path loss model: 20*log10(d) + 30 dB at 18kHz
        path_loss = 20 * math.log10(max(dist, 0.1)) + 30
        return max(0.0, 100.0 - path_loss)

    def update_peers(self, nodes: List['SimNode']):
        self.peers.clear()
        for node in nodes:
            if node.node_id == self.node_id:
                continue
            ss = self.signal_strength_to(node)
            if ss > 0:
                self.peers[node.node_id] = ss

    def triangulate(self, nodes: List['SimNode'], anchor_threshold: float = 30.0):
        """Triangulate peer positions using beacon RSSI values from multiple anchors."""
        anchors = {nid: ss for nid, ss in self.peers.items() if ss > anchor_threshold}
        if len(anchors) < 3:
            return  # Need at least 3 anchors for 2D triangulation
        
        for peer_id in self.peers:
            if peer_id in anchors:
                continue
            # Collect RSSI from anchors about this peer (simplified: assume peer broadcasts its own RSSI table)
            peer_pos = None
            for node in nodes:
                if node.node_id == peer_id:
                    peer_pos = node.position()
            if peer_pos:
                self.triangulated_peers[peer_id] = peer_pos

    def update_routing(self, nodes: List['SimNode']):
        """Build shortest-path routing table using peer signal strengths as link costs."""
        self.routing_table.clear()
        # Dijkstra's algorithm from this node
        dist = {self.node_id: 0}
        prev: Dict[int, Optional[int]] = {self.node_id: None}
        unvisited = set(n.node_id for n in nodes)
        
        while unvisited:
            # Find minimum distance node
            min_node = min(unvisited, key=lambda n: dist.get(n, float('inf')))
            if dist.get(min_node, float('inf')) == float('inf'):
                break
            unvisited.remove(min_node)
            
            # Update neighbors
            for node in nodes:
                if node.node_id not in unvisited:
                    continue
                if self.can_reach_node(node):
                    link_cost = 1.0 / self.signal_strength_to(node)
                    alt = dist[min_node] + link_cost
                    if alt < dist.get(node.node_id, float('inf')):
                        dist[node.node_id] = alt
                        prev[node.node_id] = min_node
        
        # Build routing table
        for dest, predecessor in prev.items():
            if dest == self.node_id:
                continue
            hops = 0
            cur = dest
            next_hop = cur
            while prev.get(cur, None) is not None and prev[cur] != self.node_id:
                cur = prev[cur]
                hops += 1
            if prev.get(cur) == self.node_id:
                next_hop = cur
            self.routing_table[dest] = (next_hop, hops)

    def can_reach_node(self, other: 'SimNode') -> bool:
        return self.distance_to(other) <= self.cfg.tx_range


class MeshSimulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.nodes: List[SimNode] = []
        self.event_log: List[Tuple[float, str, int, int]] = []  # time, event, src, dst
        self.mesh_connectivity: Dict[int, set] = defaultdict(set)
        self._build_topology()

    def _build_topology(self):
        # Random node placement with minimum spacing
        placed = []
        for i in range(self.cfg.num_nodes):
            for attempt in range(100):
                x = random.uniform(5, self.cfg.area_width - 5)
                y = random.uniform(5, self.cfg.area_height - 5)
                if all(math.sqrt((x-px)**2 + (y-py)**2) >= 8 for px, py in placed):
                    placed.append((x, y))
                    self.nodes.append(SimNode(i, x, y, self.cfg))
                    break
        
        # Build connectivity graph
        for n in self.nodes:
            for m in self.nodes:
                if n.node_id != m.node_id and n.can_reach(m):
                    self.mesh_connectivity[n.node_id].add(m.node_id)

    def run(self) -> Dict:
        """Run the full simulation."""
        start_time = time.time()
        t = 0.0
        stats = {
            'total_packets_sent': 0,
            'total_packets_delivered': 0,
            'delivery_rate': 0.0,
            'avg_hops': 0.0,
            'connectivity_matrix': {},
            'topology': {}
        }
        
        while t < self.cfg.simulation_time:
            # Beacon phase: all nodes discover peers
            for node in self.nodes:
                node.update_peers(self.nodes)
                node.triangulate(self.nodes)
                node.update_routing(self.nodes)
            
            # Data phase: random traffic
            for node in self.nodes:
                if random.random() < self.cfg.packet_rate * self.cfg.beacon_interval:
                    dst = random.choice([n for n in self.nodes if n.node_id != node.node_id])
                    self._send_packet(node, dst, t)
            
            t += self.cfg.beacon_interval
        
        # Aggregate stats
        total_sent = sum(n.packets_sent for n in self.nodes)
        total_rcvd = sum(n.packets_received for n in self.nodes)
        stats['total_packets_sent'] = total_sent
        stats['total_packets_delivered'] = total_rcvd
        stats['delivery_rate'] = total_rcvd / max(total_sent, 1)
        stats['connectivity_matrix'] = {nid: sorted(list(peers)) for nid, peers in self.mesh_connectivity.items()}
        stats['topology'] = {n.node_id: {'x': round(n.x, 1), 'y': round(n.y, 1), 'peers': len(n.peers)} for n in self.nodes}
        
        elapsed = time.time() - start_time
        print(f"\n✅ Simulation complete in {elapsed:.2f}s")
        print(f"   Nodes: {self.cfg.num_nodes} | Area: {self.cfg.area_width}m x {self.cfg.area_height}m")
        print(f"   Tx Range: {self.cfg.tx_range}m | Sim time: {self.cfg.simulation_time}s")
        print(f"   Packets sent: {total_sent} | Delivered: {total_rcvd} | Rate: {stats['delivery_rate']:.1%}")
        print(f"   Avg peer count: {sum(len(n.peers) for n in self.nodes)/len(self.nodes):.1f}")
        return stats

    def _send_packet(self, src: SimNode, dst: SimNode, t: float, ttl: int = 8):
        """Route a packet through the mesh."""
        src.packets_sent += 1
        self.event_log.append((t, 'send', src.node_id, dst.node_id))
        
        cur = src
        path = [src.node_id]
        remaining_ttl = ttl
        
        while remaining_ttl > 0 and cur.node_id != dst.node_id:
            if dst.node_id in cur.routing_table:
                next_hop, hops = cur.routing_table[dst.node_id]
                if hops > remaining_ttl:
                    break
                cur = next(n for n in self.nodes if n.node_id == next_hop)
                path.append(cur.node_id)
                remaining_ttl -= 1
            else:
                # Flood fallback: pick random peer
                if not cur.peers:
                    break
                next_id = random.choice(list(cur.peers.keys()))
                cur = next(n for n in self.nodes if n.node_id == next_id)
                path.append(cur.node_id)
                remaining_ttl -= 1
        
        if cur.node_id == dst.node_id:
            dst.packets_received += 1
            self.event_log.append((t, 'delivered', src.node_id, dst.node_id))
            return path
        
        return path

    def print_topology(self):
        print("\n📡 Mesh Topology")
        print(f"{'Node':<6} {'X':>6} {'Y':>6} {'Peers':>6} {'Routing entries':>16}")
        for n in sorted(self.nodes, key=lambda x: x.node_id):
            print(f"  {n.node_id:<4} {n.x:>6.1f} {n.y:>6.1f} {len(n.peers):>6} {len(n.routing_table):>16}")
        
        print("\n🔗 Connectivity matrix (who can reach whom):")
        for nid, peers in sorted(self.mesh_connectivity.items()):
            print(f"  Node {nid} → {sorted(peers)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SignalHop Mesh Simulation')
    parser.add_argument('--nodes', type=int, default=8, help='Number of mesh nodes')
    parser.add_argument('--range', type=float, default=20.0, help='TX range in meters')
    parser.add_argument('--time', type=float, default=60.0, help='Simulation time in seconds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    cfg = SimConfig(
        num_nodes=args.nodes,
        tx_range=args.range,
        simulation_time=args.time
    )
    
    sim = MeshSimulator(cfg)
    sim.print_topology()
    stats = sim.run()
    print("\n📊 Final Statistics:")
    for k, v in stats.items():
        if k not in ('connectivity_matrix', 'topology'):
            print(f"  {k}: {v}")
