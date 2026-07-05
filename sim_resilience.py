#!/usr/bin/env python3
"""
SignalHop — Mesh Resilience & Failure Simulation
Tests how the acoustic mesh recovers when nodes fail or get noisy.
Run: python sim_resilience.py
"""
import random
import math
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from sim_demo import SimConfig, SimNode, MeshSimulator


class FailureProfile:
    """Describes a node failure schedule."""
    def __init__(self, fail_at: Dict[int, Tuple[float, float]]):
        # node_id -> (fail_time, recover_time or inf)
        self.fail_at = fail_at

    def is_alive(self, node_id: int, t: float) -> bool:
        schedule = self.fail_at.get(node_id)
        if not schedule:
            return True
        fail_t, recover_t = schedule
        return not (fail_t <= t < recover_t)

    def noise_db(self, node_id: int, t: float) -> float:
        """Extra noise in dB added to the link at time t."""
        if not self.is_alive(node_id, t):
            return 200.0  # saturated, no link
        return 0.0


class ResilientSimulator(MeshSimulator):
    """Extends MeshSimulator to model node failure + noise injection."""
    def __init__(self, cfg: SimConfig, failure_profile: FailureProfile):
        super().__init__(cfg)
        self.failure_profile = failure_profile
        self.failure_events: List[Tuple[float, str, int]] = []  # (time, event, node_id)
        self.delivered_after_failure = 0
        self.sent_after_failure = 0

    def signal_strength_to(self, src: 'SimNode', dst: 'SimNode', t: float) -> float:
        """Override base signal calc with noise + failure awareness."""
        if not self.failure_profile.is_alive(src.node_id, t) or \
           not self.failure_profile.is_alive(dst.node_id, t):
            return 0.0
        dist = src.distance_to(dst)
        if dist > self.cfg.tx_range:
            return 0.0
        path_loss = 20 * math.log10(max(dist, 0.1)) + 30
        noise = self.failure_profile.noise_db(src.node_id, t) + \
                self.failure_profile.noise_db(dst.node_id, t)
        return max(0.0, 100.0 - path_loss - noise * 0.3)

    def update_peers(self, node: SimNode, t: float):
        """Snapshot of peers at time t considering failures."""
        node.peers.clear()
        for other in self.nodes:
            if other.node_id == node.node_id:
                continue
            ss = self.signal_strength_to(node, other, t)
            if ss > 0:
                node.peers[other.node_id] = ss

    def update_routing(self, node: SimNode, t: float):
        """Dijkstra over currently-alive neighbors only."""
        node.routing_table.clear()
        dist = {node.node_id: 0}
        prev: Dict[int, Optional[int]] = {node.node_id: None}
        unvisited = set(n.node_id for n in self.nodes
                        if self.failure_profile.is_alive(n.node_id, t))

        while unvisited:
            min_node = min(unvisited, key=lambda n: dist.get(n, float('inf')))
            if dist.get(min_node, float('inf')) == float('inf'):
                break
            unvisited.remove(min_node)
            for other in self.nodes:
                if other.node_id not in unvisited:
                    continue
                if other.node_id in node.peers:
                    link_cost = 1.0 / max(node.peers[other.node_id], 0.001)
                    alt = dist[min_node] + link_cost
                    if alt < dist.get(other.node_id, float('inf')):
                        dist[other.node_id] = alt
                        prev[other.node_id] = min_node

        for dest, predecessor in prev.items():
            if dest == node.node_id or prev.get(dest) is None:
                continue
            cur = dest
            next_hop = cur
            hops = 0
            while prev.get(cur) is not None and prev[cur] != node.node_id:
                cur = prev[cur]
                hops += 1
            if prev.get(cur) == node.node_id:
                next_hop = cur
            node.routing_table[dest] = (next_hop, hops)

    def run_with_failures(self) -> Dict:
        """Run sim and track resilience metrics."""
        random.seed(getattr(self, '_seed', 42))
        t = 0.0
        start_failure_watch = False
        stats = {
            'delivery_rate_baseline': 0.0,
            'delivery_rate_after_failures': 0.0,
            'avg_hops_baseline': 0.0,
            'avg_hops_after_failures': 0.0,
            'packets_lost_due_to_failure': 0,
            'recovery_time_s': 0.0,
            'failure_schedule': self.failure_profile.fail_at,
        }
        baseline_sent = 0
        baseline_rcvd = 0
        post_sent = 0
        post_rcvd = 0
        post_hop_total = 0
        post_hop_count = 0
        recovery_time = None

        while t < self.cfg.simulation_time:
            # Refresh state for this tick
            for node in self.nodes:
                self.update_peers(node, t)
                self.update_routing(node, t)
                node.triangulated_peers.clear()

            # Generate traffic
            for node in self.nodes:
                if random.random() < self.cfg.packet_rate * self.cfg.beacon_interval:
                    candidates = [n for n in self.nodes
                                  if n.node_id != node.node_id
                                  and self.failure_profile.is_alive(n.node_id, t)]
                    if not candidates:
                        continue
                    dst = random.choice(candidates)
                    path = self._route_packet(node, dst, t)
                    if path and path[-1] == dst.node_id:
                        node.packets_sent += 1
                        dst.packets_received += 1
                        hops = len(path) - 1
                        if t < 20.0:
                            baseline_sent += 1
                            baseline_rcvd += 1
                        else:
                            post_sent += 1
                            post_rcvd += 1
                            post_hop_total += hops
                            post_hop_count += 1
                    else:
                        node.packets_sent += 1
                        if t < 20.0:
                            baseline_sent += 1
                        else:
                            post_sent += 1
                            stats['packets_lost_due_to_failure'] += 1
                            if recovery_time is None:
                                recovery_time = t - 20.0

            t += self.cfg.beacon_interval

        stats['delivery_rate_baseline'] = baseline_rcvd / max(baseline_sent, 1)
        stats['delivery_rate_after_failures'] = post_rcvd / max(post_sent, 1)
        stats['avg_hops_baseline'] = 0.0  # not tracked per-packet in baseline
        stats['avg_hops_after_failures'] = post_hop_total / max(post_hop_count, 1)
        stats['recovery_time_s'] = recovery_time if recovery_time is not None else 0.0
        return stats

    def _route_packet(self, src: SimNode, dst: SimNode, t: float, ttl: int = 8) -> Optional[List[int]]:
        cur = src
        path = [src.node_id]
        remaining = ttl
        while remaining > 0 and cur.node_id != dst.node_id:
            if dst.node_id in cur.routing_table:
                next_hop, _ = cur.routing_table[dst.node_id]
                cur = next(n for n in self.nodes if n.node_id == next_hop)
                path.append(cur.node_id)
                remaining -= 1
            else:
                if not cur.peers:
                    return None
                cur = next(n for n in self.nodes if n.node_id in cur.peers)
                path.append(cur.node_id)
                remaining -= 1
        return path

    def print_resilience_report(self, stats: Dict):
        print("\n🛡️  Mesh Resilience Report")
        print(f"   Baseline delivery rate: {stats['delivery_rate_baseline']:.1%}")
        print(f"   Post-failure delivery:  {stats['delivery_rate_after_failures']:.1%}")
        print(f"   Avg hops (post-fail):   {stats['avg_hops_after_failures']:.2f}")
        print(f"   Packets lost to failure: {stats['packets_lost_due_to_failure']}")
        print(f"   First loss at:          {stats['recovery_time_s']:.1f}s after failure")
        print("\n   Failure schedule:")
        for nid, (fail_t, recover_t) in stats['failure_schedule'].items():
            rt = "∞" if recover_t == float('inf') else f"{recover_t:.0f}s"
            print(f"     Node {nid}: down @ {fail_t:.0f}s, back @ {rt}")


def default_failure_schedule(num_nodes: int) -> FailureProfile:
    """Sample scenario: 2 nodes fail at t=20, one recovers at t=45."""
    fails = {}
    if num_nodes >= 3:
        fails[1] = (20.0, float('inf'))   # permanent
        fails[3] = (20.0, 45.0)            # temporary
    return FailureProfile(fails)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SignalHop Mesh Resilience Test')
    parser.add_argument('--nodes', type=int, default=8)
    parser.add_argument('--range', type=float, default=20.0)
    parser.add_argument('--time', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = SimConfig(num_nodes=args.nodes, tx_range=args.range, simulation_time=args.time)
    profile = default_failure_schedule(args.nodes)
    sim = ResilientSimulator(cfg, profile)
    sim._seed = args.seed
    sim.print_topology()
    stats = sim.run_with_failures()
    sim.print_resilience_report(stats)
