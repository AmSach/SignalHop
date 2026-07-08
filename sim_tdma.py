#!/usr/bin/env python3
"""
SignalHop — TDMA Slot Scheduler
Acoustic mesh links collide if every node transmits on the same beacon tick.
A simple TDMA scheduler assigns each node a fixed slot in the beacon frame,
trading peak throughput for collision-free delivery.

Frame layout:
    [slot 0: sync] [slot 1: node 1] [slot 2: node 2] ... [slot N: node N]
                   <-- transmission window per node -->

Run: python sim_tdma.py
"""
import argparse
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from sim_demo import SimConfig, SimNode, MeshSimulator


SLOT_DURATION_MS = 250.0  # acoustic frame is 250ms per node slot


@dataclass
class TDMASchedule:
    """Per-frame slot assignment: slot_index -> node_id."""
    slots: Dict[int, int] = field(default_factory=dict)
    sync_slot: int = 0
    frame_ms: float = 0.0

    def slot_for(self, node_id: int) -> int:
        for slot, nid in self.slots.items():
            if nid == node_id:
                return slot
        return -1

    def is_active(self, node_id: int, t_ms: float) -> bool:
        """True if the given node is allowed to transmit at t_ms."""
        slot = self.slot_for(node_id)
        if slot < 0:
            return False
        slot_start = (self.sync_slot * SLOT_DURATION_MS) + (slot * SLOT_DURATION_MS)
        slot_end = slot_start + SLOT_DURATION_MS
        t_in_frame = t_ms % self.frame_ms
        return slot_start <= t_in_frame < slot_end


def build_schedule(node_ids: List[int]) -> TDMASchedule:
    """Assign one slot per node, plus a sync slot at index 0."""
    slots: Dict[int, int] = {0: -1}  # -1 = sync beacon
    for i, nid in enumerate(sorted(node_ids), start=1):
        slots[i] = nid
    frame_ms = len(slots) * SLOT_DURATION_MS
    return TDMASchedule(slots=slots, frame_ms=frame_ms)


class TDMASimulator(MeshSimulator):
    """Mesh simulator that gates transmissions by the TDMA schedule."""

    def __init__(self, cfg: SimConfig, schedule: TDMASchedule):
        super().__init__(cfg)
        self.schedule = schedule
        self.collisions_avoided = 0
        self.gated_transmissions = 0
        self.delivered_via_tdma = 0
        self.sent_via_tdma = 0

    def run_tdma(self) -> Dict:
        random.seed(getattr(self, '_seed', 42))
        t_s = 0.0
        dt_s = self.cfg.beacon_interval

        while t_s < self.cfg.simulation_time:
            t_ms = t_s * 1000.0
            for node in self.nodes:
                node.update_peers(self.nodes)
                node.update_routing(self.nodes)
                if random.random() < self.cfg.packet_rate * dt_s:
                    if not self.schedule.is_active(node.node_id, t_ms):
                        self.gated_transmissions += 1
                        continue
                    candidates = [n for n in self.nodes if n.node_id != node.node_id]
                    if not candidates:
                        continue
                    dst = random.choice(candidates)
                    path = self._route_packet_simple(node, dst)
                    self.sent_via_tdma += 1
                    if path and path[-1] == dst.node_id:
                        dst.packets_received += 1
                        self.delivered_via_tdma += 1
                    else:
                        node.packets_sent += 1
            t_s += dt_s

        delivered_unscheduled = sum(n.packets_received for n in self.nodes)
        return {
            'sent_via_tdma': self.sent_via_tdma,
            'delivered_via_tdma': self.delivered_via_tdma,
            'delivery_rate': self.delivered_via_tdma / max(self.sent_via_tdma, 1),
            'gated_transmissions': self.gated_transmissions,
            'frame_ms': self.schedule.frame_ms,
            'n_slots': len(self.schedule.slots),
            'baseline_delivered': delivered_unscheduled,
        }

    def _route_packet_simple(self, src: 'SimNode', dst: 'SimNode', ttl: int = 8) -> List[int]:
        cur = src
        path = [src.node_id]
        remaining = ttl
        while remaining > 0 and cur.node_id != dst.node_id:
            if dst.node_id in cur.routing_table:
                next_hop, _ = cur.routing_table[dst.node_id]
                cur = next(n for n in self.nodes if n.node_id == next_hop)
                path.append(cur.node_id)
            else:
                return []
            remaining -= 1
        return path

    def print_tdma_report(self, stats: Dict) -> None:
        print("\n📅  TDMA Schedule Report")
        print(f"   Slots per frame:    {stats['n_slots']}")
        print(f"   Frame length:       {stats['frame_ms']:.0f} ms")
        print(f"   Packets attempted:  {stats['sent_via_tdma']}")
        print(f"   Delivered:          {stats['delivered_via_tdma']} "
              f"({stats['delivery_rate']:.1%})")
        print(f"   Gated (not in slot):{stats['gated_transmissions']}")
        print("\n   Slot map:")
        for slot, nid in sorted(self.schedule.slots.items()):
            label = "SYNC" if nid == -1 else f"node {nid}"
            print(f"     [{slot:2}]  {label}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SignalHop TDMA Scheduler')
    parser.add_argument('--nodes', type=int, default=8)
    parser.add_argument('--range', type=float, default=20.0)
    parser.add_argument('--time', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = SimConfig(num_nodes=args.nodes, tx_range=args.range, simulation_time=args.time)
    sim = MeshSimulator(cfg)
    sim._seed = args.seed
    sim.print_topology()
    schedule = build_schedule([n.node_id for n in sim.nodes])
    tdma = TDMASimulator(cfg, schedule)
    tdma._seed = args.seed
    stats = tdma.run_tdma()
    tdma.print_tdma_report(stats)
