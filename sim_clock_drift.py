#!/usr/bin/env python3
"""
SignalHop - Clock-Drift & TDMA Collision Simulator

Real nodes have cheap oscillators. Even after a sync beacon, each node's
local clock drifts tens of ppm, so its "slot 3" gradually slides forward
or backward relative to the others'. The sync slot protects the network
for a while, then collisions start to appear - and once they do, the
mesh degrades fast because the cheap acoustic PHY can't CSMA/CA cleanly.

This sim models:
  - Per-node oscillator drift (ppm, normal distribution)
  - Periodic re-sync (configurable interval)
  - Guard time around each slot
  - Collision count vs. time, delivery rate vs. time

Run:  python sim_clock_drift.py
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sim_tdma import TDMASchedule, SLOT_DURATION_MS


# 1 ppm of drift = 1 microsecond of error per second = 1e-3 ms per second
PPM_TO_MS_PER_S = 1e-3


@dataclass
class NodeClock:
    """A node's local clock, modelled as a real offset from master time.

    After resync(), offset_ms is set so local_t(t) == t exactly at sync.
    As master time advances, the local clock accumulates drift at the rate
    ``drift_ppm * 1e-3 ms per second``. Positive drift = local runs fast
    (gets ahead of master), so local_t(t) > t after enough time.
    """
    node_id: int
    drift_ppm: float
    last_sync_t_ms: float = 0.0
    offset_ms: float = 0.0  # local - master at last_sync_t_ms; reset to 0

    def local_t(self, master_t_ms: float) -> float:
        """Return the node's local clock reading at master time t."""
        elapsed_ms = master_t_ms - self.last_sync_t_ms
        # 1 ppm = 1e-6 of elapsed time, in same units as elapsed.
        drift_ms = elapsed_ms * self.drift_ppm * 1e-6
        return master_t_ms + self.offset_ms + drift_ms

    def resync(self, master_t_ms: float) -> None:
        """Re-zero the offset to master time."""
        self.last_sync_t_ms = master_t_ms
        self.offset_ms = 0.0


@dataclass
class CollisionReport:
    """Result of one collision sweep."""
    collisions: int
    total_transmissions: int
    delivered: int
    delivery_rate: float
    worst_overlap_ms: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "collisions": self.collisions,
            "total_transmissions": self.total_transmissions,
            "delivered": self.delivered,
            "delivery_rate": round(self.delivery_rate, 4),
            "worst_overlap_ms": round(self.worst_overlap_ms, 3),
        }


def _assigned_slot_wire_time(
    clock: NodeClock,
    master_t_ms: float,
    node_slot_idx: int,
    frame_ms: float,
) -> Tuple[float, float]:
    """Return the master-time interval when the node's assigned slot fires on the wire.

    A node fires its slot whenever its local clock believes it is in the
    middle of that slot. We compute the local-time interval [s, e] for
    node_slot_idx, then convert to master time using local_t = master_t + drift.
    """
    # The node believes the frame started at last_sync_t_ms in local time.
    # Its assigned slot in local-time coordinates is
    # [last_sync_t_ms + node_slot_idx * SLOT_DURATION_MS, +SLOT_DURATION_MS]
    # But over time, multiple frames pass. Find the *current* slot window
    # (the one that contains the node's current local time).
    local_now = clock.local_t(master_t_ms)
    elapsed_local = local_now - clock.last_sync_t_ms
    if elapsed_local < 0:
        # Clock is way off, but pretend it just resynced for safety
        slot_local_start = clock.last_sync_t_ms + node_slot_idx * SLOT_DURATION_MS
    else:
        # Number of full frames elapsed in local time
        frames_passed = int(elapsed_local // frame_ms)
        slot_local_start = (
            clock.last_sync_t_ms
            + frames_passed * frame_ms
            + node_slot_idx * SLOT_DURATION_MS
        )
    slot_local_end = slot_local_start + SLOT_DURATION_MS

    # Convert local-time interval back to master time using master = local - drift
    # For each local-time instant L, master = (L - last_sync_t) / (1 + ppm*1e-6) + last_sync_t
    # (the clock runs at rate (1 + drift_ppm * 1e-6) relative to master)
    rate = 1.0 + clock.drift_ppm * 1e-6

    slot_master_start = clock.last_sync_t_ms + (slot_local_start - clock.last_sync_t_ms) / rate
    slot_master_end = clock.last_sync_t_ms + (slot_local_end - clock.last_sync_t_ms) / rate

    return slot_master_start, slot_master_end


def simulate(
    num_nodes: int,
    duration_s: float,
    sync_interval_s: float,
    drift_ppm_std: float = 30.0,
    guard_ms: float = 5.0,
    seed: int = 42,
) -> CollisionReport:
    """Run the clock-drift simulation for ``duration_s`` seconds.

    At each tick, every node's local clock is consulted. Each node picks
    the slot it believes it's in; the master-time window of that slot is
    computed; pairwise overlaps exceeding ``guard_ms`` are counted as
    collisions.
    """
    rng = random.Random(seed)

    clocks: Dict[int, NodeClock] = {
        nid: NodeClock(
            node_id=nid,
            drift_ppm=rng.gauss(0.0, drift_ppm_std),
        )
        for nid in range(1, num_nodes + 1)
    }

    # Each node owns slot index = its id, sync slot is 0
    schedule = TDMASchedule()
    for nid in range(1, num_nodes + 1):
        schedule.slots[nid] = nid
    frame_ms = SLOT_DURATION_MS * (num_nodes + 1)  # +1 for sync

    total_transmissions = 0
    collisions = 0
    delivered = 0
    worst_overlap_ms = 0.0

    step_ms = 10.0
    t_ms = 0.0
    end_ms = duration_s * 1000.0
    next_sync_ms = 0.0

    while t_ms < end_ms:
        if t_ms >= next_sync_ms:
            for c in clocks.values():
                c.resync(t_ms)
            next_sync_ms += sync_interval_s * 1000.0

        # For each node, compute the master-time window of its assigned slot
        active: List[Tuple[int, float, float]] = []
        for nid, clock in clocks.items():
            s, e = _assigned_slot_wire_time(clock, t_ms, nid, frame_ms)
            active.append((nid, s, e))

        # Pairwise overlap (only count if both are transmitting at the same
        # master time)
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                _, s1, e1 = active[i]
                _, s2, e2 = active[j]
                overlap = min(e1, e2) - max(s1, s2)
                if overlap > guard_ms:
                    collisions += 1
                    worst_overlap_ms = max(worst_overlap_ms, overlap)
                else:
                    delivered += 1
                total_transmissions += 1

        t_ms += step_ms

    delivery_rate = (delivered / total_transmissions) if total_transmissions else 1.0
    return CollisionReport(
        collisions=collisions,
        total_transmissions=total_transmissions,
        delivered=delivered,
        delivery_rate=delivery_rate,
        worst_overlap_ms=worst_overlap_ms,
    )


def sweep(
    sync_intervals: List[float],
    num_nodes: int = 5,
    duration_s: float = 300.0,
    drift_ppm_std: float = 30.0,
) -> List[Tuple[float, CollisionReport]]:
    """Sweep over re-sync intervals and return (interval_s, report) pairs."""
    out: List[Tuple[float, CollisionReport]] = []
    for si in sync_intervals:
        out.append((si, simulate(
            num_nodes=num_nodes,
            duration_s=duration_s,
            sync_interval_s=si,
            drift_ppm_std=drift_ppm_std,
        )))
    return out


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=int, default=5)
    p.add_argument("--duration-s", type=float, default=600.0)
    p.add_argument("--drift-std-ppm", type=float, default=30.0,
                   help="Per-node drift std dev in ppm (real cheap XTAL ~20-50 ppm)")
    p.add_argument("--guard-ms", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep", action="store_true",
                   help="Sweep over sync intervals and print a comparison table")
    args = p.parse_args()

    if args.sweep:
        print(f"=== Sync-interval sweep ({args.nodes} nodes, {args.duration_s}s, "
              f"drift std={args.drift_std_ppm} ppm, guard={args.guard_ms} ms) ===")
        print(f"{'interval_s':>12} {'collisions':>12} {'delivery':>10} {'worst_ms':>10}")
        for si, rep in sweep(
            sync_intervals=[1, 5, 10, 30, 60, 120, 300],
            num_nodes=args.nodes,
            duration_s=args.duration_s,
            drift_ppm_std=args.drift_std_ppm,
        ):
            print(f"{si:>12.1f} {rep.collisions:>12} {rep.delivery_rate:>10.3f} "
                  f"{rep.worst_overlap_ms:>10.2f}")
        return

    rep = simulate(
        num_nodes=args.nodes,
        duration_s=args.duration_s,
        sync_interval_s=30.0,
        drift_ppm_std=args.drift_std_ppm,
        guard_ms=args.guard_ms,
        seed=args.seed,
    )
    print(rep.as_dict())


if __name__ == "__main__":
    _cli()
