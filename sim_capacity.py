#!/usr/bin/env python3
"""SignalHop — Mesh Capacity Sweep

Answers: "How does the mesh scale?" — at what node count does packet
delivery start collapsing, and how does average routing-table size
grow with density?

Run: python sim_capacity.py
"""
import argparse
import statistics
import sys
import contextlib
import io

from sim_demo import SimConfig, MeshSimulator


def run_once(num_nodes: int, area: float, tx_range: float, seed: int) -> dict:
    cfg = SimConfig(
        num_nodes=num_nodes,
        area_width=area,
        area_height=area,
        tx_range=tx_range,
        simulation_time=30.0,
    )
    import random
    random.seed(seed)
    sim = MeshSimulator(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        stats = sim.run()
    delivered = stats.get("total_packets_delivered", 0)
    sent = stats.get("total_packets_sent", 0)
    rate = (delivered / sent * 100.0) if sent else 0.0
    table_sizes = [len(n.routing_table) for n in sim.nodes]
    return {
        "delivered_rate": rate,
        "delivered": delivered,
        "sent": sent,
        "avg_routes": statistics.mean(table_sizes) if table_sizes else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", type=float, default=80.0,
                        help="square area side length (m)")
    parser.add_argument("--tx-range", type=float, default=20.0,
                        help="acoustic tx range (m)")
    parser.add_argument("--trials", type=int, default=3,
                        help="trials per node count (random seed per trial)")
    args = parser.parse_args()

    sweep = [4, 6, 8, 12, 16, 24, 32, 48]
    print(f"{'nodes':>6}  {'trials':>6}  {'avg delivery':>12}  "
          f"{'avg routes/node':>16}")
    print("-" * 56)
    for n in sweep:
        rates = []
        routes = []
        for t in range(args.trials):
            r = run_once(n, args.area, args.tx_range, seed=1000 + 7 * n + t)
            rates.append(r["delivered_rate"])
            routes.append(r["avg_routes"])
        avg_rate = statistics.mean(rates)
        avg_routes = statistics.mean(routes)
        print(f"{n:>6}  {args.trials:>6}  {avg_rate:>10.1f}%   "
              f"{avg_routes:>14.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
