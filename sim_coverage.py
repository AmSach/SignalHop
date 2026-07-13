#!/usr/bin/env python3
"""
SignalHop — Coverage Area Estimator
====================================

Given a 2D deployment area and a transmit radius, estimate the number of
nodes reachable in 1, 2, and 3 hops. This is the planning tool you'd run
before physically placing acoustic mesh nodes in a warehouse, vineyard,
or search-and-rescue grid.

Uses a 1/r^2 path loss approximation (acoustic) and a uniform random
distribution of receivers. Repeats the trial many times to smooth the
estimate and reports mean and 95% confidence interval.

Run:  python sim_coverage.py
Test: python tests/test_coverage.py

Example:
    python sim_coverage.py --area 200 --nodes 25 --radius 30 \
        --hops 3 --trials 200 --seed 42
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class CoverageResult:
    hops: int
    reach_mean: float
    reach_stdev: float
    reach_ci95_lo: float
    reach_ci95_hi: float
    reach_fraction: float        # mean reach / total nodes


def _dist2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _hop_reachable(positions: List[Tuple[float, float]], src: int, radius: float,
                   hops: int) -> set:
    """BFS up to `hops` edges from src, returning all reachable node indices."""
    if hops <= 0:
        return {src}
    reached = {src}
    frontier = {src}
    for _ in range(hops):
        next_frontier = set()
        for u in frontier:
            for v, p in enumerate(positions):
                if v in reached:
                    continue
                if _dist2d(positions[u], p) <= radius:
                    next_frontier.add(v)
        if not next_frontier:
            break
        reached |= next_frontier
        frontier = next_frontier
    return reached


def estimate_coverage(area_m: float, n_nodes: int, radius_m: float,
                      hops: int, trials: int = 200,
                      seed: int | None = None) -> CoverageResult:
    """Monte-Carlo coverage estimator.

    Args:
        area_m: square area side length in meters
        n_nodes: total nodes placed uniformly at random
        radius_m: one-hop acoustic link radius
        hops: max hops from a "gateway" (node 0) to count as covered
        trials: number of independent random placements
        seed: optional RNG seed for reproducibility
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2")
    if area_m <= 0 or radius_m <= 0 or hops < 1:
        raise ValueError("area_m, radius_m must be > 0 and hops >= 1")
    if trials < 1:
        raise ValueError("trials must be >= 1")

    rng = random.Random(seed)
    reaches: List[int] = []
    for _ in range(trials):
        positions = [(rng.uniform(0, area_m), rng.uniform(0, area_m))
                     for _ in range(n_nodes)]
        reached = _hop_reachable(positions, src=0, radius=radius_m, hops=hops)
        reaches.append(len(reached))

    mean = statistics.fmean(reaches)
    stdev = statistics.pstdev(reaches) if len(reaches) > 1 else 0.0
    if len(reaches) > 1:
        # 95% CI for the mean under approximate normality
        ci_half = 1.96 * stdev / math.sqrt(len(reaches))
    else:
        ci_half = 0.0

    return CoverageResult(
        hops=hops,
        reach_mean=mean,
        reach_stdev=stdev,
        reach_ci95_lo=mean - ci_half,
        reach_ci95_hi=mean + ci_half,
        reach_fraction=mean / n_nodes,
    )


def sweep_radius(area_m: float, n_nodes: int, radii: List[float],
                 hops: int = 3, trials: int = 200,
                 seed: int | None = None) -> List[CoverageResult]:
    """Run estimate_coverage for each radius in `radii`."""
    return [estimate_coverage(area_m, n_nodes, r, hops, trials, seed)
            for r in radii]


def _format_table(results: List[CoverageResult], n_nodes: int) -> str:
    header = (f"  {'hops':>4}  {'radius':>8}  {'reach':>10}  "
              f"{'std':>8}  {'95% CI':>20}  {'frac':>6}")
    sep = "  " + "-" * (len(header) - 2)
    rows = [header, sep]
    for r in results:
        ci = f"[{r.reach_ci95_lo:5.2f}, {r.reach_ci95_hi:5.2f}]"
        rows.append(
            f"  {r.hops:>4}  {'-':>8}  {r.reach_mean:>7.2f}/{n_nodes:<2}  "
            f"{r.reach_stdev:>8.2f}  {ci:>20}  {r.reach_fraction:>5.1%}"
        )
    return "\n".join(rows)


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "coverage")
    p.add_argument("--area", type=float, default=100.0,
                   help="square area side length, meters (default 100)")
    p.add_argument("--nodes", type=int, default=20,
                   help="number of nodes to place (default 20)")
    p.add_argument("--radius", type=float, default=25.0,
                   help="one-hop acoustic link radius, meters (default 25)")
    p.add_argument("--hops", type=int, default=3,
                   help="max hops from gateway (default 3)")
    p.add_argument("--trials", type=int, default=200,
                   help="Monte-Carlo trials (default 200)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for reproducibility")
    p.add_argument("--sweep", action="store_true",
                   help="sweep hops 1, 2, 3 and report a table")
    args = p.parse_args()

    if args.sweep:
        results = [estimate_coverage(args.area, args.nodes, args.radius, h,
                                     args.trials, args.seed)
                   for h in (1, 2, 3)]
    else:
        results = [estimate_coverage(args.area, args.nodes, args.radius,
                                     args.hops, args.trials, args.seed)]

    print(f"Coverage estimate over {args.trials} random placements")
    print(f"  area = {args.area} x {args.area} m   "
          f"nodes = {args.nodes}   radius = {args.radius} m")
    print(_format_table(results, args.nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
