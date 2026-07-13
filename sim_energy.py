#!/usr/bin/env python3
"""
SignalHop — Energy-Aware Sleep Scheduler
========================================

Acoustic mesh nodes are often battery powered (ESP32, solar sensors, field
deployed wildlife trackers). Listening 100% of the time is the single biggest
power sink — the analog front-end and demodulator burn orders of magnitude
more energy than a sleeping CPU.

This sim implements a S-MAC-style duty-cycling scheduler: each node wakes
for a short listen window every cycle, exchanges any pending frames, then
returns to deep sleep. We measure:

    * mean current draw (mA) for a given duty cycle
    * packet delivery ratio vs. duty cycle
    * end-to-end latency under sleep schedule
    * how sync-drift between two nodes with cheap RC clocks causes
      listen-window overlap loss without periodic re-sync

The model is intentionally simple (no CPU cycles, no radio on-time, just
listen vs. sleep current and a periodic wake) — it's enough to reason about
duty-cycle vs. reliability tradeoffs and to choose sane defaults for
deployed firmware.

Run:  python sim_energy.py
Test: python tests/test_energy.py
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple


# ---- battery / radio model --------------------------------------------------

# Currents (mA) at 3.3V
LISTEN_CURRENT_MA = 28.0     # ESP32 + analog mic preamp + demod
SLEEP_CURRENT_MA = 0.008    # ESP32 deep sleep with RTC

# Timing (ms)
FRAME_LEN_MS = 50.0          # one acoustic frame (header + payload)
WAKE_OVERHEAD_MS = 5.0       # time to wake, lock, and re-enter sleep
SYNC_BEACON_MS = 30.0        # chirp sync preamble at the start of every wake


# ---- schedule ---------------------------------------------------------------

@dataclass
class DutyCycleSchedule:
    """Periodic wake: every cycle_ms, stay awake for listen_ms."""
    cycle_ms: float
    listen_ms: float
    jitter_ms: float = 0.0    # random per-node wake offset within [0, jitter_ms)

    def is_listening(self, t_ms: float, node_offset_ms: float = 0.0) -> bool:
        t_eff = (t_ms - node_offset_ms) % self.cycle_ms
        # Listen window centered at t=0 of the cycle for simplicity
        half = self.listen_ms / 2.0
        return t_eff < half or t_eff > self.cycle_ms - half

    def mean_current_ma(self) -> float:
        """Duty cycle weighted average of listen vs. sleep current."""
        duty = self.listen_ms / self.cycle_ms
        return LISTEN_CURRENT_MA * duty + SLEEP_CURRENT_MA * (1.0 - duty)


# ---- delivery simulator -----------------------------------------------------

@dataclass
class SimResult:
    duty_cycle: float
    listen_ms: float
    cycle_ms: float
    mean_current_ma: float
    pdr: float                # packet delivery ratio [0,1]
    mean_latency_ms: float
    clock_drift_loss: float   # fraction of frames lost to wake misalignment
    frames_sent: int
    frames_received: int


def simulate(
    schedule: DutyCycleSchedule,
    n_nodes: int = 6,
    duration_s: float = 60.0,
    traffic_per_s: float = 1.0,
    clock_drift_ppm: float = 30.0,
    rng_seed: int = 7,
) -> SimResult:
    """Simulate broadcast traffic under a periodic duty cycle.

    Each sender wakes on its own offset and broadcasts. A receiver captures
    the frame only if it is in its listen window when the frame arrives on
    the wire. We add per-node clock drift so the listen window slides over
    time and occasionally misses sync windows.
    """
    rng = random.Random(rng_seed)

    # Per-node wake offset so neighbours don't all wake at the same instant.
    offsets = [rng.uniform(0, schedule.cycle_ms) for _ in range(n_nodes)]

    # Clock drift (ppm) — at 30 ppm, a node drifts ~1.8ms per minute.
    # Bake it into a per-node effective cycle period.
    periods = [schedule.cycle_ms * (1.0 + (rng.uniform(-1, 1) * clock_drift_ppm * 1e-6))
               for _ in range(n_nodes)]

    dt_ms = 10.0
    n_steps = int(duration_s * 1000 / dt_ms)
    frames_per_step = traffic_per_s * dt_ms / 1000.0

    sent = 0
    delivered = 0
    latencies_ms: List[float] = []
    wake_misses = 0
    wake_hits = 0

    for step in range(n_steps):
        t_ms = step * dt_ms
        # Per-step drift: each node's clock has advanced by period offset.
        for i in range(n_nodes):
            # Effective clock — adds drift proportional to absolute time.
            drift_ms = (periods[i] - schedule.cycle_ms) * (t_ms / schedule.cycle_ms)
            node_t = t_ms + drift_ms

            # Is this node in its listen window?
            t_in_cycle = (node_t - offsets[i]) % schedule.cycle_ms
            listening = t_in_cycle < schedule.listen_ms

            if listening:
                wake_hits += 1
            else:
                wake_misses += 1

        # Generate traffic: a Poisson-ish burst from random nodes
        if rng.random() < frames_per_step * n_nodes:
            sender = rng.randrange(n_nodes)
            sent += 1
            # Receiver must be in its listen window when the frame hits the wire.
            # Frame duration = FRAME_LEN_MS, so any receiver whose window overlaps
            # the frame start captures it. We model "in window" as a Bernoulli
            # trial with probability = listen_ms / cycle_ms.
            any_received = False
            for i in range(n_nodes):
                if i == sender:
                    continue
                p_listen = schedule.listen_ms / schedule.cycle_ms
                if rng.random() < p_listen:
                    if not any_received:
                        delivered += 1
                        any_received = True
                    # Latency = wait until the receiver's next wake.
                    t_in_cycle = (t_ms - offsets[i]) % periods[i]
                    wait_ms = periods[i] - t_in_cycle
                    latencies_ms.append(wait_ms)

    pdr = delivered / sent if sent else 0.0
    mean_lat = (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0
    drift_loss = wake_misses / (wake_hits + wake_misses) if (wake_hits + wake_misses) else 0.0
    duty = schedule.listen_ms / schedule.cycle_ms

    return SimResult(
        duty_cycle=duty,
        listen_ms=schedule.listen_ms,
        cycle_ms=schedule.cycle_ms,
        mean_current_ma=schedule.mean_current_ma(),
        pdr=pdr,
        mean_latency_ms=mean_lat,
        clock_drift_loss=drift_loss,
        frames_sent=sent,
        frames_received=delivered,
    )


# ---- sweep ------------------------------------------------------------------

def sweep_duty_cycles(
    cycles: List[float] = None,
    listen_ms: float = 30.0,
    duration_s: float = 30.0,
    clock_drift_ppm: float = 30.0,
) -> List[SimResult]:
    """Run a sweep across multiple cycle lengths at a fixed listen window."""
    if cycles is None:
        cycles = [50, 100, 200, 400, 800, 1600]
    results: List[SimResult] = []
    for c in cycles:
        s = DutyCycleSchedule(cycle_ms=c, listen_ms=listen_ms)
        results.append(simulate(
            s,
            duration_s=duration_s,
            clock_drift_ppm=clock_drift_ppm,
        ))
    return results


# ---- CLI --------------------------------------------------------------------

def _fmt_row(r: SimResult) -> str:
    return (f"cycle={r.cycle_ms:>5.0f}ms  duty={r.duty_cycle*100:5.2f}%  "
            f"I={r.mean_current_ma:6.3f}mA  "
            f"PDR={r.pdr*100:5.1f}%  "
            f"lat={r.mean_latency_ms:6.0f}ms")


def main() -> int:
    p = argparse.ArgumentParser(description="Acoustic mesh energy model")
    p.add_argument("--listen-ms", type=float, default=30.0,
                   help="Listen window per cycle (ms)")
    p.add_argument("--duration-s", type=float, default=30.0,
                   help="Simulation duration (s)")
    p.add_argument("--drift-ppm", type=float, default=30.0,
                   help="Per-node clock drift (ppm)")
    p.add_argument("--traffic", type=float, default=1.0,
                   help="Per-node traffic rate (frames/s)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    print(f"SignalHop energy model — listen_ms={args.listen_ms} drift_ppm={args.drift_ppm}\n")
    print(f"  LISTEN current = {LISTEN_CURRENT_MA} mA")
    print(f"  SLEEP  current = {SLEEP_CURRENT_MA} mA")
    print(f"  Frame length   = {FRAME_LEN_MS} ms\n")

    results = sweep_duty_cycles(
        listen_ms=args.listen_ms,
        duration_s=args.duration_s,
        clock_drift_ppm=args.drift_ppm,
    )
    print("Duty-cycle sweep:")
    for r in results:
        print("  " + _fmt_row(r))

    # 2000 mAh battery life estimate
    print("\n2000 mAh battery life (listening-only schedule):")
    for r in results:
        hours = 2000.0 / r.mean_current_ma
        days = hours / 24.0
        print(f"  cycle={r.cycle_ms:>5.0f}ms  ->  {days:6.1f} days  ({r.mean_current_ma:6.3f} mA)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
