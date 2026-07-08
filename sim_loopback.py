#!/usr/bin/env python3
"""
sim_loopback.py — Round-trip self-test for the SignalHop acoustic modem.

Encode → transmit → receive → decode, with configurable per-trial payload
size and a CLI-friendly pass/fail summary. Useful for sanity-checking the
modem after a code change without dragging a real speaker and microphone
into the loop. All synthetic; no audio hardware needed.

Usage:
    python3 sim_loopback.py
    python3 sim_loopback.py --bytes 64 --trials 10
    python3 sim_loopback.py --bytes 256 --trials 3 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

# Make the `core` package importable when this file is run directly.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core.modem import AcousticModem  # noqa: E402


def _run_trial(modem: AcousticModem, n_bytes: int, rng: random.Random) -> tuple[bool, float]:
    """One round-trip. Returns (ok, seconds)."""
    payload = bytes(rng.getrandbits(8) for _ in range(n_bytes))
    t0 = time.perf_counter()
    waveform = modem.tx(payload)
    decoded = modem.rx(waveform)
    elapsed = time.perf_counter() - t0
    return (decoded == payload), elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--bytes", type=int, default=32, help="payload size per trial (default 32)")
    p.add_argument("--trials", type=int, default=5, help="number of round-trips (default 5)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    args = p.parse_args()

    if args.bytes < 1 or args.bytes > 256:
        print("error: --bytes must be in 1..256", file=sys.stderr)
        return 2
    if args.trials < 1:
        print("error: --trials must be >= 1", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    modem = AcousticModem()

    print(f"SignalHop loopback self-test")
    print(f"  payload:  {args.bytes} bytes per trial")
    print(f"  trials:   {args.trials}")
    print(f"  seed:     {args.seed}")
    print(f"  carriers: {modem.cfg.carrier_low} Hz / {modem.cfg.carrier_high} Hz @ {modem.cfg.sample_rate} Hz")
    print()

    passed = 0
    total_ms = 0.0
    for i in range(1, args.trials + 1):
        ok, elapsed = _run_trial(modem, args.bytes, rng)
        total_ms += elapsed * 1000
        mark = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  trial {i:>2d}: {mark}  ({elapsed * 1000:7.2f} ms)")

    avg_ms = total_ms / args.trials
    print()
    print(f"  result:  {passed}/{args.trials} round-trips OK")
    print(f"  avg:     {avg_ms:.2f} ms per round-trip")
    return 0 if passed == args.trials else 1


if __name__ == "__main__":
    sys.exit(main())
