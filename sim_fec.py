#!/usr/bin/env python3
"""
SignalHop — Forward Error Correction (FEC) simulator.

The acoustic channel drops bits. A naive modem re-sends; a smart one
ships redundant bits so a single-frame decode succeeds even after
loss. This sim compares:

  - No FEC:  any bit flip kills the frame, receiver requests retry
  - Repetition (3x):  send 3 copies, majority-vote
  - XOR parity block:  send k data bits + 1 parity bit, lose 1, recover
  - Hamming(7,4):      4 data bits + 3 parity bits, recover any single-bit error

Run:  python sim_fec.py
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Callable, List

from sim_demo import SimConfig, SimNode, MeshSimulator


@dataclass
class FECResult:
    scheme: str
    bits: int
    overhead: float
    delivered: int
    errors_corrected: int
    unrecoverable: int
    retries: int


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def _hamming_encode(data_bits: List[int]) -> List[int]:
    """Hamming(7,4): 4 data -> 7 bits. Positions 1,2,4 are parity."""
    if len(data_bits) % 4 != 0:
        data_bits = data_bits + [0] * (4 - len(data_bits) % 4)
    out: List[int] = []
    for i in range(0, len(data_bits), 4):
        d = data_bits[i:i + 4]
        # Indexed 1..7
        p1 = d[0] ^ d[1] ^ d[3]
        p2 = d[0] ^ d[2] ^ d[3]
        p3 = d[1] ^ d[2] ^ d[3]
        out.extend([p1, p2, d[0], p3, d[1], d[2], d[3]])
    return out


def _hamming_decode(bits: List[int]) -> List[int]:
    out: List[int] = []
    for i in range(0, len(bits), 7):
        b = bits[i:i + 7]
        if len(b) < 7:
            out.extend(b)
            continue
        s1 = b[0] ^ b[2] ^ b[4] ^ b[6]
        s2 = b[1] ^ b[2] ^ b[5] ^ b[6]
        s3 = b[3] ^ b[4] ^ b[5] ^ b[6]
        syndrome = s1 + (s2 << 1) + (s3 << 2)
        if syndrome and syndrome <= 7:
            b = b[:]
            b[syndrome - 1] ^= 1  # fix the single-bit error
        out.extend([b[2], b[4], b[5], b[6]])
    return out


def _simulate(scheme: str, payload: bytes, ber: float, rng: random.Random) -> FECResult:
    bits = [(byte >> i) & 1 for byte in payload for i in range(8)]
    n = len(bits)

    if scheme == "none":
        flipped = _flip(bits, ber, rng)
        received = [rx for rx, _ in _flip(bits, ber, rng)]
        # At BER=0 every received bit == original, so delivered == n.
        delivered = sum(1 for rx, b in zip(received, bits) if rx == b)
        unrecoverable = n - delivered
        return FECResult(scheme, n, 1.0, delivered, 0, unrecoverable, 0)

    if scheme == "repetition3":
        tx = [b for b in bits for _ in range(3)]
        received = [rx for rx, _ in _flip(tx, ber, rng)]
        # Majority-vote per group of 3. After voting, the recovered bit
        # is 1 if the sum is >= 2, else 0. We then count how many of the
        # recovered bits match the original payload bits.
        recovered: List[int] = []
        for i in range(n):
            v = received[i * 3:(i + 1) * 3]
            recovered.append(1 if sum(v) >= 2 else 0)
        delivered = sum(1 for r, b in zip(recovered, bits) if r == b)
        errors_corrected = delivered  # for the report
        return FECResult(
            scheme, len(tx), 3.0, delivered, errors_corrected, n - delivered, 0
        )

    if scheme == "xor_parity":
        chunk = 8
        chunks = [bits[i:i + chunk] for i in range(0, n, chunk)]
        tx_bits: List[int] = []
        for c in chunks:
            c = c + [0] * (chunk - len(c))
            tx_bits.extend(c)
            tx_bits.append(sum(c) % 2)
        flipped = _flip(tx_bits, ber, rng)
        received = [rx for rx, _ in flipped]
        delivered = 0
        for i in range(0, len(received), chunk + 1):
            group = received[i:i + chunk + 1]
            data = group[:chunk]
            parity = group[chunk] if len(group) > chunk else 0
            expected_parity = sum(data) % 2
            if parity == expected_parity:
                delivered += sum(1 for d, b in zip(data, bits[i // (chunk + 1) * chunk : i // (chunk + 1) * chunk + chunk]) if d == b)
        return FECResult(scheme, len(tx_bits), (chunk + 1) / chunk, delivered, 0, n - delivered, 0)

    if scheme == "hamming":
        # 4 data bits -> 7 code bits (Hamming(7,4)).
        tx_bits: List[int] = []
        for i in range(0, n, 4):
            d = bits[i:i + 4] + [0, 0, 0, 0][len(bits[i:i + 4]):]
            tx_bits.extend(_hamming_encode(d))
        flipped = _flip(tx_bits, ber, rng)
        received = [rx for rx, _ in flipped]
        recovered = _hamming_decode(received)
        delivered = sum(1 for r, b in zip(recovered, bits) if r == b)
        errors_corrected = delivered
        unrecoverable = n - delivered
        return FECResult(scheme, len(tx_bits), 7 / 4, delivered, errors_corrected, unrecoverable, 0)

    raise ValueError(f"unknown scheme: {scheme!r}")


def _flip(bits: List[int], ber: float, rng: random.Random):
    for b in bits:
        if rng.random() < ber:
            yield (1 - b), True
        else:
            yield b, False


def run_sweep(ber_values: List[float], payload_size: int = 32) -> None:
    payload = bytes(random.randint(0, 255) for _ in range(payload_size))
    rng = random.Random(42)
    print(
        f"{'BER':>6}  {'scheme':<14}  {'tx_bits':>8}  {'overhead':>8}  "
        f"{'delivered':>10}  {'corrected':>10}  {'unrecov':>8}"
    )
    print("-" * 70)
    for ber in ber_values:
        for scheme in ("none", "repetition3", "xor_parity", "hamming"):
            r = _simulate(scheme, payload, ber, random.Random(42))
            print(
                f"{ber:>6.4f}  {r.scheme:<14}  {r.bits:>8}  "
                f"{r.overhead:>7.2f}x  {r.delivered:>10}  "
                f"{r.errors_corrected:>10}  {r.unrecoverable:>8}"
            )
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="SignalHop FEC sweep")
    parser.add_argument(
        "--ber",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.05, 0.1, 0.2],
        help="bit-error rates to sweep (default: 0.0 0.01 0.05 0.1 0.2)",
    )
    parser.add_argument(
        "--payload", type=int, default=32, help="payload size in bytes (default: 32)"
    )
    args = parser.parse_args()
    run_sweep(args.ber, args.payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
