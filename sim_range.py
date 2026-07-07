#!/usr/bin/env python3
"""SignalHop — Physical Layer Range Sweep

Answers: "How far can the acoustic modem actually go in the real world?"

Atmospheric absorption, ambient noise, and transducer limits all cap the
usable range. This script sweeps distance + ambient-noise and reports:

  * Decoded-bit error rate (BER) — fraction of bits decoded wrong
  * Frame success rate (FSR)    — fraction of frames that round-trip
  * Estimated SNR at the receiver
  * Plausible "operational envelope" — distances / noise levels where
    FSR >= 95%

The acoustic model is intentionally simple (free-space attenuation +
thorpe-style atmospheric absorption + Gaussian noise). It is calibrated
against the SignalHop FSK modem (18 kHz / 20 kHz carriers, 500 bps).

Run:  python sim_range.py
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import zlib
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Match the FSK modem constants in core/modem.py
SAMPLE_RATE = 48_000
FREQ_LOW = 18_000.0
FREQ_HIGH = 20_000.0
SYMBOL_RATE = 500.0
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / SYMBOL_RATE)  # 96
NETWORK_ID = b"SH_V1\x00\x00\x00\x00"[:4]
FRAME_HEADER = struct.Struct("!4s BI")  # id, payload_len, crc32
PREAMBLE_LEN = SAMPLES_PER_SYMBOL * 4  # 384 samples


# ---------------------------------------------------------------------------
# Acoustic propagation model
# ---------------------------------------------------------------------------


def atmospheric_absorption_db(distance_m: float, freq_hz: float) -> float:
    """Approximate air absorption (dB) for ultrasound in still air.

    Uses a simplified ISO 9613-1 style model: alpha in dB/m is roughly
    proportional to frequency squared in the 10-30 kHz band at room
    humidity. Good enough for first-order envelope estimation.
    """
    # alpha (dB/m) ~ 1e-10 * f^2 at ~50% RH (rough order of magnitude)
    alpha_db_per_m = 1.0e-10 * (freq_hz ** 2) / 1000.0
    return alpha_db_per_m * distance_m


def geometric_spreading_db(distance_m: float, ref_distance_m: float = 0.1) -> float:
    """Spherical spreading loss vs a near-field reference (dB)."""
    if distance_m <= ref_distance_m:
        return 0.0
    return 20.0 * math.log10(distance_m / ref_distance_m)


def received_snr_db(
    distance_m: float,
    ambient_noise_db_spl: float,
    tx_spl: float = 90.0,
) -> float:
    """Estimate receiver SNR (dB) at the given distance.

    tx_spl:        transmit sound pressure level (dB SPL re 20 µPa)
    ambient_noise: ambient noise floor (dB SPL)
    """
    spread = geometric_spreading_db(distance_m)
    absorption = atmospheric_absorption_db(distance_m, FREQ_HIGH)
    rx_signal = tx_spl - spread - absorption
    return rx_signal - ambient_noise_db_spl


# ---------------------------------------------------------------------------
# Modem helpers (numpy-only, mirrors core/modem.py at the symbol level)
# ---------------------------------------------------------------------------


def _preamble() -> np.ndarray:
    """Rising chirp sync signal of PREAMBLE_LEN samples."""
    samples = PREAMBLE_LEN
    t = np.arange(samples) / SAMPLE_RATE
    chirp = np.sin(2 * np.pi * np.linspace(FREQ_LOW, FREQ_HIGH, samples) * t)
    return chirp.astype(np.float32)


def encode_frame(payload: bytes) -> np.ndarray:
    """Build a full FSK acoustic frame (preamble + header + payload)."""
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = FRAME_HEADER.pack(NETWORK_ID, len(payload), crc)
    bits: List[int] = []
    for byte in header + payload:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    samples_per_bit = SAMPLES_PER_SYMBOL
    symbols: List[np.ndarray] = []
    t = np.arange(samples_per_bit) / SAMPLE_RATE
    for b in bits:
        freq = FREQ_HIGH if b else FREQ_LOW
        symbols.append(np.sin(2 * np.pi * freq * t).astype(np.float32))
    payload_audio = np.concatenate(symbols)
    preamble = _preamble()
    return np.concatenate([preamble, payload_audio])


def decode_frame(signal: np.ndarray) -> Tuple[bool, bytes]:
    """Best-effort FSK demodulator using goertzel. Returns (ok, payload)."""
    bits: List[int] = []
    # skip preamble
    start = PREAMBLE_LEN
    while start + SAMPLES_PER_SYMBOL <= len(signal):
        chunk = signal[start : start + SAMPLES_PER_SYMBOL]
        bits.append(_goertzel_bit(chunk))
        start += SAMPLES_PER_SYMBOL

    if len(bits) < 8 * (FRAME_HEADER.size):
        return False, b""

    # convert bit list to bytes
    decoded = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        decoded.append(byte)

    if len(decoded) < FRAME_HEADER.size:
        return False, b""
    _id, payload_len, crc = FRAME_HEADER.unpack(bytes(decoded[: FRAME_HEADER.size]))
    if len(decoded) < FRAME_HEADER.size + payload_len:
        return False, b""
    payload = bytes(decoded[FRAME_HEADER.size : FRAME_HEADER.size + payload_len])
    return zlib.crc32(payload) & 0xFFFFFFFF == crc, payload


def _goertzel_bit(chunk: np.ndarray) -> int:
    """One-shot goertzel: pick the carrier with more energy."""
    low = _goertzel_energy(chunk, FREQ_LOW)
    high = _goertzel_energy(chunk, FREQ_HIGH)
    return 1 if high > low else 0


def _goertzel_energy(samples: np.ndarray, target_freq: float) -> float:
    """Standard goertzel magnitude^2 for a single frequency bin."""
    n = len(samples)
    k = int(0.5 + (n * target_freq) / SAMPLE_RATE)
    w = (2.0 * math.pi / n) * k
    coeff = 2.0 * math.cos(w)
    s1 = 0.0
    s2 = 0.0
    for x in samples:
        s0 = x + coeff * s1 - s2
        s2 = s1
        s1 = s0
    return s1 * s1 + s2 * s2 - coeff * s1 * s2


# ---------------------------------------------------------------------------
# Channel simulation
# ---------------------------------------------------------------------------


def add_awgn(signal: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    """Additive white Gaussian noise scaled to hit a target SNR."""
    sig_power = float(np.mean(signal ** 2)) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    sigma = math.sqrt(noise_power)
    noise = np.array([rng.gauss(0.0, sigma) for _ in range(len(signal))], dtype=np.float32)
    return signal + noise


def apply_distance(signal: np.ndarray, distance_m: float) -> np.ndarray:
    """Apply geometric spreading + atmospheric absorption (no noise)."""
    spread_db = geometric_spreading_db(distance_m)
    absorption_db = atmospheric_absorption_db(distance_m, FREQ_HIGH)
    attenuation_db = spread_db + absorption_db
    scale = 10.0 ** (-attenuation_db / 20.0)
    return signal * scale


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    distance_m: float
    noise_db_spl: float
    rx_snr_db: float
    bit_error_rate: float
    frame_success_rate: float


def sweep_one(
    distance_m: float,
    noise_db_spl: float,
    trials: int,
    payload: bytes,
    tx_spl: float,
    rng: random.Random,
) -> SweepResult:
    """Run `trials` round-trips at the given (distance, noise) point."""
    snr = received_snr_db(distance_m, noise_db_spl, tx_spl=tx_spl)
    ber_sum = 0.0
    ok_frames = 0

    for _ in range(trials):
        tx = encode_frame(payload)
        rx_clean = apply_distance(tx, distance_m)
        rx_noisy = add_awgn(rx_clean, snr, rng)
        ok, decoded = decode_frame(rx_noisy)
        if ok and decoded == payload:
            ok_frames += 1
        # crude BER: count mismatches over the bytes we got back
        ref = payload
        mismatches = sum(
            bin(a ^ b).count("1") for a, b in zip(decoded.ljust(len(ref), b"\x00"), ref)
        )
        ber_sum += mismatches / max(1, len(ref) * 8)

    return SweepResult(
        distance_m=distance_m,
        noise_db_spl=noise_db_spl,
        rx_snr_db=snr,
        bit_error_rate=ber_sum / trials,
        frame_success_rate=ok_frames / trials,
    )


def run_sweep(
    distances: List[float],
    noise_levels: List[float],
    trials: int,
    payload: bytes,
    tx_spl: float,
    seed: int,
) -> List[SweepResult]:
    """Run a 2-D sweep over (distance, ambient noise) and return all results."""
    rng = random.Random(seed)
    results: List[SweepResult] = []
    total = len(distances) * len(noise_levels)
    done = 0
    t0 = time.time()
    for d in distances:
        for n in noise_levels:
            r = sweep_one(d, n, trials, payload, tx_spl, rng)
            results.append(r)
            done += 1
    elapsed = time.time() - t0
    print(
        f"[sim_range] swept {done}/{total} (distance x noise) points in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return results


def render_table(results: List[SweepResult]) -> str:
    """ASCII table: rows = distance, columns = ambient noise, cells = FSR%."""
    distances = sorted({r.distance_m for r in results})
    noises = sorted({r.noise_db_spl for r in results})
    by_key = {(r.distance_m, r.noise_db_spl): r for r in results}

    lines: List[str] = []
    header = "FSR %    " + "  ".join(f"n={int(n):>3}dB" for n in noises)
    lines.append(header)
    lines.append("-" * len(header))
    for d in distances:
        row = [f"d={d:>4.1f}m"]
        for n in noises:
            r = by_key.get((d, n))
            if r is None:
                row.append("  --  ")
            else:
                row.append(f"{r.frame_success_rate * 100:>5.0f}")
        lines.append("    ".join(row))
    return "\n".join(lines)


def render_envelope(results: List[SweepResult], threshold: float = 0.95) -> str:
    """Pretty-print the operational envelope where FSR >= threshold."""
    lines: List[str] = [f"Operational envelope (FSR >= {threshold * 100:.0f}%):"]
    by_noise: dict = {}
    for r in results:
        by_noise.setdefault(r.noise_db_spl, []).append(r)
    for n in sorted(by_noise):
        passing = [r for r in by_noise[n] if r.frame_success_rate >= threshold]
        if not passing:
            lines.append(f"  noise={n:.0f} dB SPL: <no range>")
        else:
            max_d = max(passing, key=lambda r: r.distance_m).distance_m
            lines.append(f"  noise={n:.0f} dB SPL: up to {max_d:.1f} m")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SignalHop physical-layer range sweep")
    parser.add_argument("--distances", type=str, default="0.5,1,2,5,10,20,30",
                        help="comma-separated distances in meters")
    parser.add_argument("--noises", type=str, default="20,40,60,80",
                        help="comma-separated ambient noise levels in dB SPL")
    parser.add_argument("--trials", type=int, default=20,
                        help="trials per (distance, noise) cell")
    parser.add_argument("--payload", type=str, default="SIGNALHOP PKT",
                        help="payload bytes to round-trip")
    parser.add_argument("--tx-spl", type=float, default=90.0,
                        help="transmit SPL (dB)")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)

    distances = [float(x) for x in args.distances.split(",")]
    noises = [float(x) for x in args.noises.split(",")]
    payload = args.payload.encode("utf-8")

    results = run_sweep(
        distances=distances,
        noise_levels=noises,
        trials=args.trials,
        payload=payload,
        tx_spl=args.tx_spl,
        seed=args.seed,
    )
    print()
    print("=" * 72)
    print("SignalHop — Physical Layer Range Sweep")
    print(f"  payload  : {payload!r}  ({len(payload)} bytes)")
    print(f"  trials   : {args.trials} per cell")
    print(f"  tx SPL   : {args.tx_spl} dB")
    print("=" * 72)
    print()
    print(render_table(results))
    print()
    print(render_envelope(results))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
