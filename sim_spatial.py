#!/usr/bin/env python3
"""
SignalHop — Spatial Channel Simulator
Adds multipath / Rayleigh-fading room acoustics to the loopback channel so the
acoustic stack can be exercised against a non-trivial impulse response without
leaving the host. Models:

  * direct path  (line-of-sight, distance attenuation 1/r^2)
  * one floor reflection (delay = 2*h/c, -3 dB)
  * one wall reflection  (delay = 2*d_wall/c, -6 dB)
  * Rayleigh fading taps  (jakes-sum, 8 oscillators, normalised)

The resulting impulse response is convolved with the transmitted waveform and
white Gaussian noise is added to reach a target SNR. The simulator is pure
NumPy and runs end-to-end in well under a second on a laptop.

Usage:
    python3 sim_spatial.py                 # default: 12m indoor, SNR=20dB
    python3 sim_spatial.py --snr 10 --range 5 --reflections 3
    python3 sim_spatial.py --no-fading
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import numpy as np

# Acoustic constants
SPEED_OF_SOUND = 343.0      # m/s in dry air at 20°C
FREQ_LOW = 18_000.0
FREQ_HIGH = 20_000.0
SAMPLE_RATE = 48_000


@dataclass
class RoomConfig:
    """Rough dimensions of an indoor space."""
    length: float = 8.0     # metres
    width: float = 5.0
    height: float = 2.7
    wall_attenuation_db: float = 6.0
    floor_attenuation_db: float = 3.0


@dataclass
class ChannelConfig:
    """End-to-end acoustic channel configuration."""
    range_m: float = 12.0
    snr_db: float = 20.0
    reflections: int = 2          # 0=direct, 1=+floor, 2=+wall
    fading_taps: int = 8
    seed: int = 1337

    # Derived / runtime
    impulse: np.ndarray | None = None
    delay_spread_s: float = 0.0
    path_loss_db: float = 0.0


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _distance_attenuation_db(distance_m: float) -> float:
    """Free-space 1/r^2 attenuation in dB, normalised to 1m reference."""
    if distance_m < 0.1:
        distance_m = 0.1
    return 20.0 * math.log10(distance_m)


def build_impulse_response(room: RoomConfig, ch: ChannelConfig) -> np.ndarray:
    """Synthesise a discrete-time impulse response for the configured room.

    Returns a NumPy array of length ceil(max_delay * sample_rate) + 1.
    Taps are real-valued; phase is randomised so successive frames decorrelate.
    """
    rng = np.random.default_rng(ch.seed)
    max_delay = ch.range_m / SPEED_OF_SOUND
    taps: list[tuple[int, float]] = []

    # 1. Direct path
    direct_db = _distance_attenuation_db(ch.range_m)
    direct_lin = _db_to_linear(-direct_db)
    direct_delay_samples = int(round(ch.range_m / SPEED_OF_SOUND * SAMPLE_RATE))
    taps.append((direct_delay_samples, direct_lin))
    ch.path_loss_db = direct_db

    # 2. Floor reflection: extra path length ~ 2 * height
    if ch.reflections >= 1:
        floor_path_m = 2.0 * math.sqrt((ch.range_m / 2.0) ** 2 + room.height ** 2)
        floor_db = _distance_attenuation_db(floor_path_m) + room.floor_attenuation_db
        floor_delay = int(round(floor_path_m / SPEED_OF_SOUND * SAMPLE_RATE))
        taps.append((floor_delay, _db_to_linear(-floor_db)))

    # 3. Wall reflection: extra path length ~ 2 * perpendicular offset
    if ch.reflections >= 2:
        wall_path_m = 2.0 * math.sqrt((ch.range_m / 2.0) ** 2 + (room.width / 2.0) ** 2)
        wall_db = _distance_attenuation_db(wall_path_m) + room.wall_attenuation_db
        wall_delay = int(round(wall_path_m / SPEED_OF_SOUND * SAMPLE_RATE))
        taps.append((wall_delay, _db_to_linear(-wall_db)))

    max_delay = max(t[0] for t in taps) + 32  # 32-sample tail for late reverberation
    h = np.zeros(max_delay + 1, dtype=np.float32)

    for delay, amp in taps:
        if delay < len(h):
            # Random phase models the small Doppler from moving objects
            phase = rng.uniform(0, 2 * math.pi)
            h[delay] += amp * math.cos(phase)

    # 4. Rayleigh fading tail (Jakes sum)
    if ch.fading_taps > 0:
        late_start = max(t[0] for t in taps) + 8
        n_late = min(ch.fading_taps, len(h) - late_start - 1)
        if n_late > 0:
            osc = 8
            t_axis = np.arange(n_late) / SAMPLE_RATE
            rayleigh = np.zeros(n_late, dtype=np.float32)
            for k in range(1, osc + 1):
                theta = rng.uniform(0, 2 * math.pi)
                f_d = 5.0  # 5 Hz max Doppler (slow walking)
                rayleigh += np.cos(2 * math.pi * f_d * t_axis * math.cos(2 * math.pi * k / osc) + theta)
            rayleigh = rayleigh / math.sqrt(osc)  # normalise power
            tail = rayleigh * _db_to_linear(-ch.path_loss_db - 12.0)
            h[late_start:late_start + n_late] += tail

    # Estimate RMS delay spread
    energy = h ** 2
    total_energy = energy.sum()
    if total_energy > 0:
        t = np.arange(len(h)) / SAMPLE_RATE
        mean_t = float((t * energy).sum() / total_energy)
        mean_t2 = float((t * t * energy).sum() / total_energy)
        ch.delay_spread_s = math.sqrt(max(mean_t2 - mean_t * mean_t, 0.0))
    else:
        ch.delay_spread_s = 0.0

    return h


def apply_channel(signal: np.ndarray, impulse: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Convolve signal with the room impulse response and add AWGN to hit SNR."""
    rng = np.random.default_rng(seed)
    # mode='same' keeps the convolution aligned with the input so symbol windows
    # line up with where the modulator placed them.
    received = np.convolve(signal, impulse, mode="full").astype(np.float32)

    sig_power = float(np.mean(received ** 2))
    if sig_power <= 0:
        return received
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, math.sqrt(noise_power), size=received.shape).astype(np.float32)
    return received + noise


def simulate(symbols: int = 64, ch: ChannelConfig | None = None,
             room: RoomConfig | None = None, seed: int = 0) -> dict:
    """Run a full spatial channel simulation and return a stats dict."""
    ch = ch or ChannelConfig()
    room = room or RoomConfig()

    # Encode a known FSK waveform (alternating bits) and convolve.
    samples_per_symbol = SAMPLE_RATE // 500
    bits = (np.arange(symbols) % 2).astype(np.int8)
    wave = np.zeros(symbols * samples_per_symbol, dtype=np.float32)
    for i, b in enumerate(bits):
        f = FREQ_HIGH if b else FREQ_LOW
        t = np.arange(samples_per_symbol) / SAMPLE_RATE
        seg = np.sin(2 * math.pi * f * t)
        wave[i * samples_per_symbol:(i + 1) * samples_per_symbol] = seg.astype(np.float32)

    impulse = build_impulse_response(room, ch)
    ch.impulse = impulse
    received = apply_channel(wave, impulse, ch.snr_db, seed=seed)
    # The first channel_delay_samples of the received waveform are corrupted
    # by the channel's own propagation delay; skip past them before demodulating.
    if impulse.any():
        pass
    direct_path_samples = int(round(ch.range_m / SPEED_OF_SOUND * SAMPLE_RATE))
    channel_delay = direct_path_samples
    received = received[channel_delay:]

    # Crude bit-error estimate via Goertzel energy comparison
    sym_energy_low = []
    sym_energy_high = []
    for i in range(symbols):
        seg = received[i * samples_per_symbol:(i + 1) * samples_per_symbol]
        n = len(seg)
        k_low = int(0.5 + n * FREQ_LOW / SAMPLE_RATE)
        k_high = int(0.5 + n * FREQ_HIGH / SAMPLE_RATE)
        w_low = 2 * math.pi * k_low / n
        w_high = 2 * math.pi * k_high / n
        c_low = 2 * math.cos(w_low)
        c_high = 2 * math.cos(w_high)
        s1 = s2 = 0.0
        for x in seg:
            s = x + c_low * s1 - s2
            s2, s1 = s1, s
        e_low = s1 * s1 + s2 * s2 - c_low * s1 * s2
        s1 = s2 = 0.0
        for x in seg:
            s = x + c_high * s1 - s2
            s2, s1 = s1, s
        e_high = s1 * s1 + s2 * s2 - c_high * s1 * s2
        sym_energy_low.append(e_low)
        sym_energy_high.append(e_high)

    decoded = [0 if lo > hi else 1 for lo, hi in zip(sym_energy_low, sym_energy_high)]
    errors = sum(1 for a, b in zip(bits.tolist(), decoded) if a != b)
    ber = errors / symbols

    return {
        "range_m": ch.range_m,
        "snr_db": ch.snr_db,
        "reflections": ch.reflections,
        "path_loss_db": round(ch.path_loss_db, 2),
        "delay_spread_ms": round(ch.delay_spread_s * 1000.0, 3),
        "impulse_length_samples": len(impulse),
        "channel_delay_samples": direct_path_samples,
        "symbol_errors": errors,
        "symbols": symbols,
        "ber": round(ber, 4),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SignalHop spatial channel simulator")
    p.add_argument("--snr", type=float, default=20.0, help="Signal-to-noise ratio in dB")
    p.add_argument("--range", type=float, default=12.0, help="Transmitter-receiver range (metres)")
    p.add_argument("--reflections", type=int, default=2, help="0=direct, 1=+floor, 2=+wall")
    p.add_argument("--no-fading", action="store_true", help="Disable Rayleigh fading tail")
    p.add_argument("--symbols", type=int, default=64, help="Number of FSK symbols to transmit")
    args = p.parse_args(argv)

    ch = ChannelConfig(
        range_m=args.range,
        snr_db=args.snr,
        reflections=args.reflections,
        fading_taps=0 if args.no_fading else 8,
    )
    result = simulate(symbols=args.symbols, ch=ch)

    print("=== SignalHop Spatial Channel Simulation ===")
    for k, v in result.items():
        print(f"  {k:24s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
