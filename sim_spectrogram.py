#!/usr/bin/env python3
"""
sim_spectrogram.py — ASCII spectrogram of a transmitted acoustic modem frame.

SignalHop is an acoustic OFDM-style modem: it sends symbols on a grid of
subcarrier frequencies over audio. This simulator visualises a single frame
as a plain-ASCII spectrogram so you can "see" the spectral shape of one
transmission from a terminal. It's also handy for explaining to non-RF
people what the modem is actually doing.

The output is a column-per-time-slot, row-per-frequency-bin grid of
greyscale characters. Because we don't depend on numpy or matplotlib,
this file is pure stdlib (`struct`, `math`, `array`).

Run it directly:

    python3 sim_spectrogram.py
    python3 sim_spectrogram.py --symbols 32 --subcarriers 24 --snr 18
    python3 sim_spectrogram.py --help

It can also be imported as a module: see `synth_frame()`, `stft()`, and
`render()` for the building blocks.
"""
from __future__ import annotations

import argparse
import math
import random
import struct
import sys
import wave
from array import array
from typing import Iterable, List, Sequence, Tuple

# Greyscale ramp, dense → sparse. First char = "loudest".
_RAMP = "█▓▒░:;-+*#@%&8X0xWMN"

# A 16-tone default for the OFDM grid. In a real modem these would be spaced
# ~46 Hz apart centered around 1.8 kHz. We just want them to look distinct.
DEFAULT_SUBCARRIERS_HZ: Tuple[int, ...] = (
    800, 1000, 1200, 1400, 1600, 1800, 2000, 2200,
    2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800,
)

# Sample rate of the synthetic audio. Stdlib `wave` is happiest with these.
SAMPLE_RATE = 48_000


def synth_frame(
    symbols: int = 16,
    subcarriers: int = 16,
    snr_db: float = 24.0,
    seed: int = 1337,
) -> array:
    """Synthesize a 2-D grid of (symbol x subcarrier) amplitudes.

    Each cell holds an amplitude in [0, 1] representing the energy on that
    subcarrier during that symbol period. We add an SNR-controlled amount
    of noise so you can see how the picture degrades as the link gets worse.
    """
    rng = random.Random(seed)
    freqs = DEFAULT_SUBCARRIERS_HZ[:subcarriers]
    signal_power = 10 ** (snr_db / 10.0)
    grid: List[List[float]] = []
    for s in range(symbols):
        row: List[float] = []
        for f in freqs:
            base = 0.5 + 0.5 * math.sin(2 * math.pi * (s + 1) / max(symbols, 1))
            # Modulate per-frequency so the grid isn't a flat gradient.
            base *= 0.6 + 0.4 * math.cos(2 * math.pi * f / 1000.0)
            # Add noise calibrated to the requested SNR.
            noise = rng.gauss(0, 1.0 / math.sqrt(signal_power))
            row.append(max(0.0, min(1.0, base + noise)))
        grid.append(row)
    return grid


def stft(
    grid: Sequence[Sequence[float]],
    sample_rate: int = SAMPLE_RATE,
) -> array:
    """Render a (symbol, subcarrier) grid into a flat audio buffer.

    We don't really need a real STFT here — we're going to feed the buffer
    straight into `wave`, so it just needs to be a self-consistent PCM
    stream. Each row is windowed with a Hann envelope and added to the
    output at the right offset. This is a toy synthesizer, not a real OFDM
    modulator, but the spectrogram you get back looks right.
    """
    symbols = len(grid)
    subcarriers = len(grid[0])
    freqs = DEFAULT_SUBCARRIERS_HZ[:subcarriers]
    samples_per_symbol = sample_rate // 8  # 8 symbols/sec → 125 ms each
    total = samples_per_symbol * symbols
    out = array("f", [0.0] * total)
    for s, row in enumerate(grid):
        for f, amp in zip(freqs, row):
            if amp <= 0.01:
                continue
            base = s * samples_per_symbol
            for i in range(samples_per_symbol):
                t = (base + i) / sample_rate
                envelope = 0.5 * (1 - math.cos(2 * math.pi * (i + 1) / samples_per_symbol))
                out[base + i] += 0.3 * amp * envelope * math.sin(2 * math.pi * f * t)
    # Soft clip to prevent harsh wrap-around if the user cranks the SNR.
    for i in range(len(out)):
        out[i] = math.tanh(out[i])
    return out


def render(
    grid: Sequence[Sequence[float]],
    width: int = 80,
    height: int = 20,
) -> str:
    """Render the (symbol, subcarrier) grid as ASCII art.

    Convention: time flows DOWN the screen, frequency runs LEFT→RIGHT. So
    `height` is the number of time rows in the output and `width` is the
    number of frequency columns. We average adjacent bins to fit the
    requested shape. Brightness picks a character from `_RAMP`; bins below
    the noise floor render as a space.
    """
    if not grid or not grid[0]:
        return ""
    # grid[t][f]: t = time symbol index, f = subcarrier (freq) index
    time_len = len(grid)
    freq_len = len(grid[0])
    out_time = height if height > 0 else time_len
    out_freq = width if width > 0 else freq_len
    lines: List[str] = []
    for r in range(out_time):
        # top row = earliest symbol
        t0 = r * time_len // out_time
        t1 = max(t0 + 1, (r + 1) * time_len // out_time)
        t1 = min(t1, time_len)
        line: List[str] = []
        for c in range(out_freq):
            f0 = c * freq_len // out_freq
            f1 = max(f0 + 1, (c + 1) * freq_len // out_freq)
            f1 = min(f1, freq_len)
            acc = sum(grid[t][f] for t in range(t0, t1) for f in range(f0, f1)) / max(
                1, (t1 - t0) * (f1 - f0)
            )
            idx = min(len(_RAMP) - 1, int(acc * len(_RAMP)))
            line.append(_RAMP[idx] if acc > 0.05 else " ")
        lines.append("".join(line))
    return "\n".join(lines)


def to_wav(
    samples: array,
    out_path: str = "sim_spectrogram.wav",
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """Write the synthesized audio out as a 16-bit PCM WAV file."""
    pcm = array("h", (max(-32768, min(32767, int(s * 32767))) for s in samples))
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ASCII spectrogram of a synthetic acoustic-modem frame.",
    )
    parser.add_argument("--symbols", type=int, default=24, help="OFDM symbols per frame (default: 24)")
    parser.add_argument("--subcarriers", type=int, default=16, help="Subcarrier count (default: 16, max 16)")
    parser.add_argument("--snr", type=float, default=24.0, help="Simulated link SNR in dB (default: 24)")
    parser.add_argument("--width", type=int, default=64, help="ASCII art width in columns (default: 64)")
    parser.add_argument("--height", type=int, default=16, help="ASCII art height in rows (default: 16)")
    parser.add_argument("--wav", type=str, default="", help="Optional path to write a 16-bit WAV of the frame")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed for reproducible noise (default: 1337)")
    args = parser.parse_args(argv)

    if args.subcarriers < 1 or args.subcarriers > len(DEFAULT_SUBCARRIERS_HZ):
        parser.error(f"--subcarriers must be 1..{len(DEFAULT_SUBCARRIERS_HZ)}")
    if args.symbols < 1:
        parser.error("--symbols must be >= 1")

    grid = synth_frame(args.symbols, args.subcarriers, args.snr, args.seed)
    print(f"# SignalHop frame: {args.symbols} symbols × {args.subcarriers} subcarriers @ SNR={args.snr:.0f} dB")
    print(render(grid, args.width, args.height))
    if args.wav:
        path = to_wav(stft(grid), args.wav)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
