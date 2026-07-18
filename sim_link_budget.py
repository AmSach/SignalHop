#!/usr/bin/env python3
"""
SignalHop — Acoustic Link Budget Calculator
Realistic SNR / path-loss / absorption model for ultrasonic mesh planning.

Combines:
  - Spherical spreading (20*log10(r))
  - Atmospheric absorption (ISO 9613-1 style, T + humidity)
  - Ambient noise floors per environment
  - TX directivity + RX directivity
  - Receiver self-noise floor
  - BFSK BER vs Eb/N0 (coherent FSK approximation)

Outputs:
  - rx_spl_db    : received SPL (dB re 20 uPa)
  - rx_snr_db    : SNR in dB (signal / ambient+self noise in 500 Hz band)
  - ber_estimate : bit error rate at 500 bps FSK
  - is_usable()  : True if BER < 1e-3 (one error per ~1000 bits)

This is the planning tool: "if I put two nodes 12 m apart in a noisy factory
at 35 C, can they actually hear each other?"
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Atmospheric absorption (ISO 9613-1 simplified, accurate to ~0.5 dB/m for
# 10-30 kHz, 0-50 C, 10-100 % RH). Coefficients from Bass & Sutherland,
# J. Acoust. Soc. Am. 2004.
# ---------------------------------------------------------------------------

def _absorp_coeff_db_per_m(freq_hz: float, temp_c: float, rel_humidity: float) -> float:
    """Atmospheric absorption coefficient in dB/m for sound in air.

    Uses the Bass, Sutherland, Zuckerwar atmospheric absorption model
    (Bass & Sutherland 2004, J. Acoust. Soc. Am. 115(3), simplified to the
    1.84e-11 + relaxation-terms form). Accurate to ~0.5 dB/m for
    10-30 kHz, 0-50 C, 5-100 % RH.

    Reference: ISO 9613-1:1993 and Bass et al., J. Acoust. Soc. Am. 97(1),
    680-685 (1995).
    """
    T = temp_c + 273.15
    T_ref = 293.15
    f = freq_hz  # Hz, kept in Hz for the relaxation term denominators

    # Molar concentration of water vapour (dimensionless, ~0.01-0.05).
    # Tetens formula for saturation vapor pressure (valid 0-50 C, ~1% accuracy):
    if temp_c < 0:
        psat_kpa = 0.61078 * math.exp(21.875 * temp_c / (temp_c + 265.5))
    else:
        psat_kpa = 0.61078 * math.exp(17.27 * temp_c / (temp_c + 237.3))
    h = (rel_humidity / 100.0) * psat_kpa / 101.325
    h = max(0.001, min(0.10, h))

    # Relaxation frequencies of oxygen and nitrogen (Bass et al. 1984, 1995),
    # in Hz, valid for the ultrasonic band 50 Hz - 1 MHz.
    # Bass 1995 corrections to the 1984 formula.
    pa = 101325.0
    tref = 293.15  # 20 C
    tro = T / tref
    fro = (pa / 101325.0) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h)) * 1e3  # kHz -> Hz
    frn = (pa / 101325.0) * tro ** (-0.5) * (
        9.0 + 280.0 * h * math.exp(-4.170 * (tro ** (-1.0 / 3.0) - 1.0))
    ) * 1e3

    # Coefficients from Bass 1984 (matched to ISO 9613-1 within 5% for 50 Hz-10 kHz;
    # used here for the ultrasonic 18-22 kHz band where it agrees with the
    # Sutherland-Bass extension to within ~10% for h=0.01-0.05).
    a1 = 1.84e-11  # classical + vibrational loss of air, Pa^-1
    a2 = 0.01275 * math.exp(-2239.1 / T)  # O2 relaxation
    a3 = 0.1068 * math.exp(-3352.0 / T)  # N2 relaxation

    tro_pow = tro ** 2.5
    term_thermal = a1 / pa
    term_o2 = a2 / (fro + (f * f) / fro)
    term_n2 = a3 / (frn + (f * f) / frn)
    alpha_npm = f * f * (term_thermal + tro_pow * (term_o2 + term_n2))
    # Convert Nepers/m to dB/m: 1 Np = 8.686 dB.
    return alpha_npm * 8.686


# ---------------------------------------------------------------------------
# Environment catalog: ambient SPL (dB re 20 uPa, A-weighted rms over 1 s)
# plus multipath/excess loss factor in dB.
# ---------------------------------------------------------------------------

@dataclass
class Environment:
    name: str
    ambient_spl_db: float   # broadband noise floor in 1 Hz band ~ 20 kHz
    excess_loss_db: float   # multipath / scattering penalty (per link)


ENVIRONMENTS: List[Environment] = [
    Environment("anechoic",     10.0, 0.0),
    Environment("quiet_office", 22.0, 2.0),
    Environment("home",         30.0, 4.0),
    Environment("office",       38.0, 6.0),
    Environment("cafe",         48.0, 7.0),
    Environment("industrial",   62.0, 9.0),
    Environment("subway",       72.0, 12.0),
]

ENV_BY_NAME = {e.name: e for e in ENVIRONMENTS}


# ---------------------------------------------------------------------------
# Link budget
# ---------------------------------------------------------------------------

@dataclass
class LinkBudget:
    range_m: float
    temp_c: float = 22.0
    rel_humidity: float = 50.0
    freq_hz: float = 19_000.0
    tx_spl_db: float = 100.0      # ~100 dB re 20 uPa at 1 m (typical small speaker)
    tx_directivity_db: float = 0.0
    rx_directivity_db: float = 0.0
    rx_self_noise_db: float = 5.0  # receiver self-noise (1 Hz band)
    bandwidth_hz: float = 500.0    # signal bandwidth = symbol rate
    env: Environment = field(default_factory=lambda: ENVIRONMENTS[3])

    # populated by compute()
    path_loss_db: float = 0.0
    absorption_db: float = 0.0
    total_loss_db: float = 0.0
    rx_spl_db: float = 0.0
    noise_floor_db: float = 0.0
    rx_snr_db: float = 0.0
    eb_n0_db: float = 0.0
    ber_estimate: float = 1.0

    def compute(self) -> None:
        """Run the link budget end to end."""
        r = max(1e-3, float(self.range_m))
        # Spherical spreading referenced to 1 m.
        self.path_loss_db = 20.0 * math.log10(r)
        # Atmospheric absorption over r metres.
        alpha = _absorp_coeff_db_per_m(self.freq_hz, self.temp_c, self.rel_humidity)
        self.absorption_db = alpha * r
        # Total one-way loss.
        self.total_loss_db = (
            self.path_loss_db
            + self.absorption_db
            + self.excess_loss_db_for_env()
            - self.tx_directivity_db
            - self.rx_directivity_db
        )
        # Received SPL.
        self.rx_spl_db = self.tx_spl_db - self.total_loss_db
        # Noise floor: combine ambient (in signal BW) and self noise.
        # ambient_spl_db is per Hz; widen to bandwidth.
        ambient_in_bw = self.env.ambient_spl_db + 10.0 * math.log10(self.bandwidth_hz)
        # Combine: 10*log10(10^(a/10) + 10^(b/10))
        self.noise_floor_db = 10.0 * math.log10(
            10 ** (ambient_in_bw / 10.0) + 10 ** (self.rx_self_noise_db / 10.0)
        )
        # SNR.
        self.rx_snr_db = self.rx_spl_db - self.noise_floor_db
        # Eb/N0 = SNR + 10*log10(BW / bitrate). For FSK with bitrate == BW, 0 dB.
        # Our default is 500 bps with 500 Hz BW so Eb/N0 ~ SNR.
        bitrate = 500.0
        self.eb_n0_db = self.rx_snr_db + 10.0 * math.log10(self.bandwidth_hz / bitrate)
        # BER for coherent BFSK: Q(sqrt(Eb/N0)).
        # Q function approximation (Press et al. NR3 7.1.26).
        self.ber_estimate = _q(math.sqrt(max(0.0, 10 ** (self.eb_n0_db / 10.0))))

    def excess_loss_db_for_env(self) -> float:
        return self.env.excess_loss_db

    def is_usable(self, ber_threshold: float = 1e-3) -> bool:
        return self.ber_estimate < ber_threshold

    def summary(self) -> str:
        lines = [
            f"  range              = {self.range_m:>7.2f} m",
            f"  env                = {self.env.name}",
            f"  freq / BW          = {self.freq_hz/1000:.1f} kHz / {self.bandwidth_hz:.0f} Hz",
            f"  temperature        = {self.temp_c:.1f} C   RH = {self.rel_humidity:.0f} %",
            f"  path loss          = {self.path_loss_db:>7.2f} dB  (spreading)",
            f"  absorption         = {self.absorption_db:>7.2f} dB  (alpha = "
            f"{_absorp_coeff_db_per_m(self.freq_hz, self.temp_c, self.rel_humidity):.3f} dB/m)",
            f"  multipath / excess = {self.env.excess_loss_db:>7.2f} dB",
            f"  total loss         = {self.total_loss_db:>7.2f} dB",
            f"  rx_spl             = {self.rx_spl_db:>7.2f} dB re 20 uPa",
            f"  noise floor        = {self.noise_floor_db:>7.2f} dB re 20 uPa in BW",
            f"  rx_snr             = {self.rx_snr_db:>7.2f} dB",
            f"  Eb/N0              = {self.eb_n0_db:>7.2f} dB",
            f"  BER est. (coh. FSK)= {self.ber_estimate:.2e}",
            f"  usable (<1e-3)     = {self.is_usable()}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _q(x: float) -> float:
    """Q function (Gaussian tail). Press et al. NR3 7.1.26."""
    if x <= 0.0:
        return 0.5
    return _erfc(x / math.sqrt(2.0)) / 2.0


def _erfc(x: float) -> float:
    """Complementary error function, accurate to ~1.5e-7."""
    # NR3 7.1.26
    t = 1.0 / (1.0 + 0.5 * abs(x))
    ans = (
        t * math.exp(
            -x * x - 1.26551223
            + t * (1.00002368
                   + t * (0.37409196
                          + t * (0.09678418
                                 + t * (-0.18628806
                                        + t * (0.27886807
                                               + t * (-1.13520398
                                                      + t * (1.48851587
                                                             + t * (-0.82215223
                                                                    + t * 0.17087277))))))))
        )
    )
    return ans if x >= 0.0 else 2.0 - ans


def _max_usable_range(env: Environment, temp_c: float, rel_humidity: float,
                      threshold: float = 1e-3, iterations: int = 32) -> float:
    """Binary-search the maximum range where BER < threshold."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    lo, hi = 0.5, 200.0
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        lb = LinkBudget(range_m=mid, temp_c=temp_c, rel_humidity=rel_humidity, env=env)
        lb.compute()
        if lb.is_usable(threshold):
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_single(args: argparse.Namespace) -> None:
    env = ENV_BY_NAME[args.environment]
    lb = LinkBudget(
        range_m=args.range,
        temp_c=args.temp,
        rel_humidity=args.humidity,
        freq_hz=args.freq,
        env=env,
    )
    lb.compute()
    print(f"=== SignalHop Link Budget ({env.name}) ===")
    print(lb.summary())


def _print_sweep(args: argparse.Namespace) -> None:
    env = ENV_BY_NAME[args.environment]
    print(f"=== Sweep in {env.name} (T={args.temp}C, RH={args.humidity}%) ===")
    print(f"{'range_m':>8}  {'rx_spl':>8}  {'rx_snr':>8}  {'BER':>10}  usable")
    for r in [0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120]:
        lb = LinkBudget(range_m=r, temp_c=args.temp, rel_humidity=args.humidity, env=env)
        lb.compute()
        print(f"{r:>8.2f}  {lb.rx_spl_db:>8.2f}  {lb.rx_snr_db:>8.2f}  "
              f"{lb.ber_estimate:>10.3e}  {'YES' if lb.is_usable() else 'no'}")
    mx = _max_usable_range(env, args.temp, args.humidity)
    print(f"\noperational envelope (BER<1e-3): up to {mx:.2f} m")


def _print_compare(args: argparse.Namespace) -> None:
    print(f"=== Environment comparison (T={args.temp}C, RH={args.humidity}%, "
          f"freq={args.freq/1000:.1f} kHz) ===")
    print(f"{'environment':<14}  {'max_range_m':>12}  {'rx_spl@1m':>10}  "
          f"{'ambient_dB':>11}")
    for env in ENVIRONMENTS:
        lb = LinkBudget(range_m=1.0, temp_c=args.temp, rel_humidity=args.humidity,
                        freq_hz=args.freq, env=env)
        lb.compute()
        rx1m = lb.rx_spl_db
        mx = _max_usable_range(env, args.temp, args.humidity)
        print(f"{env.name:<14}  {mx:>12.2f}  {rx1m:>10.2f}  "
              f"{env.ambient_spl_db:>11.1f}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SignalHop acoustic link budget calculator")
    p.add_argument("--range", type=float, default=5.0,
                   help="range in metres (default 5)")
    p.add_argument("--environment", choices=list(ENV_BY_NAME.keys()), default="office")
    p.add_argument("--temp", type=float, default=22.0, help="temperature C")
    p.add_argument("--humidity", type=float, default=50.0, help="relative humidity %%")
    p.add_argument("--freq", type=float, default=19_000.0,
                   help="carrier frequency in Hz (default 19000)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--sweep", action="store_true",
                      help="sweep range values and show operational envelope")
    mode.add_argument("--compare-environments", action="store_true",
                      help="compare all environments' max range")
    args = p.parse_args(argv)

    if args.compare_environments:
        _print_compare(args)
    elif args.sweep:
        _print_sweep(args)
    else:
        _print_single(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
