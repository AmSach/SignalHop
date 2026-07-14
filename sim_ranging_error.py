#!/usr/bin/env python3
"""
SignalHop - Acoustic Ranging Error Model
========================================

The acoustic link budget (sim_link_budget.py) tells you whether a frame can
arrive at all. This sim answers the follow-up question every mesh deployer
asks next: "If the frame arrives, how accurately can I estimate the
distance between the two nodes?"

For ultrasonic ranging we almost always use time-of-flight (ToF) or
time-difference-of-arrival (TDoA) on a chirp preamble. The ideal answer is
    d = c * (t_rx - t_tx) / 2
but in practice several error sources corrupt the estimate:

  1. Clock skew between the two nodes' cheap RC oscillators
     (e.g. ESP32 timing drift ~ 40 ppm)
  2. Detection jitter - the cross-correlation peak is not a delta function,
     it has a width related to the chirp bandwidth and SNR
  3. Multipath bias - a strong reflection off a wall or floor arrives a
     few samples later and biases the peak, often POSITIVELY (the reflection
     looks taller than the direct path in NLOS)
  4. NLOS blockage - if the direct path is blocked, only the reflection
     arrives, and the range estimate is always too long
  5. Air-temperature error in the speed-of-sound constant (343 m/s is
     only right at 20 C; +1 m/s per ~+5.8 C)

This sim models all five and reports the resulting range error distribution.
Useful inputs: "given a 6 m room, a 25 C environment, ESP32-class hardware,
how bad will my distances be?" Outputs: bias, std-dev, RMSE, fraction of
estimates usable for trilateration.

Run:  python sim_ranging_error.py
Test: python tests/test_ranging_error.py
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

SPEED_OF_SOUND_M_PER_S = 343.0  # at 20 C, dry air
SPEED_OF_SOUND_DT_DC = 0.606     # m/s per degree Celsius (linear approx)


def speed_of_sound(temp_c: float) -> float:
    """Speed of sound in m/s given air temperature in Celsius.

    Linear approximation valid for -20 to +40 C:
        c(T) = 331.3 + 0.606 * T
    Accurate to ~0.5 m/s vs. the full Cramer/Bass formula.
    """
    return 331.3 + SPEED_OF_SOUND_DT_DC * temp_c


# ---------------------------------------------------------------------------
# Error model components
# ---------------------------------------------------------------------------


def clock_skew_ppm_to_phase_error_m(
    clock_ppm: float,
    range_m: float,
    carrier_hz: float,
) -> float:
    """Convert a relative clock skew (ppm) into an apparent range bias (m).

    A 1 ppm skew means the RX clock runs 1 us slow per second. The
    time-of-flight measurement scales by the same factor, so the
    apparent range drifts by:
        d_err = c * skew_ppm * 1e-6 * (2 * d / c)
              = skew_ppm * 1e-6 * 2 * d

    For a 10 m link and 40 ppm clock, that's 0.8 mm - usually negligible.
    For a 100 m link and 200 ppm it becomes 4 cm.
    """
    return clock_ppm * 1e-6 * 2.0 * range_m


def detection_jitter_std_m(
    snr_db: float,
    chirp_bandwidth_hz: float,
    sample_rate_hz: float = 50_000.0,
) -> float:
    """Standard deviation of detection-jitter range error (m).

    The cross-correlation peak of a chirp of bandwidth B has a width of
    ~1/B seconds. With AWGN noise, the peak-time estimator variance is
        var(t_hat) = 1 / (8 * pi^2 * B^2 * SNR_lin)
    (Ristic, "Statistical Signal Decomposition", ch. 5). We multiply by
    c to convert to meters and take the square root.

    Returns 0 (clamped) below 0 dB SNR - the peak is meaningless.
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    if snr_lin <= 1.0:
        # Below 0 dB the estimator variance diverges - clamp to "huge"
        return float("inf")
    var_t = 1.0 / (8.0 * math.pi ** 2 * chirp_bandwidth_hz ** 2 * snr_lin)
    sigma_t = math.sqrt(var_t)
    c = SPEED_OF_SOUND_M_PER_S
    return c * sigma_t


def multipath_bias_m(
    range_m: float,
    reflection_extra_path_m: float,
    reflection_strength_db: float,
) -> float:
    """Bias added to the range estimate by a single specular reflection.

    `reflection_strength_db` is the reflection's amplitude in dB RELATIVE
    to the direct path. So:
      * +30 dB   -> reflection is 30 dB WEAKER than direct (LOS dominates)
      *   0 dB   -> reflection and direct are equal
      *  -6 dB   -> reflection is LOUDER than direct (NLOS dominates)

    A reflection that arrives `reflection_extra_path_m` LONGER than the
    direct path pulls the cross-correlation peak toward itself in
    proportion to its amplitude vs. direct:

        weight = 1 / (1 + 10^(reflection_strength_db / 10))

    This is a smooth sigmoid bounded by 0 (LOS dominates) and 1
    (reflection dominates).
    """
    weight = 1.0 / (1.0 + 10.0 ** (reflection_strength_db / 10.0))
    return reflection_extra_path_m * weight


# ---------------------------------------------------------------------------
# NLOS detection: ratio of peak to second-peak amplitude
# ---------------------------------------------------------------------------


def peak_to_secondary_ratio(secondary_db_below_peak: float) -> float:
    """Ratio of the cross-correlation peak to the second-highest side peak.

    A clean LOS signal has a dominant single peak and a much smaller
    secondary peak from the matched-filter sidelobes. NLOS signals
    show a SECOND physical peak close in amplitude, because the
    reflection and direct paths are roughly the same height.
    """
    return 10.0 ** (secondary_db_below_peak / 10.0)


def is_likely_nlos(secondary_db_below_peak: float, threshold_db: float = 6.0) -> bool:
    """Heuristic: if the second peak is within `threshold_db` of the main peak,
    we are likely in NLOS and the range estimate should be flagged."""
    return secondary_db_below_peak < threshold_db


# ---------------------------------------------------------------------------
# Per-measurement error simulator
# ---------------------------------------------------------------------------


@dataclass
class RangingConfig:
    """Knobs for a ranging campaign."""
    true_range_m: float = 6.0
    temp_c: float = 22.0
    clock_ppm: float = 40.0             # typical cheap ESP32 RC drift
    snr_db: float = 15.0               # detection SNR at the receiver
    chirp_bandwidth_hz: float = 2_000.0
    sample_rate_hz: float = 50_000.0
    # multipath
    nlos: bool = False
    reflection_extra_path_m: float = 1.5
    reflection_strength_db: float = 6.0  # reflection is 6 dB weaker than direct
    secondary_peak_db_below: float = 12.0  # matched-filter second peak
    # temperature error: the node thinks it's 20 C when it isn't
    assumed_temp_c: float = 20.0
    # rng
    seed: int = 42


@dataclass
class RangingResult:
    """One ranging attempt."""
    true_range_m: float
    measured_range_m: float
    error_m: float
    clock_bias_m: float
    temp_bias_m: float
    detection_jitter_m: float
    multipath_bias_m: float
    flagged_nlos: bool


def simulate_one_measurement(cfg: RangingConfig, rng: random.Random) -> RangingResult:
    """Simulate one acoustic ranging measurement and return the broken-down error."""
    c_real = speed_of_sound(cfg.temp_c)
    c_assumed = speed_of_sound(cfg.assumed_temp_c)

    # 1. clock skew (deterministic, same direction every time)
    clock_bias = clock_skew_ppm_to_phase_error_m(cfg.clock_ppm, cfg.true_range_m, 0.0)

    # 2. temperature error (deterministic)
    # The node uses c_assumed; the real flight is at c_real. So
    # measured = real_tof * c_assumed = (2*d / c_real) * c_assumed
    temp_bias = cfg.true_range_m * (c_assumed / c_real - 1.0) * 2.0

    # 3. detection jitter (random, Gaussian)
    sigma_j = detection_jitter_std_m(cfg.snr_db, cfg.chirp_bandwidth_hz, cfg.sample_rate_hz)
    if math.isinf(sigma_j):
        # Below 0 dB SNR the estimator is meaningless - draw a huge error
        jitter = rng.gauss(0.0, cfg.true_range_m)
    else:
        jitter = rng.gauss(0.0, sigma_j)

    # 4. multipath bias (deterministic for a given geometry)
    if cfg.nlos:
        # Pure NLOS: only the reflection arrives, so measured = direct + extra
        mp_bias = cfg.reflection_extra_path_m
    else:
        mp_bias = multipath_bias_m(
            cfg.true_range_m,
            cfg.reflection_extra_path_m,
            cfg.reflection_strength_db,
        )

    # Combine: measured = true + clock + temp + jitter + multipath
    measured = (
        cfg.true_range_m
        + clock_bias
        + temp_bias
        + jitter
        + mp_bias
    )
    error = measured - cfg.true_range_m

    flagged = is_likely_nlos(cfg.secondary_peak_db_below)

    return RangingResult(
        true_range_m=cfg.true_range_m,
        measured_range_m=measured,
        error_m=error,
        clock_bias_m=clock_bias,
        temp_bias_m=temp_bias,
        detection_jitter_m=jitter,
        multipath_bias_m=mp_bias,
        flagged_nlos=flagged,
    )


def simulate_campaign(
    cfg: RangingConfig,
    n_measurements: int = 1000,
) -> Tuple[List[RangingResult], dict]:
    """Run `n_measurements` ranging attempts and return per-attempt results
    plus a summary statistics dict."""
    rng = random.Random(cfg.seed)
    results: List[RangingResult] = []
    for _ in range(n_measurements):
        results.append(simulate_one_measurement(cfg, rng))

    errors = [r.error_m for r in results]
    flagged = sum(1 for r in results if r.flagged_nlos)

    summary = {
        "n": n_measurements,
        "true_range_m": cfg.true_range_m,
        "mean_error_m": statistics.fmean(errors),
        "median_error_m": statistics.median(errors),
        "std_error_m": statistics.pstdev(errors) if len(errors) > 1 else 0.0,
        "rmse_m": math.sqrt(statistics.fmean([e * e for e in errors])),
        "min_error_m": min(errors),
        "max_error_m": max(errors),
        "nlos_fraction": flagged / n_measurements,
    }
    return results, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary(summary: dict) -> None:
    print(f"  N                = {summary['n']}")
    print(f"  True range       = {summary['true_range_m']:.2f} m")
    print(f"  Mean error       = {summary['mean_error_m']:+.4f} m")
    print(f"  Median error     = {summary['median_error_m']:+.4f} m")
    print(f"  Std-dev error    = {summary['std_error_m']:.4f} m")
    print(f"  RMSE             = {summary['rmse_m']:.4f} m")
    print(f"  Min / max error  = {summary['min_error_m']:+.4f} / {summary['max_error_m']:+.4f} m")
    print(f"  NLOS-flagged     = {summary['nlos_fraction'] * 100:.1f}%")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Acoustic ranging error model: clock skew, SNR jitter, multipath, temperature.",
    )
    p.add_argument("--range", type=float, default=6.0, help="true range in meters (default: 6)")
    p.add_argument("--temp-c", type=float, default=22.0, help="real air temperature, C (default: 22)")
    p.add_argument("--assumed-temp-c", type=float, default=20.0, help="temperature the node THINKS it is, C (default: 20)")
    p.add_argument("--clock-ppm", type=float, default=40.0, help="relative clock skew ppm (default: 40)")
    p.add_argument("--snr-db", type=float, default=15.0, help="detection SNR dB (default: 15)")
    p.add_argument("--chirp-bw", type=float, default=2000.0, help="chirp bandwidth Hz (default: 2000)")
    p.add_argument("--nlos", action="store_true", help="simulate non-line-of-sight (only reflection arrives)")
    p.add_argument("--reflection-extra-m", type=float, default=1.5, help="reflection path length excess, m (default: 1.5)")
    p.add_argument("--reflection-strength-db", type=float, default=6.0, help="reflection weaker than direct by N dB (default: 6)")
    p.add_argument("--secondary-peak-db", type=float, default=12.0, dest="secondary_peak_db", help="matched-filter 2nd peak below main, dB (default: 12)")
    p.add_argument("--n", type=int, default=1000, help="number of measurements (default: 1000)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = p.parse_args(argv)

    cfg = RangingConfig(
        true_range_m=args.range,
        temp_c=args.temp_c,
        clock_ppm=args.clock_ppm,
        snr_db=args.snr_db,
        chirp_bandwidth_hz=args.chirp_bw,
        nlos=args.nlos,
        reflection_extra_path_m=args.reflection_extra_m,
        reflection_strength_db=args.reflection_strength_db,
        secondary_peak_db_below=args.secondary_peak_db,
        assumed_temp_c=args.assumed_temp_c,
        seed=args.seed,
    )

    print(f"SignalHop - Acoustic Ranging Error Model")
    print(f"  c(real)         = {speed_of_sound(cfg.temp_c):.2f} m/s (T={cfg.temp_c} C)")
    print(f"  c(assumed)      = {speed_of_sound(cfg.assumed_temp_c):.2f} m/s (T={cfg.assumed_temp_c} C)")
    print(f"  clock skew      = {cfg.clock_ppm} ppm  => {clock_skew_ppm_to_phase_error_m(cfg.clock_ppm, cfg.true_range_m, 0.0) * 1000:+.2f} mm bias")
    print(f"  detection sigma = {detection_jitter_std_m(cfg.snr_db, cfg.chirp_bandwidth_hz, cfg.sample_rate_hz) * 1000:.2f} mm (SNR={cfg.snr_db} dB, BW={cfg.chirp_bandwidth_hz} Hz)")
    print(f"  multipath       = {'NLOS (reflection only)' if cfg.nlos else f'extra-path {cfg.reflection_extra_path_m} m, reflection {cfg.reflection_strength_db} dB weaker'}")
    print(f"")
    print(f"  Campaign of {args.n} measurements:")
    _, summary = simulate_campaign(cfg, n_measurements=args.n)
    _print_summary(summary)

    # also show NLOS-flagged campaign
    cfg_flag = RangingConfig(
        true_range_m=cfg.true_range_m,
        temp_c=cfg.temp_c,
        clock_ppm=cfg.clock_ppm,
        snr_db=cfg.snr_db,
        chirp_bandwidth_hz=cfg.chirp_bandwidth_hz,
        nlos=cfg.nlos,
        reflection_extra_path_m=cfg.reflection_extra_path_m,
        reflection_strength_db=cfg.reflection_strength_db,
        secondary_peak_db_below=2.0,  # strong second peak
        assumed_temp_c=cfg.assumed_temp_c,
        seed=cfg.seed,
    )
    print("")
    print(f"  Same campaign, but with secondary peak 2 dB below main (NLOS-like):")
    _, summary2 = simulate_campaign(cfg_flag, n_measurements=args.n)
    _print_summary(summary2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
