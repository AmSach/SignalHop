"""
sim_absorption.py - Underwater acoustic absorption (Thorp's model) for SignalHop.

Models frequency-dependent transmission loss in the ocean. This is the "extra
dB/km" on top of geometric spreading that makes underwater acoustic comms
such a challenge: 1 kHz travels much farther than 100 kHz.

Two variants are implemented:
  - "thorp"      : Thorp (1967) form, expressed in dB/km, valid < 50 kHz
  - "francois"   : Francois-Garrison (1982) simplified, valid 0.2-1000 kHz

References:
  - Thorp, W.H. (1967). "Analytic Description of the Low-Frequency
    Attenuation Coefficient". J. Acoust. Soc. Am. 42(1).
  - Stojanovic, M. (1999). "Acoustic (Underwater) Communications".
    Wiley Encyclopedia of Electrical and Electronics Engineering.

Pure stdlib. No numpy, no scipy.
"""
from __future__ import annotations
import math


# ---------------------------------------------------------------------------
# Absorption coefficient formulae
# ---------------------------------------------------------------------------
def _thorp_alpha_dbkm(f_khz: float) -> float:
    """
    Thorp (1967) form expressed in dB/km. Valid f < 50 kHz.

    Standard published formula:
        alpha = 0.11 f^2 / (1 + f^2)
              + 44 f^2 / (4100 + f^2)
              + 2.75e-4 f^2
              + 0.003
    where f is in kHz and the result is in dB/km.
    """
    f = max(f_khz, 0.001)
    a = (0.11 * f ** 2) / (1.0 + f ** 2)
    b = (44.0 * f ** 2) / (4100.0 + f ** 2)
    c = 2.75e-4 * f ** 2
    d = 0.003
    return a + b + c + d


def _francois_garrison_alpha_dbkm(f_khz: float) -> float:
    """
    Francois-Garrison (1982) full seawater form, valid 0.2-1000 kHz.
    Returns absorption in dB/km.

    Standard form (Lurton & Ainslie):
        alpha = A1 * P1 * f1 * f^2 / (f1^2 + f^2)
              + A2 * P2 * f2 * f^2 / (f2^2 + f^2)
              + A3 * P3 * f^2
    Defaults (T=10 C, S=35 ppt, D=50 m, pH=8):
        A1=0.001, f1=0.78 kHz, P1=0.7
        A2=0.001, f2=42 kHz,   P2=0.7
        A3=0.001, P3=0.04
    """
    f = max(f_khz, 0.001)
    A1, f1, P1 = 0.001, 0.78, 0.7
    A2, f2, P2 = 0.001, 42.0, 0.7
    A3, P3 = 0.001, 0.04
    p1 = A1 * P1 * f1 * f ** 2 / (f1 ** 2 + f ** 2)
    p2 = A2 * P2 * f2 * f ** 2 / (f2 ** 2 + f ** 2)
    p3 = A3 * P3 * f ** 2
    return p1 + p2 + p3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def absorption_db_per_km(frequency_hz: float, model: str = "auto") -> float:
    """
    Absorption coefficient alpha(f) in dB/km for the given frequency.
    `model`:
        "auto"      -> Thorp for f < 1 kHz, Francois-Garrison otherwise
        "thorp"     -> always Thorp
        "francois"  -> always Francois-Garrison
    """
    if frequency_hz <= 0:
        raise ValueError("frequency must be > 0")
    f_khz = frequency_hz / 1000.0
    if model == "thorp" or (model == "auto" and f_khz <= 50.0):
        return _thorp_alpha_dbkm(f_khz)
    if model == "francois" or model == "auto":
        return _francois_garrison_alpha_dbkm(f_khz)
    raise ValueError(f"Unknown model: {model!r}")


def total_transmission_loss_db(
    distance_m: float,
    frequency_hz: float,
    spreading: float = 1.5,
    model: str = "auto",
) -> float:
    """
    Full TL(f, d) = spreading * 10*log10(d_km) + alpha(f) * d_km
        spreading=1.5 -> practical spreading (mixed cylindrical/spherical)
        spreading=1.0 -> cylindrical
        spreading=2.0 -> spherical
    """
    if distance_m <= 0:
        raise ValueError("distance must be > 0")
    if frequency_hz <= 0:
        raise ValueError("frequency must be > 0")
    d_km = distance_m / 1000.0
    if d_km < 1e-6:
        d_km = 1e-6
    geom = spreading * 10.0 * math.log10(d_km)
    absorb = absorption_db_per_km(frequency_hz, model=model) * d_km
    return geom + absorb


def max_range_for_loss(
    target_loss_db: float,
    frequency_hz: float,
    spreading: float = 1.5,
    model: str = "auto",
    tol: float = 1.0,
    max_iter: int = 200,
) -> float:
    """
    Solve TL(d) = target_loss_db for d (meters) via bisection.
    Useful for "given a 60 dB budget, how far can I talk on 25 kHz?"
    """
    if target_loss_db <= 0:
        raise ValueError("target_loss_db must be > 0")
    lo, hi = 1.0, 200_000.0
    # Guarantee that the bracket covers the answer: TL grows with distance.
    tl_hi = total_transmission_loss_db(hi, frequency_hz, spreading=spreading, model=model)
    if tl_hi < target_loss_db:
        # The chosen `hi` is still too close. Try a larger bracket.
        hi = 1_000_000.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        tl = total_transmission_loss_db(mid, frequency_hz, spreading=spreading, model=model)
        if abs(tl - target_loss_db) < tol:
            return mid
        if tl < target_loss_db:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def band_summary(
    f_low_hz: float = 100.0,
    f_high_hz: float = 50_000.0,
    distance_m: float = 1000.0,
    n_points: int = 64,
) -> list[tuple[float, float, float]]:
    """
    Return [(f_hz, alpha_db_km, total_loss_db), ...] sampled log-uniformly
    between f_low and f_high at the given range.
    """
    if f_low_hz <= 0 or f_high_hz <= f_low_hz:
        raise ValueError("invalid frequency band")
    log_lo = math.log10(f_low_hz)
    log_hi = math.log10(f_high_hz)
    out: list[tuple[float, float, float]] = []
    for i in range(n_points):
        frac = i / max(n_points - 1, 1)
        f = 10.0 ** (log_lo + frac * (log_hi - log_lo))
        a = absorption_db_per_km(f)
        tl = total_transmission_loss_db(distance_m, f)
        out.append((f, a, tl))
    return out


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _fmt_band(rows: list[tuple[float, float, float]]) -> str:
    lines = ["  f(Hz)        alpha(dB/km)   TL @ 1 km (dB)"]
    lines.append("  " + "-" * 42)
    for f, a, tl in rows:
        lines.append(f"  {f:9.1f}    {a:10.4f}    {tl:10.2f}")
    return "\n".join(lines)


def main() -> None:
    print("SignalHop Underwater Acoustic Absorption (Thorp / Francois-Garrison)")
    print("=" * 64)
    print()
    print("Sanity check at 25 kHz: alpha should be ~5-7 dB/km.")
    a25 = absorption_db_per_km(25_000.0)
    print(f"  alpha(25 kHz)        = {a25:.3f} dB/km")
    a10 = absorption_db_per_km(10_000.0)
    print(f"  alpha(10 kHz)        = {a10:.3f} dB/km")
    a01 = absorption_db_per_km(1_000.0)
    print(f"  alpha(1 kHz)         = {a01:.3f} dB/km")
    print()
    print("Max range with 80 dB link budget (25 kHz, 1.5 spreading):")
    r = max_range_for_loss(80.0, 25_000.0)
    print(f"  -> {r:.0f} m ({r/1000.0:.2f} km)")
    print()
    print("Band summary (100 Hz - 50 kHz @ 1 km, 1.5 spreading):")
    print(_fmt_band(band_summary()))


if __name__ == "__main__":
    main()
