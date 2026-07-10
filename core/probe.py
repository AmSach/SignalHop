#!/usr/bin/env python3
"""
SignalHop — Acoustic Link Probe
Round-trip time + chirp-correlation distance + SNR estimator.

Measures the live quality of an acoustic link between two modems
without needing time synchronization. Useful for:
  - picking the best relay in a mesh,
  - triggering route re-discovery when SNR drops,
  - placing a "how far away is that phone" estimate on a UI.

All numbers come from a single round-trip exchange of two short
FSK frames. Speed-of-sound in air ≈ 343 m/s at 20°C.

Public API
----------
    LinkProbe(modem, speed_of_sound=343.0)
        .ping_pair(tx_samples, rx_samples, sample_offset=0) -> ProbeResult
        .ping_local(samples) -> ProbeResult            # self-loopback

    ProbeResult(distance_m, rtt_s, snr_db, chirp_corr, peer_id=None)

A 0.5 m resolution is typical indoors; outdoors, wind/temperature
can shift the estimate by ~10%.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

try:
    from .modem import AcousticModem, SAMPLE_RATE  # package import
except ImportError:  # fallback when run as a flat script
    from modem import AcousticModem, SAMPLE_RATE


# ----------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """One acoustic link measurement."""
    distance_m: float       # one-way, in meters
    rtt_s: float            # round-trip time, seconds
    snr_db: float           # estimated SNR of the received signal
    chirp_corr: float       # max chirp-correlation peak (0..1-ish)
    peer_id: Optional[bytes] = None  # set by mesh layer

    def is_usable(self, snr_min_db: float = 6.0, corr_min: float = 0.1) -> bool:
        """A link is "usable" if we got a clean chirp and decent SNR."""
        return self.snr_db >= snr_min_db and self.chirp_corr >= corr_min

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["peer_id"] is not None:
            d["peer_id"] = d["peer_id"].hex()
        return d


# ----------------------------------------------------------------------------
# Probe
# ----------------------------------------------------------------------------

class LinkProbe:
    """Estimate distance, RTT, and SNR for an acoustic round-trip."""

    def __init__(self, modem: Optional[AcousticModem] = None,
                 speed_of_sound: float = 343.0) -> None:
        self.modem = modem or AcousticModem()
        # 343 m/s at 20°C dry air; tweak for hot/cold environments
        self.c = float(speed_of_sound)

    # ---- core measurement ----

    def ping_pair(self,
                  tx_samples: np.ndarray,
                  rx_samples: np.ndarray,
                  sample_offset: int = 0,
                  peer_id: Optional[bytes] = None) -> ProbeResult:
        """Measure a link from a transmitted frame and what came back.

        Args:
            tx_samples:    The frame we sent (used to know its duration).
            rx_samples:    The audio we received (loopback / remote mic).
            sample_offset: Index in rx_samples where our TX started
                           (subtract this to ignore capture buffer lag).
            peer_id:       Optional 8-byte mesh node id, for logging.
        """
        if not isinstance(tx_samples, np.ndarray) or tx_samples.size == 0:
            raise ValueError("tx_samples must be a non-empty numpy array")
        if not isinstance(rx_samples, np.ndarray) or rx_samples.size == 0:
            raise ValueError("rx_samples must be a non-empty numpy array")

        # Chirp correlation peak in the RECEIVED signal. We use the
        # strongest chirp anywhere in rx (caller controls what rx is).
        chirp_ref = self.modem.generate_chirp(up=True)
        if rx_samples.size < chirp_ref.size:
            chirp_corr = 0.0
        else:
            corr = np.correlate(rx_samples, chirp_ref, mode="valid")
            peak = float(np.max(np.abs(corr))) if corr.size else 0.0
            norm = float(np.linalg.norm(chirp_ref) * np.linalg.norm(rx_samples)) + 1e-9
            chirp_corr = peak / norm

        # RTT = samples between the END of our TX and the chirp peak in RX.
        # The TX itself contains a chirp, so we look for the FIRST chirp
        # peak that is at or after tx_samples.size — that is the echo.
        rtt_s = 0.0
        if chirp_corr > 0 and rx_samples.size >= chirp_ref.size:
            corr_full = np.correlate(rx_samples, chirp_ref, mode="valid")
            peaks = np.flatnonzero(np.abs(corr_full) >= 0.5 * np.max(np.abs(corr_full)))
            echo_idx = int(peaks[peaks >= tx_samples.size][0]) if np.any(peaks >= tx_samples.size) else int(peaks[0])
            gap_samples = max(0, echo_idx - tx_samples.size)
            rtt_s = gap_samples / SAMPLE_RATE
        elif rx_samples.size == tx_samples.size:
            # True loopback with no separate echo — distance is zero.
            rtt_s = 0.0
        else:
            # Couldn't find a chirp; fall back to one frame duration.
            rtt_s = tx_samples.size / SAMPLE_RATE

        # One-way distance from RTT (divided by 2 for round-trip)
        distance_m = max(0.0, (rtt_s * self.c) / 2.0)

        # SNR: signal RMS over noise-floor RMS in the quiet region
        # (use first 100ms of rx as noise estimate)
        snr_db = _estimate_snr_db(rx_samples, noise_floor_samples=min(4800, rx_samples.size // 4))

        return ProbeResult(
            distance_m=distance_m,
            rtt_s=rtt_s,
            snr_db=snr_db,
            chirp_corr=chirp_corr,
            peer_id=peer_id,
        )

    def ping_local(self, samples: np.ndarray) -> ProbeResult:
        """Self-loopback: speaker -> mic of the same device (distance ~0)."""
        return self.ping_pair(samples, samples, sample_offset=0)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _estimate_snr_db(signal: np.ndarray, noise_floor_samples: int) -> float:
    """Crude SNR estimate. Treats the first chunk as noise, the rest as signal.

    Returns a value in dB. If the "noise" floor is louder than the signal
    (very weird captures), clamps to a low value so callers see a bad link.
    """
    if signal.size < 2 * noise_floor_samples:
        # Signal too short — assume moderate noise
        return 3.0
    noise = signal[:noise_floor_samples]
    sig = signal[noise_floor_samples:]
    p_noise = float(np.mean(noise ** 2)) + 1e-12
    p_sig = float(np.mean(sig ** 2)) + 1e-12
    return 10.0 * np.log10(p_sig / p_noise)


# ----------------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from .modem import AcousticModem
    except ImportError:
        from modem import AcousticModem

    m = AcousticModem()
    probe = LinkProbe(m)

    msg = b"ping"
    tx = m.tx(msg)
    # Simulate loopback: prepend 0.05s of "silence" so RTT > 0
    silence = np.zeros(int(0.05 * SAMPLE_RATE), dtype=np.float32)
    rx = np.concatenate([silence, tx])
    res = probe.ping_pair(tx, rx)
    print(f"distance : {res.distance_m:.2f} m")
    print(f"rtt      : {res.rtt_s * 1000:.1f} ms")
    print(f"snr      : {res.snr_db:.1f} dB")
    print(f"chirp    : {res.chirp_corr:.3f}")
    print(f"usable   : {res.is_usable()}")
