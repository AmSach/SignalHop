#!/usr/bin/env python3
"""
SignalHop — Core Acoustic Modem Engine
Encodes/decodes binary data into sound waves using FSK (Frequency Shift Keying).
Supports chirp sequences for synchronization and a simple mesh protocol.
"""

import numpy as np
import struct
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional


# Physical layer constants
SAMPLE_RATE = 48000          # Hz
CARRIER_FREQ_LOW = 18000     # Hz (ultrasonic, avoids audible range)
CARRIER_FREQ_HIGH = 20000   # Hz
SYMBOL_RATE = 500             # symbols per second
CHIRP_DURATION = 0.05        # seconds
PREAMBLE_DURATION = 0.1      # seconds
GUARD_INTERVAL = 0.005       # seconds between symbols

# Protocol constants
MAX_PAYLOAD = 256             # bytes
MAX_HOPS = 8                 # max mesh relay depth
NETWORK_ID = b"SIGNALHOP_V1"


@dataclass
class ModemConfig:
    sample_rate: int = SAMPLE_RATE
    carrier_low: int = CARRIER_FREQ_LOW
    carrier_high: int = CARRIER_FREQ_HIGH
    symbol_rate: int = SYMBOL_RATE
    chirp_duration: float = CHIRP_DURATION
    preamble_duration: float = PREAMBLE_DURATION
    guard_interval: float = GUARD_INTERVAL
    bits_per_symbol: int = 1  # FSK = 1 bit/symbol, could extend to QAM


class AcousticModem:
    """Encode and decode binary data as acoustic signals."""

    def __init__(self, config: Optional[ModemConfig] = None):
        self.cfg = config or ModemConfig()
        self.n_symbols = int(self.cfg.sample_rate / self.cfg.symbol_rate)
        self.chirp_samples = int(self.cfg.chirp_duration * self.cfg.sample_rate)
        self.preamble_samples = int(self.cfg.preamble_duration * self.cfg.sample_rate)

    # ─── Chirp Generation ───────────────────────────────────────────

    def generate_chirp(self, up: bool = True) -> np.ndarray:
        """Generate an up or down linear frequency sweep for sync."""
        t = np.linspace(0, self.cfg.chirp_duration, self.chirp_samples, False)
        if up:
            freq_sweep = np.linspace(self.cfg.carrier_low - 2000,
                                    self.cfg.carrier_high + 2000, self.chirp_samples)
        else:
            freq_sweep = np.linspace(self.cfg.carrier_high + 2000,
                                    self.cfg.carrier_low - 2000, self.chirp_samples)
        phase = 2 * np.pi * np.cumsum(freq_sweep) / self.cfg.sample_rate
        return np.sin(phase).astype(np.float32)

    def generate_preamble(self) -> np.ndarray:
        """Generate sync preamble: 4 chirps."""
        chirp_up = self.generate_chirp(up=True)
        return np.concatenate([chirp_up] * 4)

    # ─── Symbol Encoding ───────────────────────────────────────────

    def encode_symbol(self, bit: int) -> np.ndarray:
        """Encode a single bit as an FSK tone burst."""
        freq = self.cfg.carrier_high if bit else self.cfg.carrier_low
        n_samples = self.n_symbols
        t = np.arange(n_samples) / self.cfg.sample_rate
        phase = 2 * np.pi * freq * t
        symbol = np.sin(phase).astype(np.float32)
        # Smooth edges with cosine window
        taper = np.cos(np.linspace(0, np.pi, 20))
        symbol[:20] *= taper[:20]
        symbol[-20:] *= taper[::-1]
        return symbol

    def encode_bits(self, bits: List[int]) -> np.ndarray:
        """Convert a list of bits to audio waveform."""
        symbols = [self.encode_symbol(b) for b in bits]
        guard = np.zeros(int(self.cfg.guard_interval * self.cfg.sample_rate), dtype=np.float32)
        return np.concatenate([s for s in symbols for s in [s, guard]])

    # ─── Framing ───────────────────────────────────────────────────

    def build_frame(self, payload: bytes, seq: int = 0, ttl: int = MAX_HOPS,
                    sender_id: bytes = b"\x00" * 8) -> np.ndarray:
        """Build a complete acoustic frame."""
        # Header
        header = bytearray()
        header += NETWORK_ID
        header += struct.pack("!BHH8s", len(payload), seq, ttl, sender_id)
        header += bytes(16)  # reserved
        header_bytes = bytes(header)

        # CRC
        crc = struct.pack("!I", self._crc32(header_bytes + payload))

        # Encode header (48 bytes = 384 bits)
        header_bits = self._bytes_to_bits(header_bytes)
        header_signal = self.encode_bits(header_bits)

        # Encode payload
        payload_bits = self._bytes_to_bits(payload)
        payload_signal = self.encode_bits(payload_bits)

        # Encode CRC (32 bits)
        crc_bits = self._bytes_to_bits(bytes(crc))
        crc_signal = self.encode_bits(crc_bits)

        # Preamble + frame
        preamble = self.generate_preamble()
        return np.concatenate([preamble, header_signal, payload_signal, crc_signal])

    # ─── Demodulation ──────────────────────────────────────────────

    def detect_chirp(self, signal: np.ndarray) -> bool:
        """Simple energy-based chirp detector."""
        if len(signal) < self.chirp_samples:
            return False
        # Cross-correlate with up-chirp
        chirp = self.generate_chirp(up=True)
        corr = np.correlate(signal, chirp, mode='valid')
        threshold = np.max(corr) * 0.6
        return np.any(corr > threshold)

    def demod_bits(self, signal: np.ndarray) -> List[int]:
        """Demodulate FSK bits from a signal using Goertzel + threshold."""
        bits = []
        n = self.n_symbols
        guard = int(self.cfg.guard_interval * self.cfg.sample_rate)
        step = n + guard

        for i in range(0, len(signal) - n, step):
            segment = signal[i:i + n]
            if len(segment) < n:
                break

            # Goertzel algorithm for specific frequencies
            energy_low = self._goertzel(segment, self.cfg.carrier_low)
            energy_high = self._goertzel(segment, self.cfg.carrier_high)

            if energy_low > energy_high:
                bits.append(0)
            else:
                bits.append(1)
        return bits

    def _goertzel(self, samples: np.ndarray, freq: float) -> float:
        """Goertzel algorithm for single-frequency energy detection."""
        k = int(0.5 + (len(samples) * freq / self.cfg.sample_rate))
        w = 2 * np.pi * k / len(samples)
        coeff = 2 * np.cos(w)

        s = 0.0
        s1 = 0.0
        s2 = 0.0
        for x in samples:
            s = x + coeff * s1 - s2
            s2 = s1
            s1 = s

        return s1 * s1 + s2 * s2 - coeff * s1 * s2

    # ─── Utilities ─────────────────────────────────────────────────

    def _bytes_to_bits(self, data: bytes) -> List[int]:
        return [int(b) for byte in data for b in format(byte, '08b')]

    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        bits = bits[:len(bits) - (len(bits) % 8)]
        return bytes(int(''.join(map(str, bits[i:i+8])), 2) for i in range(0, len(bits), 8))

    @staticmethod
    def _crc32(data: bytes) -> int:
        return struct.pack(">I", 0xffffffff & hashlib.crc32(data))

    def bytes_to_signal(self, data: bytes, seq: int = 0) -> np.ndarray:
        """High-level: bytes → acoustic waveform."""
        return self.build_frame(data, seq)

    def signal_to_bytes(self, signal: np.ndarray) -> Optional[bytes]:
        """High-level: acoustic waveform → bytes (or None if failed)."""
        if not self.detect_chirp(signal):
            return None
        bits = self.demod_bits(signal)
        if len(bits) < 384:
            return None

        # Decode header (384 bits = 48 bytes)
        header_bytes = self._bits_to_bytes(bits[:384])

        # Verify network ID
        if header_bytes[:12] != NETWORK_ID:
            return None

        payload_len, seq, ttl, sender_id = struct.unpack("!BHH8s", header_bytes[12:24])
        if payload_len > MAX_PAYLOAD:
            return None

        # Decode payload
        payload_start = 384
        payload_end = payload_start + payload_len * 8
        if len(bits) < payload_end + 32:
            return None

        payload = self._bits_to_bytes(bits[payload_start:payload_end])
        received_crc = self._bits_to_bytes(bits[payload_end:payload_end + 32])

        expected_crc = struct.pack("!I", self._crc32(header_bytes + payload))
        if received_crc != expected_crc:
            return None

        return payload


def generate_beacon(location: str = "UNKNOWN", battery_pct: int = 100) -> bytes:
    """Generate an emergency beacon payload."""
    import time
    msg = f"SOS|{location}|{battery_pct}%|{int(time.time())}"
    return msg.encode()[:64].ljust(64, b'\x00')


if __name__ == "__main__":
    modem = AcousticModem()

    # Test encode/decode cycle
    test_msg = b"Hello from SignalHop!"
    signal = modem.bytes_to_signal(test_msg)
    decoded = modem.signal_to_bytes(signal)

    print(f"Original:     {test_msg}")
    print(f"Signal shape: {signal.shape} ({len(signal)/SAMPLE_RATE:.2f}s)")
    print(f"Decoded:      {decoded}")
    print(f"Match:        {test_msg == decoded}")