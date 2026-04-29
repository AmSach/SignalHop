#!/usr/bin/env python3
"""
SignalHop — Core Acoustic Modem Engine v3
FSK modem with working encode/decode cycle.
"""

import numpy as np
import struct
import zlib
from dataclasses import dataclass
from typing import List, Optional


SAMPLE_RATE = 48000
FREQ_LOW = 18000
FREQ_HIGH = 20000
SYMBOL_RATE = 500
NETWORK_ID = b"SIGNALHOP_V1"
MAX_PAYLOAD = 256


@dataclass
class ModemConfig:
    sample_rate: int = SAMPLE_RATE
    carrier_low: int = FREQ_LOW
    carrier_high: int = FREQ_HIGH
    symbol_rate: int = SYMBOL_RATE


class AcousticModem:
    def __init__(self, cfg: Optional[ModemConfig] = None):
        self.cfg = cfg or ModemConfig()
        self.samples_per_symbol = int(self.cfg.sample_rate / self.cfg.symbol_rate)
        self.preamble_samples = int(0.05 * self.cfg.sample_rate) * 4  # 100ms

    def generate_chirp(self, up: bool = True) -> np.ndarray:
        t = np.linspace(0, 0.05, int(0.05 * self.cfg.sample_rate), False)
        if up:
            f_sweep = np.linspace(self.cfg.carrier_low - 2000, self.cfg.carrier_high + 2000, len(t))
        else:
            f_sweep = np.linspace(self.cfg.carrier_high + 2000, self.cfg.carrier_low - 2000, len(t))
        phase = 2 * np.pi * np.cumsum(f_sweep) / self.cfg.sample_rate
        return np.sin(phase).astype(np.float32)

    def generate_preamble(self) -> np.ndarray:
        return np.concatenate([self.generate_chirp(up=True)] * 4)

    def encode_symbol(self, bit: int) -> np.ndarray:
        freq = self.cfg.carrier_high if bit else self.cfg.carrier_low
        n = self.samples_per_symbol
        t = np.arange(n) / self.cfg.sample_rate
        sym = np.sin(2 * np.pi * freq * t).astype(np.float32)
        taper = np.cos(np.linspace(0, np.pi, 20))
        sym[:20] *= taper[:20]
        sym[-20:] *= taper[::-1]
        return sym

    def encode_bits(self, bits: List[int]) -> np.ndarray:
        return np.concatenate([self.encode_symbol(b) for b in bits])

    def goertzel(self, samples: np.ndarray, freq: float) -> float:
        k = int(0.5 + len(samples) * freq / self.cfg.sample_rate)
        w = 2 * np.pi * k / len(samples)
        coeff = 2 * np.cos(w)
        s = s1 = s2 = 0.0
        for x in samples:
            s = x + coeff * s1 - s2
            s2, s1 = s1, s
        return s1 * s1 + s2 * s2 - coeff * s1 * s2

    def detect_chirp(self, signal: np.ndarray) -> bool:
        if len(signal) < self.samples_per_symbol:
            return False
        chirp = self.generate_chirp(up=True)
        corr = np.correlate(signal, chirp, mode='valid')
        return bool(np.max(corr) > 0)

    def demod_bits(self, signal: np.ndarray) -> List[int]:
        bits = []
        n = self.samples_per_symbol
        for i in range(0, len(signal), n):
            seg = signal[i:i + n]
            if len(seg) < n:
                break
            el = self.goertzel(seg, self.cfg.carrier_low)
            eh = self.goertzel(seg, self.cfg.carrier_high)
            bits.append(0 if el > eh else 1)
        return bits

    def build_frame(self, payload: bytes) -> np.ndarray:
        header = bytearray()
        header += NETWORK_ID  # 12 bytes
        header += struct.pack('!BHH8s', len(payload), 0, 8, b'\x00' * 8)  # 13 bytes
        header += bytes(16)  # 16 bytes reserved → total 41 bytes header

        crc = struct.pack('!I', 0xffffffff & zlib.crc32(header + payload))

        header_bits = [int(b) for byte in bytes(header) for b in format(byte, '08b')]
        payload_bits = [int(b) for byte in payload for b in format(byte, '08b')]
        crc_bits = [int(b) for byte in bytes(crc) for b in format(byte, '08b')]

        preamble = self.generate_preamble()
        return np.concatenate([
            preamble,
            self.encode_bits(header_bits),
            self.encode_bits(payload_bits),
            self.encode_bits(crc_bits)
        ])

    def parse_frame(self, signal: np.ndarray) -> Optional[bytes]:
        if not self.detect_chirp(signal):
            return None

        data = signal[self.preamble_samples:]
        bits = self.demod_bits(data)

        if len(bits) < 41 * 8:
            return None

        bit_str = ''.join(str(b) for b in bits[:41 * 8])
        header_bytes = bytes(int(bit_str[i:i+8], 2) for i in range(0, 41 * 8, 8))

        if header_bytes[:12] != NETWORK_ID:
            return None

        payload_len = header_bytes[12]
        if payload_len > MAX_PAYLOAD:
            return None

        total_bits = 41 * 8 + payload_len * 8 + 32
        if len(bits) < total_bits:
            return None

        payload_bits = bits[41 * 8:41 * 8 + payload_len * 8]
        payload = bytes(int(''.join(str(b) for b in payload_bits[i:i+8]), 2) for i in range(0, len(payload_bits), 8))

        crc_received = bits[41 * 8 + payload_len * 8:41 * 8 + payload_len * 8 + 32]
        expected_crc = struct.pack('!I', 0xffffffff & zlib.crc32(header_bytes + payload))
        received_crc = bytes(int(''.join(str(b) for b in crc_received[i:i+8]), 2) for i in range(0, 32, 8))

        return payload if expected_crc == received_crc else None

    def tx(self, data: bytes) -> np.ndarray:
        return self.build_frame(data)

    def rx(self, signal: np.ndarray) -> Optional[bytes]:
        return self.parse_frame(signal)


if __name__ == '__main__':
    modem = AcousticModem()
    msg = b"Hello from SignalHop!"
    signal = modem.tx(msg)
    decoded = modem.rx(signal)
    print(f"Original: {msg}")
    print(f"Signal:   {len(signal)} samples ({len(signal)/SAMPLE_RATE:.2f}s)")
    print(f"Decoded:  {decoded}")
    print(f"Match:    {msg == decoded}")
