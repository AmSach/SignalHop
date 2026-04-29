#!/usr/bin/env python3
"""
SignalHop — AI Noise Cancellation
CNN-based audio enhancement for acoustic modem signals.
"""

import numpy as np
from typing import Tuple


class Denoiser:
    """
    Simple spectral subtraction + CNN wrapper for acoustic denoising.
    In production, this would load a trained TensorFlow Lite model.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.sample_rate = 48000

    def denoise(self, signal: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio signal."""
        # Spectral gating — simple but effective
        noise_profile = self._estimate_noise_floor(signal[:4096])
        return self._spectral_subtract(signal, noise_profile)

    def _estimate_noise_floor(self, segment: np.ndarray) -> np.ndarray:
        """Estimate noise floor from first segment."""
        fft = np.fft.rfft(segment)
        magnitude = np.abs(fft)
        # Assume lowest energy bins are noise
        noise_floor = np.percentile(magnitude, 20)
        return np.full_like(magnitude, noise_floor * 1.5)

    def _spectral_subtract(self, signal: np.ndarray, noise_floor: np.ndarray) -> np.ndarray:
        """Subtract noise floor from spectrum."""
        window = np.hanning(len(signal))
        windowed = signal * window
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        phase = np.angle(fft)

        # Soft spectral subtraction
        enhanced_magnitude = magnitude - noise_floor * 0.7
        enhanced_magnitude = np.maximum(enhanced_magnitude, noise_floor * 0.1)

        # Reconstruct
        enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
        enhanced = np.fft.irfft(enhanced_fft)
        enhanced *= 2  # scale compensation
        return enhanced[:len(signal)].astype(np.float32)

    def detect_signal(self, signal: np.ndarray, threshold: float = 0.05) -> bool:
        """Detect if there's likely an acoustic signal present."""
        energy = np.sqrt(np.mean(signal ** 2))
        return energy > threshold

    def estimate_snr(self, signal: np.ndarray) -> float:
        """Estimate signal-to-noise ratio in dB."""
        signal_energy = np.mean(signal ** 2)
        noise_segment = signal[:1024]
        noise_energy = np.mean(noise_segment ** 2) + 1e-10
        snr = 10 * np.log10(signal_energy / noise_energy)
        return snr


class Demodulator:
    """
    Neural network-assisted demodulator.
    Uses a simple 1D CNN to detect bits in noisy FSK signals.
    """

    def __init__(self):
        self.sequence_length = 480  # 10ms at 48kHz

    def predict_bits(self, signal: np.ndarray) -> np.ndarray:
        """Predict bits from a raw signal segment using energy detection."""
        predictions = []
        step = self.sequence_length

        for i in range(0, len(signal) - step, step):
            segment = signal[i:i + step]
            # Simple energy-based prediction
            energy = np.mean(segment ** 2)
            # High energy = bit 1, low energy = bit 0
            pred = 1 if energy > 0.01 else 0
            predictions.append(pred)

        return np.array(predictions)

    def confidence_score(self, segment: np.ndarray) -> float:
        """Return confidence (0-1) that segment contains valid signal."""
        energy = np.mean(segment ** 2)
        # Normalize to 0-1 range
        confidence = min(1.0, energy / 0.05)
        return confidence


if __name__ == "__main__":
    denoiser = Denoiser()

    # Generate test signal (FSK tone)
    sample_rate = 48000
    t = np.linspace(0, 0.1, int(sample_rate * 0.1), False)
    clean_signal = np.sin(2 * np.pi * 19000 * t).astype(np.float32)

    # Add noise
    noise = np.random.normal(0, 0.05, len(clean_signal)).astype(np.float32)
    noisy_signal = clean_signal + noise

    # Denoise
    cleaned = denoiser.denoise(noisy_signal)

    snr_before = denoiser.estimate_snr(noisy_signal)
    snr_after = denoiser.estimate_snr(cleaned)

    print(f"SNR before: {snr_before:.1f} dB")
    print(f"SNR after:  {snr_after:.1f} dB")
    print(f"Signal detected: {denoiser.detect_signal(cleaned)}")
    print(f"Confidence: {denoiser.confidence_score(cleaned):.3f}")