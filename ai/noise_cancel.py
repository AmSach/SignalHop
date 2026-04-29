#!/usr/bin/env python3
"""
SignalHop — AI Noise Cancellation
Spectral subtraction for acoustic demodulation.
"""

import numpy as np


class Denoiser:
    """Spectral subtraction noise cancellation for acoustic signals."""

    def __init__(self, noise_floor=None, alpha=2.0, fft_size=512, hop_size=128):
        self.alpha = alpha  # Over-subtraction factor
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.noise_floor = noise_floor  # Pre-measured noise profile
        self.noise_estimate = None

    def estimate_noise(self, signal, n_frames=10):
        """Estimate noise floor from first n frames (assumed noise-only)."""
        frames = self._frames(signal[:n_frames * self.hop_size])
        noise_specs = [np.abs(np.fft.rfft(f)) for f in frames]
        self.noise_estimate = np.mean(noise_specs, axis=0)
        return self.noise_estimate

    def denoise(self, signal):
        """Apply spectral subtraction to reduce noise."""
        if self.noise_estimate is None:
            self.estimate_noise(signal[:self.hop_size * 10])

        frames = self._frames(signal)
        denoised_frames = []

        for frame in frames:
            spec = np.fft.rfft(frame)
            mag = np.abs(spec)

            # Subtract noise floor
            clean_mag = np.maximum(mag - self.alpha * self.noise_estimate[:len(mag)], 0)

            # Reconstruct (keep phase)
            clean_spec = clean_mag * np.exp(1j * np.angle(spec))
            denoised_frames.append(np.fft.irfft(clean_spec, n=self.fft_size))

        return self._overlap_add(denoised_frames)

    def _frames(self, signal):
        """Segment signal into overlapping frames."""
        return [signal[i:i+self.fft_size] for i in range(0, len(signal) - self.fft_size, self.hop_size)]

    def _overlap_add(self, frames):
        """Reconstruct signal from overlapping frames."""
        n = len(frames[0])
        result = np.zeros(len(frames) * self.hop_size + n)
        window = np.hanning(n)

        for i, frame in enumerate(frames):
            result[i*self.hop_size:i*self.hop_size+n] += window * frame

        return result


def cnn_denoise(signal, model_path=None):
    """
    CNN-based denoising (placeholder for TensorFlow Lite model).
    In production, load a pre-trained model that classifies
    clean vs noisy 18kHz vs 20kHz energy patterns.
    """
    # Placeholder: return signal unchanged
    # Real implementation:
    # import tflite_runtime.interpreter as tflite
    # interpreter = tflite.Interpreter(model_path=model_path)
    # input_data = signal.reshape(1, -1, 1)
    # return interpreter.predict(input_data)
    return signal