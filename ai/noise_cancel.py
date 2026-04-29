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
    CNN-based denoising.
    
    In production: load a pre-trained TensorFlow Lite model that classifies
    clean vs noisy 18kHz vs 20kHz energy patterns. The model should take
    a spectrogram as input and output a mask of same dimensions.
    
    Architecture for training (not included):
        Input: [batch, freq_bins, time_frames, 1]
        Conv2D(32, 3x3, relu) → BatchNorm → MaxPool
        Conv2D(64, 3x3, relu) → BatchNorm → MaxPool
        Conv2D(64, 3x3, relu) → UpSample
        Conv2D(32, 3x3, relu) → UpSample
        Conv2D(1,  1x1, sigmoid) → output mask
    
    Loss: binary crossentropy between clean spectrogram and noisy * mask
    
    For deployment: convert to TensorFlow Lite with dynamic range quantization.
    
    Args:
        signal: numpy array of audio samples (float32)
        model_path: path to .tflite model file
    
    Returns:
        Denoised audio signal (float32 numpy array)
    """
    if model_path is None:
        # Fallback: apply spectral subtraction as approximate denoising
        # when no model is available
        denoiser = Denoiser(alpha=2.0, fft_size=512, hop_size=128)
        return denoiser.denoise(signal)
    
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_idx = interpreter.get_input_details()[0]['index']
        output_idx = interpreter.get_output_details()[0]['index']
        
        # Compute spectrogram
        frames = np.array([signal[i:i+512] for i in range(0, len(signal)-512, 128)])
        spec = np.abs(np.fft.rfft(frames, n=512))
        spec_input = spec.reshape(1, spec.shape[0], spec.shape[1], 1).astype(np.float32)
        
        interpreter.set_tensor(input_idx, spec_input)
        interpreter.invoke()
        mask = interpreter.get_tensor(output_idx)[0, :, :, 0]
        
        # Apply mask and reconstruct
        clean_frames = frames * mask
        result = np.zeros_like(signal)
        window = np.hanning(512)
        for i, frame in enumerate(clean_frames):
            result[i*128:i*128+512] += window * frame
        return result
    except ImportError:
        # tflite_runtime not available, fall back to spectral subtraction
        denoiser = Denoiser(alpha=2.0, fft_size=512, hop_size=128)
        return denoiser.denoise(signal)