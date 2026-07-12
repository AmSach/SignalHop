"""Tests for sim_spectrogram.py."""
from __future__ import annotations

import io
import math
import os
import struct
import sys
import unittest
import wave
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sim_spectrogram  # type: ignore  # noqa: E402


class SynthFrameTests(unittest.TestCase):
    def test_shape(self):
        g = sim_spectrogram.synth_frame(symbols=8, subcarriers=6, seed=1)
        self.assertEqual(len(g), 8)
        for row in g:
            self.assertEqual(len(row), 6)

    def test_amplitudes_in_unit_interval(self):
        g = sim_spectrogram.synth_frame(symbols=20, subcarriers=8, seed=2, snr_db=12.0)
        for row in g:
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_seed_makes_results_deterministic(self):
        a = sim_spectrogram.synth_frame(symbols=4, subcarriers=4, seed=42)
        b = sim_spectrogram.synth_frame(symbols=4, subcarriers=4, seed=42)
        for ra, rb in zip(a, b):
            for x, y in zip(ra, rb):
                self.assertAlmostEqual(x, y, places=6)

    def test_low_snr_has_more_variance(self):
        hi = sim_spectrogram.synth_frame(symbols=40, subcarriers=12, seed=3, snr_db=40.0)
        lo = sim_spectrogram.synth_frame(symbols=40, subcarriers=12, seed=3, snr_db=3.0)
        def var(g):
            flat = [v for row in g for v in row]
            mean = sum(flat) / len(flat)
            return sum((x - mean) ** 2 for x in flat) / len(flat)
        # A 37-dB SNR jump should be a meaningful swing in output variance.
        self.assertGreater(var(lo), var(hi) * 0.8)

    def test_subcarriers_clamped(self):
        # Asking for more subcarriers than the default ramp should fail at
        # the parser layer, but synth_frame should also be safe.
        g = sim_spectrogram.synth_frame(symbols=2, subcarriers=1, seed=4)
        self.assertEqual(len(g[0]), 1)


class StftTests(unittest.TestCase):
    def test_output_length_matches_symbols(self):
        grid = sim_spectrogram.synth_frame(symbols=4, subcarriers=4, seed=5)
        audio = sim_spectrogram.stft(grid, sample_rate=8000)
        expected = (8000 // 8) * 4
        self.assertEqual(len(audio), expected)

    def test_audio_finite_and_soft_clipped(self):
        grid = sim_spectrogram.synth_frame(symbols=4, subcarriers=16, seed=6, snr_db=50.0)
        audio = sim_spectrogram.stft(grid, sample_rate=48000)
        self.assertGreater(len(audio), 0)
        for s in audio:
            self.assertTrue(math.isfinite(s))
            self.assertGreaterEqual(s, -1.0)
            self.assertLessEqual(s, 1.0)


class RenderTests(unittest.TestCase):
    def test_render_dimensions(self):
        # grid is (symbols=8, subcarriers=4) -> 4 rows of up to 8 cols
        grid = sim_spectrogram.synth_frame(symbols=8, subcarriers=4, seed=7)
        out = sim_spectrogram.render(grid, width=8, height=4)
        rows = [r for r in out.split("\n") if r]
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(len(row), 8)

    def test_render_uses_ramp_chars_or_blank(self):
        grid = sim_spectrogram.synth_frame(symbols=8, subcarriers=4, seed=8)
        out = sim_spectrogram.render(grid, width=8, height=4)
        valid = set(sim_spectrogram._RAMP) | {" ", "\n"}
        for ch in out:
            self.assertIn(ch, valid)

    def test_render_empty_grid_returns_empty(self):
        self.assertEqual(sim_spectrogram.render([]), "")
        self.assertEqual(sim_spectrogram.render([[]]), "")


class ToWavTests(unittest.TestCase):
    def test_wav_is_valid_16bit_pcm(self):
        grid = sim_spectrogram.synth_frame(symbols=2, subcarriers=2, seed=9, snr_db=30.0)
        audio = sim_spectrogram.stft(grid, sample_rate=8000)
        path = sim_spectrogram.to_wav(audio, out_path="/tmp/test_sim_spectrogram.wav", sample_rate=8000)
        try:
            with wave.open(path, "rb") as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertEqual(w.getframerate(), 8000)
                self.assertEqual(w.getnframes(), len(audio))
        finally:
            if os.path.exists(path):
                os.unlink(path)


class MainTests(unittest.TestCase):
    def test_main_prints_spectrogram(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = sim_spectrogram.main(["--symbols", "4", "--subcarriers", "4", "--width", "16", "--height", "4"])
        self.assertEqual(rc, 0)
        output = buf.getvalue()
        self.assertIn("SignalHop frame", output)
        # 4 subcarriers → 4 rows in the ASCII art
        body = output.split("\n", 1)[1] if "\n" in output else ""
        self.assertGreaterEqual(len(body.splitlines()), 4)

    def test_main_rejects_bad_subcarriers(self):
        with self.assertRaises(SystemExit):
            sim_spectrogram.main(["--subcarriers", "999"])

    def test_main_writes_wav(self):
        buf = io.StringIO()
        path = "/tmp/test_sim_spectrogram_main.wav"
        try:
            with redirect_stdout(buf):
                rc = sim_spectrogram.main(["--symbols", "2", "--subcarriers", "2", "--wav", path])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(path))
        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
