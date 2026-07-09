"""Smoke tests for sim_fec — ensure the four schemes run end-to-end."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim_fec import _simulate, FECResult, _hamming_encode, _hamming_decode
import random


def test_no_fec_clean_channel():
    r = _simulate("none", b"hello world" * 2, 0.0, random.Random(0))
    assert r.delivered == r.bits
    assert r.unrecoverable == 0


def test_hamming_single_bit_error_corrected():
    bits = [0, 1, 0, 1, 1, 0, 0, 1] * 4
    enc = _hamming_encode(bits)
    # Flip one bit
    enc[3] ^= 1
    dec = _hamming_decode(enc)
    assert dec == bits, f"hamming failed: {dec} != {bits}"


def test_hamming_double_bit_error_misses():
    bits = [1, 0, 1, 0] * 8
    enc = _hamming_encode(bits)
    enc[0] ^= 1
    enc[1] ^= 1  # two-bit error
    dec = _hamming_decode(enc)
    # Hamming(7,4) cannot correct 2-bit errors; result will differ in at least one position
    assert dec != bits


def test_repetition_majority_wins():
    r = _simulate("repetition3", bytes(range(16)), 0.1, random.Random(1))
    assert r.scheme == "repetition3"
    assert r.bits == 16 * 8 * 3


def test_xor_parity_runs():
    r = _simulate("xor_parity", bytes(range(32)), 0.05, random.Random(0))
    assert r.scheme == "xor_parity"
    assert r.bits > 0


def test_all_schemes_finish():
    for s in ("none", "repetition3", "xor_parity", "hamming"):
        r = _simulate(s, b"abcdefgh", 0.05, random.Random(0))
        assert isinstance(r, FECResult)


def test_unknown_scheme_raises():
    try:
        _simulate("nope", b"x", 0.1, random.Random(0))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
