"""
SignalHop — Burst-Error Interleaver
Spreads adjacent bits across time so a single acoustic fade
erases a few bits from many codewords (correctable by FEC)
rather than wiping out one whole codeword.

A (rows, cols) block interleaver writes the input bit-stream
row-by-row into an N x D matrix and reads it out column-by-column.
The deinterleaver does the reverse. A burst of up to `cols`
consecutive bit-errors on the channel becomes a 1-bit error
in every one of `rows` codewords at the receiver.
"""

from __future__ import annotations

from typing import List, Tuple


def _validate_params(rows: int, cols: int) -> None:
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be >= 1, got rows={rows}, cols={cols}")


def interleave(bits: List[int], rows: int, cols: int) -> List[int]:
    """Block-interleave a bit list using an (rows x cols) matrix.

    Pads with zeros if the input length is not a multiple of rows*cols.
    Returns a list of length ceil(len(bits) / (rows*cols)) * rows * cols.
    """
    _validate_params(rows, cols)
    if not bits:
        return []

    block_size = rows * cols
    n_blocks = (len(bits) + block_size - 1) // block_size
    padded = list(bits) + [0] * (n_blocks * block_size - len(bits))

    out: List[int] = []
    for block_idx in range(n_blocks):
        start = block_idx * block_size
        block = padded[start : start + block_size]
        for c in range(cols):
            for r in range(rows):
                out.append(block[r * cols + c])
    return out


def deinterleave(bits: List[int], rows: int, cols: int) -> List[int]:
    """Reverse of `interleave`."""
    _validate_params(rows, cols)
    if not bits:
        return []

    block_size = rows * cols
    if len(bits) % block_size != 0:
        raise ValueError(
            f"deinterleave input length {len(bits)} not a multiple of {block_size}"
        )

    n_blocks = len(bits) // block_size
    out: List[int] = []
    for block_idx in range(n_blocks):
        start = block_idx * block_size
        block = bits[start : start + block_size]
        matrix = [[0] * cols for _ in range(rows)]
        idx = 0
        for c in range(cols):
            for r in range(rows):
                matrix[r][c] = block[idx]
                idx += 1
        for r in range(rows):
            out.extend(matrix[r])
    return out


def interleave_bytes(data: bytes, rows: int, cols: int) -> bytes:
    """Convenience wrapper for byte strings."""
    bits: List[int] = []
    for byte in data:
        bits.extend(int(b) for b in format(byte, "08b"))
    out_bits = interleave(bits, rows, cols)
    out_bytes = bytearray()
    for i in range(0, len(out_bits), 8):
        out_bytes.append(int("".join(str(b) for b in out_bits[i : i + 8]), 2))
    return bytes(out_bytes)


def deinterleave_bytes(data: bytes, rows: int, cols: int) -> bytes:
    bits: List[int] = []
    for byte in data:
        bits.extend(int(b) for b in format(byte, "08b"))
    out_bits = deinterleave(bits, rows, cols)
    out_bytes = bytearray()
    for i in range(0, len(out_bits), 8):
        out_bytes.append(int("".join(str(b) for b in out_bits[i : i + 8]), 2))
    return bytes(out_bytes)


def burst_tolerance(rows: int, cols: int) -> Tuple[int, int]:
    """Return (max_burst_bits, errors_per_codeword) for given block geometry.

    A burst of up to `cols` consecutive channel-bit errors becomes
    exactly 1 error per codeword (over `rows` codewords).
    """
    _validate_params(rows, cols)
    return cols, 1
