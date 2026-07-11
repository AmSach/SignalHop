#!/usr/bin/env python3
"""
SignalHop — Interleaver demo
Shows the difference between FEC-with-interleaving and FEC-without
when the acoustic channel delivers a burst of errors (e.g. a 30 ms
fading dip that wipes 4 consecutive bits).

Run:  python3 sim_interleaver.py
"""

import random
from core.interleaver import interleave, deinterleave


def hamming_decode(bits_7):
    """Standard (7,4) Hamming decode. Returns 4 data bits or None if uncorrectable."""
    if len(bits_7) != 7:
        return None
    p1 = bits_7[0] ^ bits_7[2] ^ bits_7[4] ^ bits_7[6]
    p2 = bits_7[1] ^ bits_7[2] ^ bits_7[5] ^ bits_7[6]
    p3 = bits_7[3] ^ bits_7[4] ^ bits_7[5] ^ bits_7[6]
    syndrome = p1 + 2 * p2 + 4 * p3
    corrected = list(bits_7)
    if syndrome != 0:
        corrected[syndrome - 1] ^= 1
    # If still error → uncorrectable
    cp1 = corrected[0] ^ corrected[2] ^ corrected[4] ^ corrected[6]
    cp2 = corrected[1] ^ corrected[2] ^ corrected[5] ^ corrected[6]
    cp3 = corrected[3] ^ corrected[4] ^ corrected[5] ^ corrected[6]
    if cp1 or cp2 or cp3:
        return None
    return [corrected[2], corrected[4], corrected[5], corrected[6]]


def hamming_encode(data_4):
    return [
        data_4[0] ^ data_4[1] ^ data_4[3],
        data_4[0] ^ data_4[2] ^ data_4[3],
        data_4[0],
        data_4[1] ^ data_4[2] ^ data_4[3],
        data_4[1],
        data_4[2],
        data_4[3],
    ]


def main():
    random.seed(7)
    rows, cols = 8, 16
    print("SignalHop — Interleaver demo (Hamming 7,4 + 8x16 block interleaver)")
    print("=" * 70)
    print(f"Block geometry: {rows} rows x {cols} cols = {rows*cols} bits/block")
    print(f"Burst tolerance: up to {cols} consecutive bit-errors are correctable")
    print()

    # Build a payload of 64 random data bits -> 16 codewords
    data = [random.randint(0, 1) for _ in range(64)]
    codewords = []
    for i in range(0, 64, 4):
        codewords.append(hamming_encode(data[i : i + 4]))

    flat = [b for cw in codewords for b in cw]
    interleaved = interleave(flat, rows, cols)

    # ---- Run 1: NO interleaving, inject a burst of 4 errors ----
    no_int = list(flat)
    burst_pos = 30
    for i in range(burst_pos, burst_pos + 4):
        no_int[i] ^= 1
    decoded_no_int = []
    for i in range(0, len(no_int), 7):
        d = hamming_decode(no_int[i : i + 7])
        if d is None:
            decoded_no_int.append(None)
        else:
            decoded_no_int.extend(d)
    errors_no_int = sum(
        1 for a, b in zip(data, decoded_no_int) if a != b
    )

    # ---- Run 2: WITH interleaving, same burst on the channel ----
    burst = list(interleaved)
    for i in range(burst_pos, burst_pos + 4):
        burst[i] ^= 1
    deinterleaved = deinterleave(burst, rows, cols)
    decoded_int = []
    for i in range(0, len(deinterleaved), 7):
        d = hamming_decode(deinterleaved[i : i + 7])
        if d is None:
            decoded_int.append(None)
        else:
            decoded_int.extend(d)
    errors_int = sum(
        1 for a, b in zip(data, decoded_int) if a != b
    )

    print("Burst: 4 consecutive channel-bit errors injected")
    print()
    print(f"  Without interleaver: {errors_no_int} / 64 bit errors after FEC")
    print(f"  With    interleaver: {errors_int} / 64 bit errors after FEC")
    print()
    if errors_int == 0 and errors_no_int > 0:
        print("Interleaver saved the message. Burst of 4 would have killed")
        print("two full Hamming codewords, but the interleaver spread those 4")
        print("errors to 1 bit per codeword, which FEC can correct.")
    elif errors_int == 0 and errors_no_int == 0:
        print("Both decoded cleanly — the burst didn't exceed Hamming's 1-bit")
        print("limit. Try a longer burst.")


if __name__ == "__main__":
    main()
