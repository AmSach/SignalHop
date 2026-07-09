#!/usr/bin/env python3
"""
sim_chat.py — Interactive two-terminal text chat over the acoustic modem (simulated).

This is a developer-facing smoke test of the SignalHop chat path: it spins up two
nodes on the same machine (left = "Alice", right = "Bob"), exchanges a few typed
lines, and prints a transcript. Each "frame" is encoded by Alice's modem,
decoded by Bob's modem, and vice versa. No audio hardware, no microphone, no
speaker — purely synthetic round-trips so the chat loop is testable in CI.

Run it directly:

    python3 sim_chat.py

Or pass a list of lines to feed Alice (non-interactive):

    python3 sim_chat.py --script "hello,world,ping,pong,bye"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core.modem import AcousticModem  # noqa: E402


def _encode_decode_round_trip(modem: AcousticModem, text: str) -> tuple[str, bool, float]:
    """Encode text -> waveform -> decode. Returns (decoded_text, ok, seconds)."""
    payload = text.encode("utf-8")
    if len(payload) > 256:
        raise ValueError("payload exceeds 256 bytes; chunking not yet implemented")
    t0 = time.perf_counter()
    waveform = modem.tx(payload)
    decoded = modem.rx(waveform)
    elapsed = time.perf_counter() - t0
    return (decoded.decode("utf-8", errors="replace"), decoded == payload, elapsed)


def _print_banner(left: str, right: str) -> None:
    line = "─" * 56
    print(line)
    print(f"  SignalHop sim_chat — {left}  ⇄  {right}")
    print("  Synthetic acoustic channel.  No audio hardware required.")
    print(line)


def _format_msg(sender: str, text: str, ok: bool, elapsed_ms: float) -> str:
    status = "✓" if ok else "✗"
    return f"  {sender:>5}  {status}  {elapsed_ms:6.1f} ms   {text!r}"


def interactive(left_name: str, right_name: str, prompt: str) -> int:
    """Drive an interactive chat between two simulated modems."""
    alice = AcousticModem()
    bob = AcousticModem()
    _print_banner(left_name, right_name)
    print(f"  Type a message and press Enter.  Empty line to quit.")
    print(f"  Each round-trip is encoded by one side and decoded by the other.")
    print()
    while True:
        try:
            user_input = input(prompt)
        except EOFError:
            print()
            return 0
        if not user_input:
            print("  (empty line, exiting)")
            return 0
        try:
            alice_text, alice_ok, alice_ms = _encode_decode_round_trip(alice, user_input)
        except ValueError as exc:
            print(f"  ! {exc}")
            continue
        print(_format_msg(left_name, alice_text, alice_ok, alice_ms))
        if not alice_ok:
            continue
        # Auto-reply for demonstration: bob echoes with "ack: " prefix.
        bob_msg = f"ack: {alice_text}"
        try:
            bob_text, bob_ok, bob_ms = _encode_decode_round_trip(bob, bob_msg)
        except ValueError as exc:
            print(f"  ! {exc}")
            continue
        print(_format_msg(right_name, bob_text, bob_ok, bob_ms))


def scripted(left_name: str, right_name: str, lines: list[str]) -> int:
    """Run a deterministic script of messages through the round-trip."""
    alice = AcousticModem()
    bob = AcousticModem()
    _print_banner(left_name, right_name)
    all_ok = True
    for line in lines:
        try:
            alice_text, alice_ok, alice_ms = _encode_decode_round_trip(alice, line)
        except ValueError as exc:
            print(f"  ! {left_name} payload too large: {exc}")
            all_ok = False
            continue
        print(_format_msg(left_name, alice_text, alice_ok, alice_ms))
        if not alice_ok:
            all_ok = False
            continue
        bob_msg = f"ack: {alice_text}"
        try:
            bob_text, bob_ok, bob_ms = _encode_decode_round_trip(bob, bob_msg)
        except ValueError as exc:
            print(f"  ! {right_name} payload too large: {exc}")
            all_ok = False
            continue
        print(_format_msg(right_name, bob_text, bob_ok, bob_ms))
        if not bob_ok:
            all_ok = False
    print()
    if all_ok:
        print(f"  Result: {len(lines)} round-trips, all clean ✓")
        return 0
    print(f"  Result: at least one round-trip failed ✗")
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--left", default="alice", help="name of the left node (default alice)")
    p.add_argument("--right", default="bob", help="name of the right node (default bob)")
    p.add_argument(
        "--script",
        default=None,
        help="comma-separated lines to feed the left node (non-interactive)",
    )
    args = p.parse_args()
    if args.script is not None:
        lines = [s for s in args.script.split(",") if s]
        if not lines:
            print("error: --script must contain at least one non-empty line", file=sys.stderr)
            return 2
        return scripted(args.left, args.right, lines)
    return interactive(args.left, args.right, prompt=f"  {args.left}> ")


if __name__ == "__main__":
    raise SystemExit(main())
