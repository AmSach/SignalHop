"""
Tests for sim_chat.py — verify the interactive + scripted chat paths work.

Run with: python3 -m pytest tests/test_chat.py -v
or:        python3 tests/test_chat.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, ".."))

import sim_chat  # noqa: E402


def test_scripted_clean_round_trip():
    """Scripted messages should all encode/decode cleanly with no failures."""
    rc = sim_chat.scripted("alice", "bob", ["hi", "ping", "ack: ping", "bye"])
    assert rc == 0


def test_scripted_unicode_round_trip():
    """Unicode should survive encode -> decode through the acoustic modem."""
    rc = sim_chat.scripted("alice", "bob", ["héllo", "naïve", "résumé"])
    assert rc == 0


def test_scripted_long_message():
    """A 200-byte message should fit in the 256-byte payload window."""
    long_line = "x" * 200
    rc = sim_chat.scripted("alice", "bob", [long_line])
    assert rc == 0


def test_scripted_oversize_message_reports_failure():
    """A 300-byte message should be rejected (exceeds 256-byte payload)."""
    rc = sim_chat.scripted("alice", "bob", ["a" * 300])
    # scripted() reports failure and returns 1
    assert rc == 1


def test_round_trip_helper_direct():
    """Direct call to the round-trip helper returns matching text."""
    import sim_chat as chat
    from core.modem import AcousticModem

    modem = AcousticModem()
    text, ok, _ms = chat._encode_decode_round_trip(modem, "SignalHop test 123")
    assert ok is True
    assert text == "SignalHop test 123"


if __name__ == "__main__":
    test_scripted_clean_round_trip()
    test_scripted_unicode_round_trip()
    test_scripted_long_message()
    test_scripted_oversize_message_reports_failure()
    test_round_trip_helper_direct()
    print("All sim_chat tests passed.")
