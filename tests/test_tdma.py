#!/usr/bin/env python3
"""Tests for SignalHop TDMA scheduler."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_tdma import build_schedule, SLOT_DURATION_MS, TDMASchedule


def test_schedule_assigns_one_slot_per_node():
    schedule = build_schedule([1, 2, 3, 4, 5])
    assert len(schedule.slots) == 6  # 1 sync + 5 node slots
    assert schedule.slots[0] == -1
    assert set(nid for nid in schedule.slots.values() if nid != -1) == {1, 2, 3, 4, 5}
    print("  ✓ one slot per node")


def test_frame_duration_scales_with_nodes():
    s_small = build_schedule([1, 2, 3])
    s_large = build_schedule(list(range(1, 11)))
    assert s_large.frame_ms > s_small.frame_ms
    assert s_small.frame_ms == 4 * SLOT_DURATION_MS
    assert s_large.frame_ms == 11 * SLOT_DURATION_MS
    print("  ✓ frame duration scales")


def test_active_window_matches_slot_length():
    schedule = build_schedule([10, 20, 30])
    # node 20 is slot 2 -> starts at 2 * SLOT_DURATION_MS in frame
    t_in_slot = 2 * SLOT_DURATION_MS + 50
    t_outside_slot = 1 * SLOT_DURATION_MS + 50  # slot 1's window, not 2's
    assert schedule.is_active(20, t_in_slot) is True
    assert schedule.is_active(20, t_outside_slot) is False
    print("  ✓ active window detection")


def test_unknown_node_never_active():
    schedule = build_schedule([1, 2, 3])
    assert schedule.is_active(99, 0) is False
    assert schedule.slot_for(99) == -1
    print("  ✓ unknown node gated")


def test_schedule_repeats_each_frame():
    schedule = build_schedule([1, 2])
    # frame_ms = 3 * 250 = 750ms; node 1 slot starts at 250ms in each frame
    t_first = 250
    t_second = 750 + 250
    assert schedule.is_active(1, t_first) is True
    assert schedule.is_active(1, t_second) is True
    print("  ✓ schedule repeats")


def test_sync_slot_is_not_a_node():
    schedule = build_schedule([5, 6, 7])
    assert schedule.is_active(5, 0) is False  # t=0 is sync window
    assert schedule.is_active(5, 250) is True  # t=250 is node 5's window
    print("  ✓ sync slot reserved")


if __name__ == '__main__':
    print("Running SignalHop TDMA tests...")
    test_schedule_assigns_one_slot_per_node()
    test_frame_duration_scales_with_nodes()
    test_active_window_matches_slot_length()
    test_unknown_node_never_active()
    test_schedule_repeats_each_frame()
    test_sync_slot_is_not_a_node()
    print(f"All {6} tests passed.")
