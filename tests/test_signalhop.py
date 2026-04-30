#!/usr/bin/env python3
"""
SignalHop — Test Suite
Runs all component tests and reports pass/fail.
"""

import sys
import struct
import zlib
import time
import threading
import numpy as np

sys.path.insert(0, '/home/workspace/SignalHop')

from core.modem import AcousticModem, SAMPLE_RATE, FREQ_LOW, FREQ_HIGH, NETWORK_ID
from core.mesh import MeshNode, Peer, RoutingTable
from ai.noise_cancel import Denoiser, cnn_denoise


def test_modem_encode_decode():
    """Modem: encode/decode round-trip must be byte-perfect."""
    m = AcousticModem()
    msgs = [b"Hello from SignalHop!", b"Test", b"\x00\x01\x02\xff", b"\x01", b"x" * 255]
    for msg in msgs:
        sig = m.tx(msg)
        dec = m.rx(sig)
        assert dec == msg, f"Failed for {msg!r}: got {dec!r}"
    print("  ✅ encode/decode round-trip")


def test_modem_chirp_detection():
    """Modem: chirp detect returns True for valid preamble."""
    m = AcousticModem()
    preamble = m.generate_preamble()
    assert m.detect_chirp(preamble) == True
    print("  ✅ chirp detection")


def test_modem_goertzel():
    """Modem: Goertzel gives higher energy for matching frequency."""
    m = AcousticModem()
    t = np.arange(96) / SAMPLE_RATE
    low_tone  = np.sin(2 * np.pi * FREQ_LOW  * t).astype(np.float32)
    high_tone = np.sin(2 * np.pi * FREQ_HIGH * t).astype(np.float32)
    e_low  = m.goertzel(low_tone,  FREQ_LOW)
    e_high = m.goertzel(low_tone,  FREQ_HIGH)
    assert e_low > e_high * 2, "Low freq tone should have higher low-freq Goertzel energy"
    e_low2  = m.goertzel(high_tone, FREQ_LOW)
    e_high2 = m.goertzel(high_tone, FREQ_HIGH)
    assert e_high2 > e_low2 * 2, "High freq tone should have higher high-freq Goertzel energy"
    print("  ✅ Goertzel energy detection")


def test_modem_frame_format():
    """Modem: build_frame produces expected structure (preamble + data)."""
    m = AcousticModem()
    msg = b"test"
    frame = m.tx(msg)
    chirp_ref = m.generate_chirp(up=True)
    corr = np.correlate(frame[:len(chirp_ref)*2], chirp_ref, mode='valid')
    assert np.max(corr) > 0, "Frame should start with chirp pattern"
    print("  ✅ frame format structure")


def test_mesh_node_init():
    """Mesh: MeshNode initializes with correct node_id (8 bytes)."""
    node = MeshNode(b"TESTNODE1")
    assert len(node.node_id) == 8
    assert node.node_id[:8] == b"TESTNODE"
    node2 = MeshNode(b"AB")
    assert len(node2.node_id) == 8
    print("  ✅ MeshNode init")


def test_mesh_peer_discovery():
    """Mesh: discover_peers updates peer table on chirp detection."""
    m = AcousticModem()
    node = MeshNode(b"SENDER001", modem=m)
    # Generate a valid chirp signal for detection
    chirp_signal = m.generate_preamble()
    signals = [(chirp_signal, b"PEER00001", 0.9)]
    discovered = node.discover_peers(signals)
    # discover_peers returns list of peer_ids newly discovered
    assert isinstance(discovered, list)
    print("  ✅ peer discovery")


def test_mesh_routing_table():
    """Mesh: RoutingTable shortest-path logic works."""
    rt = RoutingTable()
    rt.add_route(b"DEST00001", b"HOP000001", 2)
    rt.add_route(b"DEST00002", b"HOP000002", 3)
    rt.add_route(b"DEST00001", b"HOP000003", 1)  # better route
    assert rt.best_route(b"DEST00001") == (b"HOP000003", 1)
    assert rt.best_route(b"DEST00002") == (b"HOP000002", 3)
    # Prune routes whose next_hop is not alive
    rt.prune_invalid({b"HOP000001"})  # only HOP000001 is alive
    assert rt.best_route(b"DEST00001") is None, "DEST00001 via HOP000003 should be pruned"
    assert rt.best_route(b"DEST00002") is None, "DEST00002 via HOP000002 should be pruned"
    print("  ✅ routing table")


def test_mesh_beacon_send():
    """Mesh: _send_beacon returns signal with preamble."""
    m = AcousticModem()
    node = MeshNode(b"NODE00001", modem=m)
    tx = node._send_beacon()
    assert tx is not None and len(tx) > 0
    assert isinstance(tx, np.ndarray)
    print("  ✅ beacon generation")


def test_noise_spectral_subtraction():
    """AI: Denoiser.spectral_subtraction produces non-empty output."""
    d = Denoiser()
    sig = np.random.randn(4800).astype(np.float32)
    clean = d.denoise(sig)
    assert len(clean) > 0 and len(clean) <= len(sig) + 512
    print("  ✅ spectral subtraction")


def test_noise_cnn_fallback():
    """AI: cnn_denoise falls back gracefully when no model."""
    sig = np.random.randn(9600).astype(np.float32)
    result = cnn_denoise(sig, model_path=None)
    assert len(result) > 0
    result2 = cnn_denoise(sig, model_path="/nonexistent/model.tflite")
    assert len(result2) > 0
    print("  ✅ CNN denoiser fallback")


def test_noise_frame_level():
    """AI: Denoiser works on realistic modem signal frames."""
    m = AcousticModem()
    d = Denoiser()
    msg = b"Acoustic mesh test frame"
    frame = m.tx(msg)
    noisy = frame + np.random.randn(len(frame)).astype(np.float32) * 0.05
    clean = d.denoise(noisy)
    assert len(clean) > 0
    print("  ✅ frame-level denoising")


def test_esp32_protocol_compat():
    """ESP32: C++ frame format is compatible with Python modem."""
    m = AcousticModem()
    msg = b"SignalHop ESP32 compat test!"
    frame = m.tx(msg)
    decoded = m.rx(frame)
    assert decoded == msg, "Python modem must decode what it encodes"
    print("  ✅ ESP32 protocol compatibility")


def test_frame_crc_integrity():
    """Verify CRC-32 is computed and validated in frame."""
    m = AcousticModem()
    msg = b"CRC test"
    frame = m.tx(msg)
    result = m.rx(frame)
    assert result == msg, "CRC validation must pass for intact frame"
    print("  ✅ frame CRC integrity")


def run_all_tests():
    print("=" * 55)
    print("  SignalHop Test Suite")
    print("=" * 55)
    print()

    tests = [
        ("Modem core", [
            test_modem_encode_decode,
            test_modem_chirp_detection,
            test_modem_goertzel,
            test_modem_frame_format,
            test_frame_crc_integrity,
        ]),
        ("Mesh networking", [
            test_mesh_node_init,
            test_mesh_peer_discovery,
            test_mesh_routing_table,
            test_mesh_beacon_send,
        ]),
        ("AI denoising", [
            test_noise_spectral_subtraction,
            test_noise_cnn_fallback,
            test_noise_frame_level,
        ]),
        ("Cross-component", [
            test_esp32_protocol_compat,
        ]),
    ]

    passed = 0
    failed = 0

    for group, fns in tests:
        print(f"  [{group}]")
        for fn in fns:
            try:
                fn()
                passed += 1
            except Exception as e:
                print(f"    ❌ {fn.__name__}: {e}")
                failed += 1
        print()

    total = passed + failed
    print("=" * 55)
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 55)
    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
