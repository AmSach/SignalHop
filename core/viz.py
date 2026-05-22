#!/usr/bin/env python3
"""
SignalHop Routing Visualizer
Renders mesh topology and routing paths as an ASCII diagram.
"""

import json
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    id: str
    label: str
    x: float
    y: float
    connections: list[str] = field(default_factory=list)
    is_gateway: bool = False


def render_topology(nodes: list[Node], path: Optional[list[str]] = None) -> str:
    """Render a simple ASCII topology diagram."""
    lines = []
    lines.append("╔══════════════════════════════════════╗")
    lines.append("║       SignalHop Mesh Topology        ║")
    lines.append("╚══════════════════════════════════════╝")
    lines.append("")

    node_map = {n.id: n for n in nodes}
    path_set = set(path) if path else set()
    path_edges = set()
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            path_edges.add((path[i], path[i+1]))
            path_edges.add((path[i+1], path[i]))

    for node in nodes:
        marker = "[G]" if node.is_gateway else "[*]"
        active = "◉" if node.id in path_set else "○"
        lines.append(f"  {active} {marker} {node.label} ({node.id})")

        for peer_id in node.connections:
            if peer_id not in node_map:
                continue
            peer = node_map[peer_id]
            edge_marker = "───" if (node.id, peer_id) in path_edges else "──·"
            lines.append(f"         {edge_marker} {peer.label}")

        lines.append("")

    if path:
        lines.append(f"Route: {' → '.join(path)}")
        lines.append(f"Hops:  {len(path) - 1}")

    return "\n".join(lines)


def render_signal_quality(node_id: str, snr_db: float, rssi_dbm: int) -> str:
    """Render signal quality bar."""
    quality = "▇▇▇▇▇"
    if snr_db < 5:
        quality = "░░░░░"
    elif snr_db < 10:
        quality = "▒░░░░"
    elif snr_db < 15:
        quality = "▓▒░░░"
    elif snr_db < 20:
        quality = "▓▓▒░░"
    else:
        quality = "▓▓▓▓▓"

    return f"  [{node_id[:8]}] SNR:{snr_db:5.1f}dB RSSI:{rssi_dbm:4d}dBm {quality}"


def demo():
    nodes = [
        Node(id="a1", label="Gateway-A", x=0.0, y=0.0, connections=["b2", "c3"], is_gateway=True),
        Node(id="b2", label="Relay-B",   x=1.0, y=1.0, connections=["a1", "c3", "d4"]),
        Node(id="c3", label="Node-C",    x=2.0, y=0.0, connections=["a1", "b2"]),
        Node(id="d4", label="Node-D",    x=2.0, y=2.0, connections=["b2", "e5"]),
        Node(id="e5", label="Node-E",    x=3.0, y=2.0, connections=["d4"]),
    ]

    path = ["e5", "d4", "b2", "a1"]

    print(render_topology(nodes, path))
    print()
    print("Signal Quality:")
    for node, snr, rssi in [("a1", 25.3, -42), ("b2", 18.7, -61), ("c3", 12.1, -73), ("d4", 8.4, -80), ("e5", 22.1, -55)]:
        print(render_signal_quality(node, snr, rssi))


if __name__ == "__main__":
    demo()