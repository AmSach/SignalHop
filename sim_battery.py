#!/usr/bin/env python3
"""
SignalHop — Battery lifetime estimator
Models node battery drain over time given duty-cycle + tx power + sleep current.

Adds another tool to the SignalHop acoustic-mesh planning suite.
"""
from __future__ import annotations
import argparse
import math
import sys
from dataclasses import dataclass


# Common coin-cell / Li-Po battery capacities (mAh) for reference
BATTERY_PRESETS = {
    "cr2032":  {"capacity_mah": 220.0, "nominal_v": 3.0},
    "cr2477":  {"capacity_mah": 1000.0, "nominal_v": 3.0},
    "lipo_500": {"capacity_mah": 500.0, "nominal_v": 3.7},
    "lipo_1500": {"capacity_mah": 1500.0, "nominal_v": 3.7},
    "aa_alkaline": {"capacity_mah": 2500.0, "nominal_v": 1.5},
}


@dataclass
class DutyCycle:
    """Power state duty cycle — fraction of time in each state."""
    sleep_pct: float      # 0-1
    rx_pct: float         # 0-1
    tx_pct: float         # 0-1
    cpu_active_pct: float # 0-1

    def __post_init__(self) -> None:
        total = self.sleep_pct + self.rx_pct + self.tx_pct + self.cpu_active_pct
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"duty cycle must sum to 1.0, got {total:.3f}")


@dataclass
class NodePowerModel:
    """Power draw per state, in mA at battery voltage."""
    sleep_ma: float = 0.020       # deep sleep, ESP32 + sensor
    rx_ma: float = 40.0           # listening
    tx_ma: float = 120.0          # full-power ultrasonic burst
    cpu_active_ma: float = 30.0   # processing / crypto / protocol work

    def avg_current_ma(self, dc: DutyCycle) -> float:
        return (
            dc.sleep_pct * self.sleep_ma
            + dc.rx_pct * self.rx_ma
            + dc.tx_pct * self.tx_ma
            + dc.cpu_active_pct * self.cpu_active_ma
        )


def battery_lifetime_hours(capacity_mah: float, avg_ma: float) -> float:
    if avg_ma <= 0:
        return float("inf")
    return capacity_mah / avg_ma


def lifetime_summary(
    capacity_mah: float,
    avg_ma: float,
    duty: DutyCycle,
    packets_per_hour: float = 0.0,
) -> dict:
    hours = battery_lifetime_hours(capacity_mah, avg_ma)
    days = hours / 24.0
    return {
        "avg_current_ma": avg_ma,
        "lifetime_hours": hours,
        "lifetime_days": days,
        "lifetime_months": days / 30.44,
        "lifetime_years": days / 365.25,
        "packets_per_hour": packets_per_hour,
        "total_packets_in_lifetime": packets_per_hour * hours,
    }


def typical_chat_duty(beacon_interval_s: float, chat_bursts_per_hour: int) -> DutyCycle:
    """Reasonable duty cycle for an always-listening chat node with bursts."""
    # Beacons: 1 short TX every beacon_interval
    beacon_tx_frac = 0.005 / beacon_interval_s
    chat_tx_frac = chat_bursts_per_hour * 0.05 / 3600.0
    rx_frac = 0.10
    cpu_frac = 0.02
    sleep_frac = max(0.0, 1.0 - beacon_tx_frac - chat_tx_frac - rx_frac - cpu_frac)
    return DutyCycle(sleep_frac, rx_frac, beacon_tx_frac + chat_tx_frac, cpu_frac)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery", choices=list(BATTERY_PRESETS), default="cr2032")
    parser.add_argument("--sleep-ma", type=float, default=0.020)
    parser.add_argument("--rx-ma", type=float, default=40.0)
    parser.add_argument("--tx-ma", type=float, default=120.0)
    parser.add_argument("--cpu-ma", type=float, default=30.0)
    parser.add_argument("--beacon-interval", type=float, default=5.0,
                        help="seconds between beacons")
    parser.add_argument("--chat-bursts-per-hour", type=int, default=12)
    args = parser.parse_args()

    preset = BATTERY_PRESETS[args.battery]
    duty = typical_chat_duty(args.beacon_interval, args.chat_bursts_per_hour)
    power = NodePowerModel(args.sleep_ma, args.rx_ma, args.tx_ma, args.cpu_ma)
    avg_ma = power.avg_current_ma(duty)
    summary = lifetime_summary(
        preset["capacity_mah"], avg_ma, duty,
        packets_per_hour=args.chat_bursts_per_hour * 8.0,
    )

    print(f"Battery: {args.battery}  ({preset['capacity_mah']:.0f} mAh)")
    print(f"Power model: sleep={power.sleep_ma} mA, rx={power.rx_ma} mA, "
          f"tx={power.tx_ma} mA, cpu={power.cpu_active_ma} mA")
    print(f"Duty: sleep={duty.sleep_pct*100:.1f}%, "
          f"rx={duty.rx_pct*100:.1f}%, "
          f"tx={duty.tx_pct*100:.3f}%, "
          f"cpu={duty.cpu_active_pct*100:.2f}%")
    print(f"Avg current: {avg_ma:.3f} mA")
    print(f"Lifetime:    {summary['lifetime_days']:.1f} days  "
          f"({summary['lifetime_months']:.2f} months, "
          f"{summary['lifetime_years']:.3f} years)")
    print(f"~{summary['total_packets_in_lifetime']:.0f} chat packets in lifetime")
    return 0


if __name__ == "__main__":
    sys.exit(main())
