#!/usr/bin/env python3
"""
simulator_generate_traces.py

Generates dummy IoT node traces (CSV). Configurable:
 - nodes, hours, event-rate, seed, packet success, sleep choices
Outputs CSV ready for training/evaluation.
"""
import csv
import argparse
import math
import random
from datetime import datetime, timedelta
from .energy_model import (
    full_joules, energy_active_seconds, energy_sleep_seconds, energy_for_tx,
    battery_pct_from_joules, voltage_from_pct
)
from .event_generator import poisson_event, bursty_multiplier

SECONDS_PER_STEP = 10  # base step resolution (seconds)

def simulate_node_trace(node_id, start_time, duration_hours,
                        event_rate_per_min, baseline_sleep_choices,
                        packet_success_prob, seed=None):
    rng = random.Random(seed)
    rows = []
    steps = int((duration_hours * 3600) / SECONDS_PER_STEP)
    remaining_joules = full_joules()
    current_time = start_time
    lambda_per_step = event_rate_per_min * (SECONDS_PER_STEP / 60.0)

    for step in range(steps):
        tod_seconds = (step * SECONDS_PER_STEP) % 86400
        mult = bursty_multiplier(tod_seconds)
        effective_lambda = lambda_per_step * mult

        event_occurs = poisson_event(effective_lambda, rng)

        base_temp = 20.0 + 5.0 * math.sin(2 * math.pi * (step * SECONDS_PER_STEP) / 86400.0)
        sensor0 = round(base_temp + rng.gauss(0, 0.5), 2)
        sensor1 = 1.0 if event_occurs and rng.random() < 0.95 else 0.0

        if event_occurs and rng.random() < 0.98:
            wake_reason = "event"
            active_duration = rng.uniform(1.0, 4.0)
            sleep_duration = rng.choice(baseline_sleep_choices)
        else:
            wake_reason = "scheduled"
            active_duration = rng.uniform(0.4, 1.6)
            sleep_duration = rng.choice(baseline_sleep_choices)

        tx_attempts = 0
        tx_success = 0
        if wake_reason == "event":
            tx_attempts = rng.randint(1, 3)
            for _ in range(tx_attempts):
                if rng.random() < packet_success_prob:
                    tx_success += 1

        # energy
        active_j = energy_active_seconds(active_duration)
        sleep_j = energy_sleep_seconds(sleep_duration)
        tx_j = energy_for_tx(tx_attempts)
        total_j = active_j + sleep_j + tx_j
        remaining_joules = max(0.0, remaining_joules - total_j)
        batt_pct = round(battery_pct_from_joules(remaining_joules), 2)
        batt_volt = round(voltage_from_pct(batt_pct), 3)
        rssi = int(rng.gauss(-80, 6))

        rows.append({
            "timestamp": current_time.isoformat(),
            "node_id": node_id,
            "battery_voltage": batt_volt,
            "battery_pct": batt_pct,
            "wake_reason": wake_reason,
            "sensor0": sensor0,
            "sensor1": sensor1,
            "tx_attempts": tx_attempts,
            "tx_success": tx_success,
            "sleep_duration_s": round(sleep_duration, 1),
            "active_duration_s": round(active_duration, 2),
            "rssi": rssi
        })

        # advance time
        current_time += timedelta(seconds=SECONDS_PER_STEP + sleep_duration)
        if remaining_joules <= 0:
            break

    return rows

def write_csv(path, rows):
    fieldnames = ["timestamp","node_id","battery_voltage","battery_pct","wake_reason",
                  "sensor0","sensor1","tx_attempts","tx_success","sleep_duration_s","active_duration_s","rssi"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/dummy_traces.csv")
    p.add_argument("--nodes", type=int, default=3)
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--event-rate", type=float, default=1.0,
                   help="avg events per minute per node")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--packet-success", type=float, default=0.9)
    p.add_argument("--sleep-choices", nargs="+", type=float, default=[30,60,120,300])
    return p.parse_args()

def main():
    args = parse_args()
    rng = random.Random(args.seed)
    all_rows = []
    start_time = datetime.utcnow()
    for n in range(args.nodes):
        node_id = f"node{n+1:02d}"
        node_rate = max(0.01, args.event_rate * rng.uniform(0.7, 1.3))
        node_seed = None if args.seed is None else args.seed + n
        rows = simulate_node_trace(node_id, start_time, args.hours,
                                   node_rate, args.sleep_choices,
                                   args.packet_success, seed=node_seed)
        all_rows.extend(rows)
    all_rows.sort(key=lambda r: r["timestamp"])
    write_csv(args.out, all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
