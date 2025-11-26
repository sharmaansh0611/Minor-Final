"""
evaluation.py
Simple evaluation script that reads a CSV and computes basic metrics:
 - average battery drop per 24h
 - fraction of event rows (wake_reason==event)
 - average sleep_duration
Generates simple plots (requires matplotlib).
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dummy_traces.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    print("Rows:", len(df))
    print("Event fraction:", (df['wake_reason']=='event').mean())
    print("Avg battery pct:", df['battery_pct'].mean())
    print("Avg sleep (s):", df['sleep_duration_s'].mean())
    os.makedirs("experiments/plots", exist_ok=True)
    plt.figure()
    df['battery_pct'].plot(title="Battery % over rows")
    plt.savefig("experiments/plots/battery_pct.png")
    print("Saved plot to experiments/plots/battery_pct.png")

if __name__ == "__main__":
    main()
