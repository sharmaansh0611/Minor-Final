"""
policy_export.py
Load Q-table and export a compact JSON policy (map from state idx -> best action)
This exported policy can be integrated into firmware as a lookup table.
"""
import numpy as np
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtable_path", required=True)
    parser.add_argument("--out", default="training/models/q_policy.json")
    args = parser.parse_args()

    Q = np.load(args.qtable_path)
    best_actions = np.argmax(Q, axis=1)
    policy = {"best_actions": best_actions.tolist(), "n_states": Q.shape[0], "n_actions": Q.shape[1]}
    with open(args.out, "w") as f:
        json.dump(policy, f, indent=2)
    print(f"Exported policy to {args.out}")

if __name__ == "__main__":
    main()
