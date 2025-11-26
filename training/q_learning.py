"""
q_learning.py
Train a discrete Q-learning agent on the SimpleRLEnv to learn sleep durations.
Saves a Q-table numpy file (if path supplied).
"""
import numpy as np
import argparse
from training.rl_env import SimpleRLEnv
import os

# Configuration of discretization bins
BATT_BINS = 10   # 0..9 -> 10 bins
EVENT_BINS = 5   # 0..4
TOD_BINS = 6     # 0..5

def qlearn(env, episodes=2000, alpha=0.1, gamma=0.99, eps=0.1):
    n_states = BATT_BINS * EVENT_BINS * TOD_BINS
    n_actions = len(env.sleep_choices)
    Q = np.zeros((n_states, n_actions))

    def state_to_idx(state):
        # ensure bins are within expected ranges
        batt_bin, event_bin, tod_bin = state
        # clamp each bin to valid range
        batt_bin = max(0, min(BATT_BINS - 1, int(batt_bin)))
        event_bin = max(0, min(EVENT_BINS - 1, int(event_bin)))
        tod_bin = max(0, min(TOD_BINS - 1, int(tod_bin)))
        return batt_bin * (EVENT_BINS * TOD_BINS) + event_bin * TOD_BINS + tod_bin

    for ep in range(episodes):
        state = env.reset()
        s_idx = state_to_idx(state)
        done = False
        while not done:
            if np.random.rand() < eps:
                a = np.random.randint(n_actions)
            else:
                a = np.argmax(Q[s_idx])
            next_state, reward, done, info = env.step(a)
            ns_idx = state_to_idx(next_state)
            # safeguard: ensure indices are valid
            if not (0 <= s_idx < n_states) or not (0 <= ns_idx < n_states):
                # defensive fallback: clamp
                s_idx = max(0, min(n_states - 1, int(s_idx)))
                ns_idx = max(0, min(n_states - 1, int(ns_idx)))
            Q[s_idx, a] += alpha * (reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, a])
            s_idx = ns_idx
        if (ep + 1) % max(1, (episodes // 10)) == 0:
            print(f"Episode {ep+1}/{episodes} complete")
    return Q

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--save", type=str, default="models/qtable.npy")
    args = parser.parse_args()
    os.makedirs("models", exist_ok=True)
    env = SimpleRLEnv()
    Q = qlearn(env, episodes=args.episodes)
    np.save(args.save, Q)
    print(f"Saved Q-table to {args.save}")

if __name__ == "__main__":
    main()
