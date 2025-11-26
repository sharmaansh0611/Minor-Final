"""
rl_env.py
A lightweight Gym-style environment wrapper that uses the simulator logic.
State: [battery_bin, recent_event_rate_bin, time_of_day_bin]
Actions: discrete sleep durations (index to choices)
Reward: -energy_cost + event_reward - latency_penalty
This is simplified and uses the energy_model functions for reward calc.
"""
import numpy as np
import random
from simulator.energy_model import energy_active_seconds, energy_sleep_seconds, energy_for_tx, full_joules

class SimpleRLEnv:
    def __init__(self, sleep_choices=[30,60,120,300], max_steps=500, seed=None):
        self.sleep_choices = sleep_choices
        self.max_steps = max_steps
        self.random = random.Random(seed)
        self.reset()

    def reset(self):
        self.step_count = 0
        self.batt_j = full_joules()
        self.time_of_day = 0.0
        # synthetic initial state
        return self._get_state()

    def _get_state(self):
        # discretize battery into 10 bins, event rate into 5, tod into 6
        batt_pct = max(0.0, min(100.0, (self.batt_j / full_joules()) * 100.0))
        batt_bin = int(batt_pct // 10)
        event_rate_bin = int(self.random.random() * 5)
        tod_bin = int((self.time_of_day % 86400) // (86400 / 6))
        return (batt_bin, event_rate_bin, tod_bin)

    def step(self, action_idx):
        # action chooses sleep duration
        sleep_s = self.sleep_choices[action_idx]
        # simulate an active duration and potential event
        event_happened = self.random.random() < 0.2
        active_s = 2.0 if event_happened else 0.8
        tx_attempts = 1 if event_happened else 0

        energy = energy_active_seconds(active_s) + energy_sleep_seconds(sleep_s) + energy_for_tx(tx_attempts)
        self.batt_j = max(0.0, self.batt_j - energy)
        reward = -energy  # lower energy is better (higher reward)
        # extra reward for handling events successfully
        if event_happened:
            reward += 0.5  # reward for processing event
        # penalty if battery low
        if self.batt_j <= 0.1 * full_joules():
            reward -= 5.0

        self.time_of_day += sleep_s
        self.step_count += 1
        done = self.step_count >= self.max_steps or self.batt_j <= 0.0
        return self._get_state(), reward, done, {"energy": energy, "batt_j": self.batt_j}
