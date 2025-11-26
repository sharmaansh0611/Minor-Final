"""
event_generator.py
Generates Poisson / bursty events for the simulator.
"""
import math
import random

def poisson_event(lambda_per_step, rng):
    # approximate Poisson: P(at least 1) = 1 - exp(-lambda)
    return rng.random() < (1.0 - math.exp(-lambda_per_step))

def bursty_multiplier(timestep_seconds, day_seconds=86400):
    # optional time-of-day burstiness multiplier: higher during day (example)
    # returns multiplier in [0.5, 2.0]
    # peak around midday (sin wave)
    import math
    phase = (timestep_seconds % day_seconds) / day_seconds
    return 1.0 + 0.8 * math.sin(2 * math.pi * phase)  # in [-0.8, 0.8] -> [0.2,1.8]
