"""
energy_model.py
Simple energy model functions used by the simulator.
"""

VOLT_FULL = 4.2
VOLT_EMPTY = 3.2
BATTERY_CAP_AH = 2.0  # 2000 mAh
V_NOMINAL = 3.7

ACTIVE_CURRENT_A = 0.020    # 20 mA
DEEP_SLEEP_CURRENT_A = 8e-6 # 8 microamps
TX_ENERGY_J_PER_TX = 0.5    # Joules per transmission (placeholder)

def full_joules():
    return V_NOMINAL * BATTERY_CAP_AH * 3600.0

def energy_active_seconds(seconds):
    return ACTIVE_CURRENT_A * V_NOMINAL * seconds

def energy_sleep_seconds(seconds):
    return DEEP_SLEEP_CURRENT_A * V_NOMINAL * seconds

def energy_for_tx(attempts):
    return attempts * TX_ENERGY_J_PER_TX

def battery_pct_from_joules(remaining_joules):
    fj = full_joules()
    return max(0.0, min(100.0, (remaining_joules / fj) * 100.0))

def voltage_from_pct(pct):
    # Linear interpolation between empty and full voltage
    return VOLT_EMPTY + (VOLT_FULL - VOLT_EMPTY) * (pct / 100.0)
