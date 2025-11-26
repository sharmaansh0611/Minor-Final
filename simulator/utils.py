"""
utils.py
Utility helpers used by the simulator.
"""
from datetime import datetime, timedelta

def iso_now():
    return datetime.utcnow().isoformat()

def advance_time_iso(iso_ts, seconds):
    from datetime import datetime, timedelta
    t = datetime.fromisoformat(iso_ts)
    return (t + timedelta(seconds=seconds)).isoformat()
