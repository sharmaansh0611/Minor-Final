# # dashboard/app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import json
# import subprocess

# st.set_page_config(page_title="IoT Sleep Scheduler Dashboard", layout="wide")
# st.title("📡 IoT Sleep Scheduler — Dashboard + Battery Lifetime Simulator")

# # ----------------------------
# # Constants
# # ----------------------------
# V_NOMINAL = 3.7
# BATTERY_CAP_AH_DEFAULT = 2.0
# ACTIVE_CURRENT_A_DEFAULT = 0.020
# DEEP_SLEEP_CURRENT_A_DEFAULT = 8e-6
# TX_ENERGY_J_PER_TX_DEFAULT = 0.5

# def full_joules(cap_ah):
#     return V_NOMINAL * cap_ah * 3600

# def batt_pct_from_j(rem, full):
#     return max(0, min(100, (rem/full)*100))

# # ----------------------------
# # CSV load
# # ----------------------------
# st.sidebar.header("Data")
# uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
# default_csv = "data/dummy_traces.csv"

# csv_path = uploaded if uploaded else (default_csv if os.path.exists(default_csv) else None)

# if not csv_path:
#     st.sidebar.error("No CSV found. Upload a CSV.")
#     st.stop()

# @st.cache_data(show_spinner=False)
# def load_csv(fp):
#     df = pd.read_csv(fp)
#     if "timestamp" in df.columns:
#         df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#     else:
#         df["timestamp"] = pd.date_range(start=pd.Timestamp.now(), periods=len(df), freq="T")
#     if "sensor1" not in df.columns:
#         df["sensor1"] = 0.0
#     df = df.sort_values("timestamp").reset_index(drop=True)
#     return df

# df = load_csv(csv_path)
# st.subheader("CSV Preview")
# st.dataframe(df.head(20))

# # ----------------------------
# # Sidebar controls
# # ----------------------------
# st.sidebar.header("Battery Model")
# battery_cap_ah = st.sidebar.number_input("Battery (Ah)", value=BATTERY_CAP_AH_DEFAULT)
# active_current_ma = st.sidebar.number_input("Active current (mA)", value=ACTIVE_CURRENT_A_DEFAULT*1000)
# sleep_current_ua = st.sidebar.number_input("Sleep current (µA)", value=DEEP_SLEEP_CURRENT_A_DEFAULT*1e6)
# tx_energy_j = st.sidebar.number_input("TX energy (J)", value=TX_ENERGY_J_PER_TX_DEFAULT)

# ACTIVE_CURRENT_A = active_current_ma/1000
# DEEP_SLEEP_CURRENT_A = sleep_current_ua/1e6
# TX_ENERGY_J_PER_TX = tx_energy_j
# FULL_J = full_joules(battery_cap_ah)

# st.sidebar.header("Simulation Policy")
# policy = st.sidebar.selectbox("Policy", ["Fixed duty cycle", "RL policy (q_policy.json)", "TinyML heuristic"])
# fixed_sleep_s = st.sidebar.number_input("Fixed sleep duration (s)", value=60)

# rl_policy_path = "training/models/q_policy.json"
# tiny_h5_path = "training/models/tiny_model.h5"

# enable_tinyml = st.sidebar.checkbox("Enable TinyML (uses subprocess + timeout)", value=False)

# if policy == "TinyML heuristic" and not enable_tinyml:
#     st.sidebar.info("TinyML disabled → using fallback rule.")

# st.sidebar.header("Simulation Params")
# max_hours = st.sidebar.number_input("Max hours", value=240)
# time_resolution_s = st.sidebar.number_input("Tick resolution (s)", value=10)

# # ----------------------------
# # Heavy helpers
# # ----------------------------
# @st.cache_data(show_spinner=False)
# def prepare_features(df):
#     local = df.copy()
#     if "sensor1" not in local.columns:
#         local["sensor1"] = 0.0
#     local["sensor1_rolling"] = local["sensor1"].rolling(5, min_periods=1).mean()
#     local["time_of_day_s"] = (
#         local["timestamp"].dt.hour*3600 +
#         local["timestamp"].dt.minute*60 +
#         local["timestamp"].dt.second
#     )
#     return local

# def load_rl_policy(path):
#     try:
#         with open(path, "r") as f:
#             return json.load(f)
#     except:
#         return None

# # ----------------------------
# # Subprocess TinyML inference (SAFE)
# # ----------------------------
# def tinyml_predict_subprocess(model_path, features, timeout_s=8):
#     worker = os.path.join(os.getcwd(), "tinyml_infer.py")
#     if not os.path.exists(worker):
#         return None

#     payload = {"model_path": model_path, "features": features}

#     try:
#         proc = subprocess.run(
#             [worker],
#             input=json.dumps(payload).encode(),
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             timeout=timeout_s
#         )
#     except subprocess.TimeoutExpired:
#         return None
#     except Exception:
#         return None

#     if proc.returncode != 0:
#         return None

#     try:
#         out = json.loads(proc.stdout.decode().strip())
#         return out.get("pred")
#     except:
#         return None

# # ----------------------------
# # Simulation
# # ----------------------------
# SLEEP_CHOICES = [30, 60, 120, 300]

# def simulate_battery(policy, df, rl_policy):
#     remaining_j = FULL_J
#     rows = []

#     t_start = df["timestamp"].iloc[0]
#     t_end = t_start + pd.Timedelta(hours=max_hours)

#     idx = 0
#     n = len(df)

#     def get_row(ts):
#         nonlocal idx
#         while idx < n and df["timestamp"].iloc[idx] < ts:
#             idx += 1
#         return df.iloc[min(idx, n-1)]

#     ts = t_start
#     steps = int((t_end - t_start).total_seconds()/time_resolution_s)

#     for _ in range(steps):
#         row = get_row(ts)

#         # --- Decision for TinyML ---
#         if policy == "TinyML heuristic":
#             rule_event = row["sensor1"] > 0

#             if not enable_tinyml or not os.path.exists(tiny_h5_path):
#                 event = rule_event
#             else:
#                 b_pct = batt_pct_from_j(remaining_j, FULL_J)
#                 feat = [
#                     float(row.get("sensor0", 0)),
#                     float(row["sensor1"]),
#                     float(row["sensor1_rolling"]),
#                     float(b_pct),
#                     float(row.get("rssi", -80))
#                 ]
#                 pred = tinyml_predict_subprocess(tiny_h5_path, feat)
#                 event = (pred > 0.5) if pred is not None else rule_event

#             sleep_s = fixed_sleep_s if not event else 0

#         # --- Fixed policy ---
#         elif policy == "Fixed duty cycle":
#             event = row["sensor1"] > 0
#             sleep_s = fixed_sleep_s

#         # --- RL Policy ---
#         else:
#             event = row["sensor1"] > 0
#             sleep_s = fixed_sleep_s  # (simple fallback unless you want full RL)

#         # --- Energy use ---
#         if event:
#             active_s = 2.0
#             tx_attempts = 1
#         else:
#             active_s = 0.5
#             tx_attempts = 0

#         active_j = ACTIVE_CURRENT_A * V_NOMINAL * active_s
#         sleep_j = DEEP_SLEEP_CURRENT_A * V_NOMINAL * sleep_s
#         tx_j = tx_attempts * TX_ENERGY_J_PER_TX

#         remaining_j = max(0, remaining_j - (active_j + sleep_j + tx_j))

#         rows.append({
#             "time": ts,
#             "battery_pct": batt_pct_from_j(remaining_j, FULL_J),
#             "event": int(event),
#             "sleep_s": sleep_s
#         })

#         ts = ts + pd.Timedelta(seconds=time_resolution_s)
#         if remaining_j <= 0:
#             break

#     return pd.DataFrame(rows)

# # -------------------------------------------------------
# #   Multi-policy comparison helper
# # -------------------------------------------------------
# def compare_policies(df_prepared, rl_policy):
#     results = {}

#     # --- Fixed Duty Cycle ---
#     sim_fixed = simulate_battery("Fixed duty cycle", df_prepared, rl_policy)
#     results["Fixed Duty"] = sim_fixed

#     # --- RL Policy ---
#     sim_rl = simulate_battery("RL policy (q_policy.json)", df_prepared, rl_policy)
#     results["RL Policy"] = sim_rl

#     # --- TinyML ---
#     # TinyML might fail; safe fallback included
#     sim_tiny = simulate_battery("TinyML heuristic", df_prepared, rl_policy)
#     results["TinyML"] = sim_tiny

#     return results


# # ----------------------------
# # UI
# # ----------------------------
# st.subheader("🔋 Battery Lifetime Simulator")
# if st.button("Run Simulation"):
#     st.subheader("📊 Compare All Policies")
# if st.button("Compare Policies"):
#     with st.spinner("Running all policies..."):
#         df_prepared = prepare_features(df)
#         rl_p = load_rl_policy(rl_policy_path) if os.path.exists(rl_policy_path) else None
#         results = compare_policies(df_prepared, rl_p)

#     # -------------------------
#     # Chart: All Policies
#     # -------------------------
#     fig, ax = plt.subplots(figsize=(10,5))
#     for name, sim in results.items():
#         if not sim.empty:
#             ax.plot(sim["time"], sim["battery_pct"], label=name)

#     ax.set_ylabel("Battery %")
#     ax.set_xlabel("Time")
#     ax.set_title("Battery Lifetime Comparison Across Policies")
#     ax.legend()
#     st.pyplot(fig)

#     # -------------------------
#     # Display lifetimes
#     # -------------------------
#     lifetime_table = []
#     for name, sim in results.items():
#         if sim.empty:
#             lifetime_table.append([name, "Error"])
#             continue
#         drained = sim[sim["battery_pct"] <= 1]
#         if drained.empty:
#             lifetime_h = (sim["time"].iloc[-1] - sim["time"].iloc[0]).total_seconds()/3600
#         else:
#             lifetime_h = (drained["time"].iloc[0] - sim["time"].iloc[0]).total_seconds()/3600
#         lifetime_table.append([name, f"{lifetime_h:.2f} hrs"])

#     st.subheader("🔍 Estimated Lifetimes")
#     st.table(pd.DataFrame(lifetime_table, columns=["Policy", "Lifetime"]))

#     df_p = prepare_features(df)
#     rl_p = load_rl_policy(rl_policy_path) if os.path.exists(rl_policy_path) else None

#     with st.spinner("Simulating..."):
#         sim = simulate_battery(policy, df_p, rl_p)

#     if sim.empty:
#         st.error("Simulation failed.")
#     else:
#         drained = sim[sim["battery_pct"] <= 1]
#         if drained.empty:
#             lifetime_h = (sim["time"].iloc[-1] - sim["time"].iloc[0]).total_seconds()/3600
#         else:
#             lifetime_h = (drained["time"].iloc[0] - sim["time"].iloc[0]).total_seconds()/3600

#         st.metric("Estimated lifetime", f"{lifetime_h:.2f} hours ({lifetime_h/24:.2f} days)")

#         fig, ax = plt.subplots(figsize=(10,4))
#         ax.plot(sim["time"], sim["battery_pct"])
#         ax.set_ylabel("Battery %")
#         ax.set_xlabel("Time")
#         st.pyplot(fig)

#         st.dataframe(sim.head(200))

#         os.makedirs("experiments", exist_ok=True)
#         sim.to_csv("experiments/sim_run.csv", index=False)
#         st.success("Saved simulations → experiments/sim_run.csv")

# st.caption("TinyML is sandboxed in a subprocess to avoid hanging Streamlit.")




# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import subprocess

st.set_page_config(page_title="IoT Sleep Scheduler Dashboard", layout="wide")
st.title("📡 IoT Sleep Scheduler — Dashboard + Battery Lifetime Simulator")

# ----------------------------
# Energy Model Constants
# ----------------------------
V_NOMINAL = 3.7
BATTERY_CAP_AH_DEFAULT = 2.0
ACTIVE_CURRENT_A_DEFAULT = 0.020
DEEP_SLEEP_CURRENT_A_DEFAULT = 8e-6
TX_ENERGY_J_PER_TX_DEFAULT = 0.5

def full_joules(cap_ah):
    return V_NOMINAL * cap_ah * 3600

def batt_pct_from_j(rem, full):
    return max(0, min(100, (rem/full)*100))

# ----------------------------
# CSV Loading
# ----------------------------
st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
default_csv = "data/dummy_traces.csv"

csv_path = uploaded if uploaded else (default_csv if os.path.exists(default_csv) else None)
if not csv_path:
    st.sidebar.error("No CSV provided.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_csv(fp):
    df = pd.read_csv(fp)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.date_range(start=pd.Timestamp.now(), periods=len(df), freq="T")
    if "sensor1" not in df.columns:
        df["sensor1"] = 0.0
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

df = load_csv(csv_path)
st.subheader("📄 CSV Preview")
st.dataframe(df.head(20))

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("Battery Model")
battery_cap_ah = st.sidebar.number_input("Battery (Ah)", value=BATTERY_CAP_AH_DEFAULT)
active_current_ma = st.sidebar.number_input("Active current (mA)", value=ACTIVE_CURRENT_A_DEFAULT*1000)
sleep_current_ua = st.sidebar.number_input("Sleep current (µA)", value=DEEP_SLEEP_CURRENT_A_DEFAULT*1e6)
tx_energy_j = st.sidebar.number_input("TX energy (J)", value=TX_ENERGY_J_PER_TX_DEFAULT)

ACTIVE_CURRENT_A = active_current_ma/1000
DEEP_SLEEP_CURRENT_A = sleep_current_ua/1e6
TX_ENERGY_J_PER_TX = tx_energy_j
FULL_J = full_joules(battery_cap_ah)

st.sidebar.header("Simulation Policy")
policy = st.sidebar.selectbox("Policy", ["Fixed duty cycle", "RL policy", "TinyML heuristic"])
fixed_sleep_s = st.sidebar.number_input("Fixed sleep duration (s)", value=60)

rl_policy_path = "training/models/q_policy.json"
tiny_h5_path = "training/models/tiny_model.h5"

enable_tinyml = st.sidebar.checkbox("Enable TinyML (subprocess safe mode)", value=False)
if policy == "TinyML heuristic" and not enable_tinyml:
    st.sidebar.info("TinyML disabled → fallback rule will be used.")

st.sidebar.header("Sim Parameters")
max_hours = st.sidebar.number_input("Max hours", value=240)
time_resolution_s = st.sidebar.number_input("Tick resolution (s)", value=10)

# ----------------------------
# Heavy Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def prepare_features(df):
    local = df.copy()
    local["sensor1_rolling"] = local["sensor1"].rolling(5, min_periods=1).mean()
    local["time_of_day_s"] = (
        local["timestamp"].dt.hour*3600 +
        local["timestamp"].dt.minute*60 +
        local["timestamp"].dt.second
    )
    return local

def load_rl_policy(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None
# ----------------------------------------------------
# TinyML Subprocess Predictor (SAFE — never hangs UI)
# ----------------------------------------------------
def tinyml_predict_subprocess(model_path, features, timeout_s=6):
    worker = os.path.join(os.getcwd(), "tinyml_infer.py")
    if not os.path.exists(worker):
        return None

    payload = {"model_path": model_path, "features": features}

    try:
        proc = subprocess.run(
            [worker],
            input=json.dumps(payload).encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

    if proc.returncode != 0:
        return None

    try:
        out = json.loads(proc.stdout.decode().strip())
        return out.get("pred")
    except:
        return None

# ----------------------------------------------------
# Simulation Function (with stats tracking)
# ----------------------------------------------------
SLEEP_CHOICES = [30, 60, 120, 300]

def simulate_battery(policy, df, rl_policy):
    remaining_j = FULL_J
    rows = []
    stats = {"active_j":0, "sleep_j":0, "tx_j":0, "wakeups":0}

    t_start = df["timestamp"].iloc[0]
    t_end = t_start + pd.Timedelta(hours=max_hours)

    idx = 0
    n = len(df)

    def get_row(ts):
        nonlocal idx
        while idx < n and df["timestamp"].iloc[idx] < ts:
            idx += 1
        return df.iloc[min(idx, n-1)]

    ts = t_start
    steps = int((t_end - t_start).total_seconds() / time_resolution_s)

    for _ in range(steps):
        row = get_row(ts)

        # ------------------------------------------------------
        # Decision Logic Based on Policy
        # ------------------------------------------------------
        # --- TinyML Heuristic ---
        if policy == "TinyML heuristic":
            rule_event = row["sensor1"] > 0

            if not enable_tinyml or not os.path.exists(tiny_h5_path):
                event = rule_event
            else:
                b_pct_now = batt_pct_from_j(remaining_j, FULL_J)
                feat = [
                    float(row.get("sensor0", 0)),
                    float(row["sensor1"]),
                    float(row["sensor1_rolling"]),
                    float(b_pct_now),
                    float(row.get("rssi", -80))
                ]

                p = tinyml_predict_subprocess(tiny_h5_path, feat)
                event = (p > 0.5) if p is not None else rule_event

            sleep_s = fixed_sleep_s if not event else 0

        # --- Fixed Duty ---
        elif policy == "Fixed duty cycle":
            event = row["sensor1"] > 0
            sleep_s = fixed_sleep_s

        # --- RL Policy ---
                # --- RL Policy ---
        else:
            # default rule for event detection (used for reward calc / logging)
            event = row["sensor1"] > 0

            # if no RL policy supplied, fallback to fixed
            if rl_policy is None:
                sleep_s = fixed_sleep_s
            else:
                # Expect rl_policy to contain a list "best_actions" and optionally "n_states"
                best_actions = rl_policy.get("best_actions") if isinstance(rl_policy, dict) else None
                if not best_actions:
                    # try if rl_policy itself is a list
                    if isinstance(rl_policy, list):
                        best_actions = rl_policy
                if not best_actions:
                    sleep_s = fixed_sleep_s
                else:
                    # replicate discretization used during training:
                    # batt bins = 10 (0..9) by battery% // 10
                    # event bins = 5 (0..4) from sensor1_rolling scaled
                    # tod bins = 6 (0..5) from time_of_day_s
                    batt_pct_now = max(0.0, min(100.0, (remaining_j / FULL_J) * 100.0))
                    batt_bin = int(batt_pct_now // 10)
                    # event_bin: sensor1_rolling is typically in 0..1 -> scale to 0..4
                    ev_rate = float(row.get("sensor1_rolling", 0.0))
                    event_bin = int(min(4, max(0, int(ev_rate * 5))))
                    tod_bin = int((row.get("time_of_day_s", 0) % 86400) // (86400 / 6))

                    # compute state index (same formula used in training)
                    state_idx = batt_bin * (5 * 6) + event_bin * 6 + tod_bin
                    # clamp to valid range
                    max_idx = len(best_actions) - 1
                    if state_idx < 0:
                        state_idx = 0
                    if state_idx > max_idx:
                        state_idx = max_idx

                    action_idx = int(best_actions[state_idx])
                    if 0 <= action_idx < len(SLEEP_CHOICES):
                        sleep_s = SLEEP_CHOICES[action_idx]
                    else:
                        sleep_s = fixed_sleep_s


        # ------------------------------------------------------
        # Energy Model
        # ------------------------------------------------------
        if event:
            active_s = 2.0
            tx_attempts = 1
            stats["wakeups"] += 1
        else:
            active_s = 0.5
            tx_attempts = 0

        active_j = ACTIVE_CURRENT_A * V_NOMINAL * active_s
        sleep_j = DEEP_SLEEP_CURRENT_A * V_NOMINAL * sleep_s
        tx_j = tx_attempts * TX_ENERGY_J_PER_TX

        stats["active_j"] += active_j
        stats["sleep_j"] += sleep_j
        stats["tx_j"] += tx_j

        remaining_j = max(0, remaining_j - (active_j + sleep_j + tx_j))

        rows.append({
            "time": ts,
            "battery_pct": batt_pct_from_j(remaining_j, FULL_J),
            "event": int(event),
            "sleep_s": sleep_s
        })

        ts = ts + pd.Timedelta(seconds=time_resolution_s)
        if remaining_j <= 0:
            break

    sim_df = pd.DataFrame(rows)
    sim_df.attrs["stats"] = stats
    return sim_df

# ----------------------------
# UI: Single simulation run
# ----------------------------
st.subheader("🔋 Battery Lifetime Simulator")

col_run, col_cmp = st.columns([2,1])
with col_run:
    run_clicked = st.button("Run Simulation")

with col_cmp:
    compare_clicked = st.button("Compare Policies")

if run_clicked:
    with st.spinner("Preparing features and running simulation..."):
        df_p = prepare_features(df)
        rl_p = load_rl_policy(rl_policy_path) if os.path.exists(rl_policy_path) else None
        sim_df = simulate_battery(policy, df_p, rl_p)

    if sim_df.empty:
        st.error("Simulation returned no data. Check CSV and parameters.")
    else:
        # Lifetime metric
        drained = sim_df[sim_df["battery_pct"] <= 1.0]
        if drained.empty:
            lifetime_h = (sim_df["time"].iloc[-1] - sim_df["time"].iloc[0]).total_seconds()/3600
        else:
            lifetime_h = (drained["time"].iloc[0] - sim_df["time"].iloc[0]).total_seconds()/3600

        st.metric("Estimated lifetime", f"{lifetime_h:.2f} hours ({lifetime_h/24:.2f} days)")

        # Battery plot
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(sim_df["time"], sim_df["battery_pct"], label=f"{policy}")
        ax.set_ylabel("Battery (%)")
        ax.set_xlabel("Time")
        ax.set_title("Battery % over time")
        ax.legend()
        st.pyplot(fig)

        # Energy breakdown
        st.subheader("⚡ Energy Breakdown")
        stats = sim_df.attrs.get("stats", {"active_j":0,"sleep_j":0,"tx_j":0})
        labels = ["Active (J)", "Sleep (J)", "TX (J)"]
        values = [stats.get("active_j",0), stats.get("sleep_j",0), stats.get("tx_j",0)]

        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.bar(labels, values)
        ax2.set_ylabel("Energy (Joules)")
        ax2.set_title("Energy consumption by mode")
        st.pyplot(fig2)

        # Save simulation CSV and show small table
        os.makedirs("experiments", exist_ok=True)
        out_path = "experiments/sim_run_single.csv"
        sim_df.to_csv(out_path, index=False)
        st.success(f"Saved simulation trace to {out_path}")

        st.subheader("Simulation sample (first 200 rows)")
        st.dataframe(sim_df.head(200))

# ----------------------------
# Multi-policy comparison
# ----------------------------
def compare_policies(df_prepared, rl_policy):
    results = {}
    results["Fixed Duty"] = simulate_battery("Fixed duty cycle", df_prepared, rl_policy)
    results["RL Policy"] = simulate_battery("RL policy", df_prepared, rl_policy)
    results["TinyML"] = simulate_battery("TinyML heuristic", df_prepared, rl_policy)
    return results

if compare_clicked:
    with st.spinner("Running all policies..."):
        df_p = prepare_features(df)
        rl_p = load_rl_policy(rl_policy_path) if os.path.exists(rl_policy_path) else None
        res = compare_policies(df_p, rl_p)

    # Overlay chart
    figc, axc = plt.subplots(figsize=(10,5))
    for name, sim in res.items():
        if sim is not None and not sim.empty:
            axc.plot(sim["time"], sim["battery_pct"], label=name)
    axc.set_ylabel("Battery (%)")
    axc.set_xlabel("Time")
    axc.set_title("Battery Comparison Across Policies")
    axc.legend()
    st.pyplot(figc)

    # Lifetimes table
    rows = []
    for name, sim in res.items():
        if sim is None or sim.empty:
            rows.append([name, "error"])
            continue
        drained = sim[sim["battery_pct"] <= 1.0]
        if drained.empty:
            lifetime = (sim["time"].iloc[-1] - sim["time"].iloc[0]).total_seconds()/3600
        else:
            lifetime = (drained["time"].iloc[0] - sim["time"].iloc[0]).total_seconds()/3600
        rows.append([name, f"{lifetime:.2f} hrs"])

    st.subheader("🔍 Estimated lifetimes")
    st.table(pd.DataFrame(rows, columns=["Policy","Lifetime"]))

    # Wakeups chart and table
    wake_rows = []
    for name, sim in res.items():
        stats = sim.attrs.get("stats", {"wakeups":0})
        wake_rows.append([name, stats.get("wakeups",0)])

    wake_df = pd.DataFrame(wake_rows, columns=["Policy","Wakeups"])
    figw, axw = plt.subplots(figsize=(6,4))
    axw.bar(wake_df["Policy"], wake_df["Wakeups"])
    axw.set_ylabel("Number of wakeups")
    axw.set_title("Wakeups per policy")
    st.pyplot(figw)
    st.table(wake_df)

    # Save comparison CSVs
    os.makedirs("experiments", exist_ok=True)
    for name, sim in res.items():
        safe_name = name.replace(" ", "_").lower()
        if sim is not None and not sim.empty:
            sim.to_csv(f"experiments/sim_{safe_name}.csv", index=False)
    st.success("Saved per-policy simulation CSVs to experiments/")

# ----------------------------
# Footer: downloads, help, and notes
# ----------------------------
st.markdown("---")
st.subheader("💾 Downloads & Help")

# Download latest single-run CSV if exists
single_path = "experiments/sim_run_single.csv"
if os.path.exists(single_path):
    with open(single_path, "rb") as f:
        st.download_button("Download last simulation CSV", data=f, file_name="sim_run_single.csv", mime="text/csv")
else:
    st.info("Run a single simulation to enable CSV download.")

# Download comparison CSVs as zip if present
import glob
comp_files = glob.glob("experiments/sim_*.csv")
if comp_files:
    import io, zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for p in comp_files:
            zf.write(p, arcname=os.path.basename(p))
    zip_buffer.seek(0)
    st.download_button("Download comparison CSVs (zip)", data=zip_buffer, file_name="sim_comparison.zip", mime="application/zip")
else:
    st.info("Run 'Compare Policies' to generate per-policy CSVs for download.")

st.markdown("**Notes & tips**")
st.markdown("""
- TinyML inference is sandboxed in a subprocess (`tinyml_infer.py`) with a timeout to prevent the Streamlit UI from hanging.  
- To enable TinyML, check the **Enable TinyML** checkbox and ensure `training/models/tiny_model.h5` exists. If the worker fails or times out, the dashboard falls back to the `sensor1>0` rule.  
- For full RL behavior, implement the RL-to-action mapping in `simulate_battery()` (currently uses a fixed-sleep fallback). The RL policy exporter should produce `training/models/q_policy.json`.  
- If Streamlit becomes unresponsive, kill old processes: `pkill -f streamlit` and restart: `python -m streamlit run dashboard/app.py --server.port 8502`.  
""")

st.caption("IoT Sleep Scheduler Dashboard — safe TinyML integration, multi-policy comparison, and energy analytics.")

