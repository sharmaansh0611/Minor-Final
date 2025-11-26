# AI/ML-Based IoT Node Sleep Scheduler (TinyML + Reinforcement Learning)

## Abstract
This project designs, implements, and evaluates an adaptive sleep scheduler for battery-powered IoT nodes using a hybrid AI/ML approach. A simulator generates realistic dummy traces; a Q-learning agent learns energy-aware sleep policies; a TinyML classifier performs on-device event detection. The project includes firmware stubs, dataset generation, training code, and evaluation scripts — all runnable without hardware.

## 1. Introduction & Motivation
Battery life is the primary constraint for many IoT deployments. Efficient sleep/wake scheduling dramatically extends lifetime while meeting application latency requirements. This work investigates combining offline RL (to optimize long-term battery lifetime) with on-device TinyML (for immediate event detection) to create a practical, deployable scheduler.

## 2. Related Work
- Duty-cycle optimization and adaptive scheduling in wireless sensor networks.
- TinyML for on-device classification.
- RL for long-term policy optimization in resource-constrained systems.

## 3. Problem Statement
Design a policy `π(state) -> sleep_duration` that maximizes battery lifetime subject to constraints on event latency and successful transmissions.

### 3.1 Objectives
- Build simulator & generate datasets
- Train RL policy and distill to a compact format
- Train TinyML classifier for event detection
- Compare baseline heuristics, TinyML-only, and RL-distilled policies

## 4. System Architecture
(See `docs/architecture.md` and Mermaid diagram `docs/diagrams/system_architecture.mmd`)

Components:
- IoT Node (ESP32 / MCU) w/ TinyML
- Gateway / Cloud (offline trainer)
- Simulator & trainer (PC)
- Data storage & evaluation

## 5. Simulator & Energy Model
Energy model uses measured-order currents:
- Active: ~20 mA
- Deep-sleep: ~8 µA
- Transmission: modeled as fixed Joules per attempt

Simulator supports Poisson & bursty events, configurable node counts, packet loss, and produces CSV traces.

## 6. ML Approach
- TinyML: binary classifier `event / no-event` using sensor readings and short rolling features. Converted to TFLite.
- RL: Q-learning on discretized state space (battery bin, event rate bin, time-of-day bin) choosing discrete sleep durations. Q-table exported as JSON.

## 7. Implementation
Code organized into `simulator/`, `training/`, `firmware/`, `docs/`.

Key scripts:
- `simulator/simulator_generate_traces.py`
- `training/q_learning.py`
- `training/tinyml_train.py`
- `training/policy_export.py`

## 8. Firmware Integration
Firmware stubs show using RTC retention and how to trigger deep-sleep. Integration steps with TinyML described in `firmware/README_firmware.md`.

## 9. Experiments & Results
Provided evaluation script `experiments/evaluation.py` generates baseline metrics and plots. Example metrics:
- Energy per simulated day (J)
- Average latency to event
- % events detected by TinyML

(Insert results after running scripts on generated datasets; include plots into this report.)

## 10. Discussion & Limitations
- Simulator approximations must be calibrated against real hardware for production deployment.
- TinyML classifier accuracy depends on quality of sensor features and labels.

## 11. Future Work
- Use DQN or policy gradient methods for larger state spaces.
- On-device online learning or federated updates.
- Real-device validation and power-trace calibration.

## 12. Conclusion
This project demonstrates an end-to-end pipeline for training and deploying an energy-aware scheduler for IoT nodes using dummy data. It is reproducible and designed to be extended to real hardware.

## References
1. ESP-IDF power management docs (for current consumption numbers)  
2. TinyML literature (for model quantization & deployment)  
3. RL in resource-constrained systems (scheduling papers)

