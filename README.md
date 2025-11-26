# 1. Create and activate virtual env (recommended)
```
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell
```

# 2. Install dependencies
```
pip install -r requirements.txt
```

# 3. Generate dummy CSV data (default 24 hours, 3 nodes)
```
python3 -m  simulator.simulator_generate_traces --out data/dummy_traces.csv --nodes 3 --hours 24 --event-rate 1.2 --seed 42
```

# 4. Train Q-learning baseline (runs on dummy sim environment)
```
python3 -m  training.q_learning --episodes 2000 --save training/models/qtable.npy
```
Then
```
python training/policy_export.py --qtable_path training/models/qtable.npy --out training/models/q_policy.json
```

# 5. Train TinyML classifier (creates small Keras model and tflite)
```
python training/tinyml_train.py --input data/dummy_traces.csv --out training/models/tiny_model.tflite
```

# 6. Export policy table to a deployable JSON
```
python training/policy_export.py --qtable_path models/qtable.npy --out training/models/q_policy.json
```

# 7. Run evaluation script to compare policies (uses CSVs)
```
python experiments/evaluation.py --data data/dummy_traces.csv
```

# To run all project at once
```
chmod +x run_all.sh
./run_all.sh

```

# AI/ML-Based IoT Node Sleep Scheduler

## About
This project demonstrates an end-to-end pipeline to design, train, and evaluate adaptive sleep scheduling for battery-powered IoT nodes using dummy data only. It combines a simulator, a Q-learning RL trainer, and a TinyML classifier training pipeline.

## Repo layout
(see top of README for tree)

## Setup
1. Create virtual env & activate
2. `pip install -r requirements.txt`

## Key commands
- Generate data:
  `python simulator/simulator_generate_traces.py --out data/dummy_traces.csv --nodes 3 --hours 24 --event-rate 1.2 --seed 42`
- Train Q-learning:
  `python training/q_learning.py --episodes 2000 --save models/qtable.npy`
- Export policy:
  `python training/policy_export.py --qtable_path models/qtable.npy --out training/models/q_policy.json`
- Train TinyML classifier & export TFLite:
  `python training/tinyml_train.py --input data/dummy_traces.csv --out training/models/tiny_model.tflite`
- Run evaluation:
  `python experiments/evaluation.py --data data/dummy_traces.csv`

## How to replace dummy data with real ESP32 logs
1. Ensure your ESP32 logs match the CSV schema.
2. Place the logs in `data/` and rename to `real_device_traces.csv`.
3. Use the same training & export pipeline.

## Diagrams
Mermaid diagrams are in `docs/diagrams/`. Use mermaid.live or VS Code to render.

### Structure
iot-sleep-scheduler/
│
├── README.md
├── requirements.txt
│
├── simulator/
│   ├── __init__.py
│   ├── simulator_generate_traces.py
│   ├── energy_model.py
│   ├── event_generator.py
│   └── utils.py
│
├── data/
│   ├── dummy_traces.csv
│   ├── high_event_traces.csv
│   └── low_event_traces.csv
│
├── training/
│   ├── rl_env.py
│   ├── q_learning.py
│   ├── policy_export.py
│   ├── tinyml_classifier_training.ipynb
│   └── models/
│       └── (empty placeholders)
│
├── firmware/
│   ├── esp32_sleep_scheduler.cpp
│   ├── model_inference.cpp
│   └── README_firmware.md
│
├── docs/
│   ├── report.md
│   ├── architecture.md
│   ├── diagrams/
│   │   ├── system_architecture.mmd
│   │   ├── flowchart_sleep_logic.mmd
│   │   └── rl_pipeline.mmd
│   └── images/
│       └── (auto generated diagrams)
│
└── experiments/
    ├── evaluation.ipynb
    └── plots/


## License & Author
Author: <Dipanshu Sharma>  
License: MIT
