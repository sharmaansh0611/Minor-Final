#!/usr/bin/env bash
set -e

echo "🔧 Initializing environment..."

# 1) Activate virtualenv
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "❌ ERROR: .venv not found. Create it using:"
    echo "python3 -m venv .venv && source .venv/bin/activate"
    exit 1
fi

echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo "📊 Generating dummy data..."
mkdir -p data
python -m simulator.simulator_generate_traces --out data/dummy_traces.csv --nodes 3 --hours 24 --event-rate 1.2 --seed 42

echo "🤖 Training Q-learning RL agent..."
mkdir -p training/models
python -m training.q_learning --episodes 500 --save training/models/qtable.npy

echo "📤 Exporting RL policy..."
python -m training.policy_export --qtable_path training/models/qtable.npy --out training/models/q_policy.json

echo "🧠 Training TinyML classifier..."
python -m training.tinyml_train --input data/dummy_traces.csv --out training/models/tiny_model.tflite

echo "📈 Running evaluation..."
python -m experiments.evaluation --data data/dummy_traces.csv

echo "🎉 ALL DONE!"
echo "👉 Outputs generated:"
echo "   • data/dummy_traces.csv"
echo "   • training/models/qtable.npy"
echo "   • training/models/q_policy.json"
echo "   • training/models/tiny_model.tflite"
echo "   • experiments/plots/"
echo
echo "You can now run: streamlit run dashboard/app.py"
