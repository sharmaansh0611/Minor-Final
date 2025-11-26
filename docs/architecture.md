# Architecture & Design Notes

## High-level components
- Node: sensors + MCU + TinyML runtime + sleep controller
- Gateway/Cloud: collects telemetry and may push updates
- Simulator/Trainer: generates data, trains RL/TinyML

## Data flow
1. Simulator produces CSV traces for training.
2. Training scripts (tinyml_train.py and q_learning.py) create artifacts.
3. policy_export.py creates a compact policy for firmware.
4. Firmware uses TinyML model for quick wake decisions and policy table for long-term scheduling.

## Diagrams
Mermaid files are in docs/diagrams/*.mmd. Use mermaid.live or VS Code mermaid preview to render.
