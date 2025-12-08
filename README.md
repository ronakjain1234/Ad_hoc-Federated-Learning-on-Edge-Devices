# Cost-Aware Dynamic Rerouting for Resilient Federated Learning

## Project Overview
This project simulates a Federated Learning (FL) system operating over an ad-hoc edge network. It models physical layer disturbances—such as node dropouts, link failures, and battery depletion—and evaluates a proposed "Cost-Aware Dynamic Rerouting" strategy against a standard Naive baseline.

## Project Structure
adhocfl-baseline/
├── configs/               # Experiment configurations (YAML)
│   ├── baseline.yaml      # Standard FL without disturbances
│   └── disturbances.yaml  # Main config for disturbances & recovery experiments
├── scripts/
│   └── run_baseline.py    # Main entry point script
├── src/adhocfl/           # Core source code
│   ├── orchestrator.py    # Main FL loop (training, sampling, aggregation)
│   ├── network.py         # Graph topology, routing logic, and smart sampling
│   ├── disturbances.py    # Failure injection (dropouts, throttling)
│   ├── device.py          # Battery and energy accounting
│   ├── fedavg.py          # FL aggregation and local training
│   └── metrics.py         # Logging and metrics collection
└── data/                  # Dataset storage (CIFAR-10, EMNIST, LEAF)

## Setup & Requirements
- Python 3.8+
- Dependencies:
  pip install torch torchvision networkx numpy pandas pyyaml

## How to Run

1. Basic Execution
   To run the simulation with the default disturbance configuration:
   python scripts/run_baseline.py --config configs/disturbances.yaml

2. Experiment Modes (Comparison)
   To compare the Naive baseline vs. the Dynamic recovery strategy, modify the `routing_mode` in `configs/disturbances.yaml`:

   a. Naive Mode (Baseline):
      - Uses static shortest-path routing (BFS).
      - Fails immediately if the path contains offline nodes or down links.
      - Uses random client sampling.
      Set: routing_mode: "naive"

   b. Dynamic Mode (Proposed Solution):
      - Uses cost-aware routing (Dijkstra) on the active graph to bypass failures.
      - Enables "Smart Sampling" to prioritize high-battery clients.
      Set: routing_mode: "dynamic"

## Outputs
Results are saved to the `runs/` directory (timestamped folders).
- eval_metrics.csv: Contains the Test Accuracy per round.
- Console Output: Displays real-time progress and accuracy.