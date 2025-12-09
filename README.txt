COST-AWARE DYNAMIC REROUTING FOR RESILIENT FEDERATED LEARNING
================================================================

1. PROJECT OVERVIEW
----------------------------------------------------------------
This software simulates a Federated Learning (FL) system operating over a volatile ad-hoc edge network. Unlike standard FL simulations that assume reliable communication, this project models physical layer disturbances—including node dropouts, link failures, bandwidth throttling, and battery depletion.

The core objective is to evaluate a "Cost-Aware Dynamic Rerouting" algorithm against a standard "Naive (Static)" baseline. The dynamic approach utilizes graph traversal algorithms on the active network topology to recover from failures and optimize delivery time, while "Smart Sampling" prioritizes energy-rich and well-connected nodes to prevent mid-round attrition.

2. DEPENDENCIES & INSTALLATION
----------------------------------------------------------------
This project is written in Python 3.8+. No compilation is required.

Required Libraries:
- torch / torchvision (Deep Learning backend)
- networkx (Graph data structures and algorithms)
- numpy / pandas (Data manipulation)
- pyyaml (Configuration management)

Installation Command:
pip install torch torchvision networkx numpy pandas pyyaml

3. USAGE INSTRUCTIONS
----------------------------------------------------------------
The simulation is controlled via a main entry script and YAML configuration files.

Basic Execution:
To run the simulation with the default disturbance configuration:
python scripts/run_baseline.py --config configs/disturbances.yaml

4. EXPERIMENT MODES
----------------------------------------------------------------
To run the comparative study, modify the "disturbances.yaml" file to switch between the two experimental modes:

MODE A: NAIVE BASELINE (STATIC)
This mode represents a standard implementation that does not account for network state changes during a round.
- Routing: Static Breadth-First Search (BFS) on the initial topology.
- Behavior: Fails immediately if the pre-calculated path contains an offline node or failed link.
- Sampling: Random client selection.
- Configuration Setting:
  disturbances:
    routing_mode: "naive"

MODE B: DYNAMIC RECOVERY (PROPOSED)
This mode employs the proposed resilience mechanisms.
- Routing: Dynamic Dijkstra's algorithm on the active topology (excluding failed components).
- Cost Function: Paths are weighted by transmission time (Latency + Size/Bandwidth).
- Sampling: "Smart Sampling" sorts eligible clients primarily by Remaining Energy (descending) and secondarily by Node Degree (descending).
- Configuration Setting:
  disturbances:
    routing_mode: "dynamic"

MODE C: NORMAL (BASELINE)
- This mode represents the baseline federated learning framework without any introduced disturbances.
- Configuration:
   disturbances
      enabled: false

5. CONFIGURATION PARAMETERS
----------------------------------------------------------------
The behavior of the simulation is defined in "configs/disturbances.yaml". Below are brief explanations for every parameter in order.

[network]
- topology: Graph generation algorithm (e.g., "barabasi", "erdos_renyi").
- n_devices: Total number of nodes (devices) in the simulation graph.
- er_probability: Probability of edge creation (used only if topology is "erdos_renyi").
- ensure_connected: If true, adds edges to bridge disconnected components after generation.
- latency_ms: Range [min, max] for assigning random latency to links.
- bandwidth_mbps: Range [min, max] for assigning random bandwidth to links.
- graph_path: Optional path to a CSV file to load a fixed topology (overrides generation).

[battery]
- initial_energy: Starting battery level for all devices (e.g., 100.0).
- send_cost_per_mb: Energy units consumed per megabyte sent.
- recv_cost_per_mb: Energy units consumed per megabyte received.
- idle_cost_per_round: Energy units consumed per round just for staying online.
- recharge_rate: Energy units regained per round (simulating harvesting/charging).
- min_energy_threshold: Battery level below which a device forces itself offline.

[training]
- rounds: Total number of global Federated Learning rounds to run.
- clients_per_round: Number of clients selected to train in each round.
- local_epochs: Number of training passes over local data per client per round.
- batch_size: Size of data batches used during local training.
- lr: Learning rate for the local optimizer (SGD).
- seed: Random seed for reproducibility (affects graph, sampling, and initialization).
- device: Hardware to use for training ("cpu" or "cuda").

[dataset]
- source: Dataset name (e.g., "emnist", "cifar10").
- non_iid: If true, data is distributed unevenly among clients.
- dirichlet_alpha: Concentration parameter for non-IID splits (lower = more skewed).

[run]
- out_dir: Base directory where simulation results/logs are saved.
- run_name: Label appended to the timestamped output folder.
- notes: Description string saved in the config dump for reference.
- export_network: If true, saves graph visualizations and CSVs (disable for large graphs).

[disturbances]
- enabled: Master switch (true/false) to turn all faults on or off.
- apply_before_sampling: If true, faults are injected before clients are selected.
- device_dropout_prob: Probability that a device goes offline temporarily this round.
- link_dropout_prob: Probability that a link is severed temporarily this round.
- throttle_bandwidth_factor: Range [min, max] multiplier to degrade link bandwidth.
- latency_jitter_ms: Maximum random noise (ms) added to link latency.
- packet_loss_prob: Probability that a completed uplink is dropped at the last moment.
- round_comm_budget_s: Max time (seconds) allowed for transmission before timeout.
- battery_floor: Soft limit; orchestrator skips devices below this energy level.
- gateways: Explicit list of gateway IDs (null = auto-pick top-degree nodes).
- max_gateway_count: Number of gateways to auto-select if 'gateways' is null.
- routing_mode: "naive" (static/random) or "dynamic" (cost-aware/smart-sampling).
- reset_each_round: If true, faults are cleared and re-rolled every round (stateless).

6. OUTPUTS
----------------------------------------------------------------
The simulation can take 1-2 hours to finish running. Results are generated in the "runs/<timestamp>_<name>/" directory:

- eval_metrics.csv: A simple CSV file containing the "round" number and global test "accuracy" (0.0-1.0).
- Console Output: Displays real-time client progress per round.