# Source Code Documentation

## 1. System Architecture
The software is modularized to separate the physical network simulation from the high-level Federated Learning orchestration.

### Key Files & Responsibilities
* **`src/adhocfl/orchestrator.py`**: The central controller. It manages the FL training loop, triggers disturbance injection, handles client sampling strategies, and aggregates model updates. It strictly enforces energy accounting, deducting battery levels for every Uplink (send) and Downlink (receive) operation.
* **`src/adhocfl/network.py`**: Manages the graph data structure. It handles graph generation, state management (online/offline nodes), and routing algorithms.
* **`src/adhocfl/disturbances.py`**: The chaos engine. It probabilistically modifies the network state at the start of each round to simulate failures.
* **`src/adhocfl/device.py`**: A class representing a physical device. It tracks battery levels (`energy`), accounting for costs incurred during transmission and idle periods.
* **`src/adhocfl/fedavg.py`**: Implements the standard Federated Averaging algorithm and the local PyTorch training loop.

## 2. Algorithms

### A. Routing Strategies
The project compares two distinct routing algorithms implemented in `network.py`:

1.  **Naive Routing (Baseline)**
    * **Function**: `shortest_gateway_path_naive`
    * **Logic**: Uses Breadth-First Search (BFS) on the *static* graph to find the minimum hop path to a gateway.
    * **Failure Model**: It does not check for active failures during path discovery. If the computed path contains a node that is currently offline or a link that is down, the transmission is marked as failed (`dropped_partition`).

2.  **Cost-Aware Dynamic Routing (Proposed)**
    * **Function**: `shortest_gateway_path_and_time`
    * **Logic**: Uses Dijkstra’s algorithm on the *active* subgraph (filtering out offline components).
    * **Cost Metric**: Edges are weighted by estimated transmission time:
        $$Cost(u,v) = \text{Latency}_{uv} + \frac{\text{PayloadSize}}{\text{Bandwidth}_{uv}}$$
    * **Resilience**: It inherently bypasses failures by finding the optimal path through currently active links.

### B. Smart Sampling
Implemented in `network.py` (`sample_active_clients`) and triggered by the orchestrator when `routing_mode="dynamic"`.
* **Logic**: Instead of random selection, the algorithm:
    1.  Identifies all currently online candidates.
    2.  Sorts candidates primarily by **Remaining Energy** (descending) and secondarily by **Node Degree** (descending).
    3.  Selects the top-$K$ devices.
* **Purpose**: This prioritizes devices that are both energy-rich (less likely to die mid-round) and central to the network (better connectivity/hubs), reducing the risk of `dropped_low_battery` and `dropped_partition` events.

### C. Disturbance Injection
Implemented in `disturbances.py`. The `DisturbanceManager` applies a stateless disturbance model at the start of every round:
1.  **Node Dropout**: Each node is marked offline with probability $P_{\text{device}}$.
2.  **Link Failure**: Each edge is removed with probability $P_{\text{link}}$.
3.  **Throttling**: Active links are assigned a temporary bandwidth multiplier to simulate congestion.

## 3. Data Structures

### Network Graph (`SimNetwork.graph`)
* **Type**: `networkx.Graph`
* **Nodes**: Integers representing device IDs.
* **Edges**: Undirected links between devices.
* **Attributes**: Each edge stores `latency_ms` (float) and `bandwidth_mbps` (float).

### Device State (`Device` Class)
* **Attributes**:
    * `energy` (float): Current battery level (0-100%).
    * `online` (bool): Operational status.
* **Methods**:
    * `_spend(amount)`: Deducts energy. If energy < `min_energy_threshold`, sets `online = False`.

### Configuration (`Config` Dataclass)
* Defined in `config.py`.
* Uses nested dataclasses (`NetworkConfig`, `DisturbanceConfig`, etc.) to map directly to the YAML input structure, ensuring type safety and clarity.