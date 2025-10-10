from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

@dataclass
class NetworkConfig:
    topology: str = "erdos_renyi"   # erdos_renyi | barabasi | watts_strogatz | ring | star
    n_devices: int = 50
    er_probability: float = 0.1
    ba_m: int = 2
    ws_k: int = 4
    ws_p: float = 0.05
    ensure_connected: bool = True
    latency_ms: Tuple[float, float] = (5.0, 50.0)
    bandwidth_mbps: Tuple[float, float] = (5.0, 20.0)
    device_dropout_prob: float = 0.0       # baseline: 0 (no abnormalities)
    link_dropout_prob: float = 0.0         # baseline: 0 (no abnormalities)

@dataclass
class BatteryConfig:
    initial_energy: float = 100.0
    send_cost_per_mb: float = 0.02
    recv_cost_per_mb: float = 0.01
    idle_cost_per_round: float = 0.005
    recharge_rate: float = 0.25
    min_energy_threshold: float = 1.0

@dataclass
class TrainingConfig:
    rounds: int = 5
    clients_per_round: int = 10
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.01
    seed: int = 42
    device: str = "cpu" # or "cuda"

@dataclass
class DatasetConfig:
    source: str = "leaf"  # leaf | emnist (fallback via torchvision)
    leaf_root: Optional[str] = None  # path to FEMNIST preprocessed data (contains train/test/all_data jsons)
    iid_fraction: float = 0.0        # 0 = fully non-iid (LEAF native splits), 1.0 = iid shuffle (optional future)

@dataclass
class RunConfig:
    out_dir: str = "runs"
    run_name: str = "baseline"
    notes: str = ""
    export_network: bool = True

@dataclass
class Config:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    run: RunConfig = field(default_factory=RunConfig)
