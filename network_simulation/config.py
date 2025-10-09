import logging
from dataclasses import dataclass
from enum import Enum
from multiprocessing.reduction import ACKNOWLEDGE
from typing import Any, Tuple


logging = logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class NetworkConfig:
    topology_type: str = 'erdos_renyi'
    n_devices: int = 50
    er_probability: float = 0.1
    latency_range: Tuple[float, float] = (10.0, 100.0)
    bandwidth_range: Tuple[float, float] = (1.0, 10.0)
    device_dropout_prob: float = 0.05
    link_dropout_prob: float = 0.02
    ensure_connected: bool = True

@dataclass
class BatteryConfig:
    initial_energy: float = 100.0
    send_cost_per_mb: float = 0.1
    receive_cost_per_mb: float = 0.05
    idle_cost_per_round: float = 0.01
    recharge_rate: float = 1.0
    min_energy_threshold: float = 5.0

class MessageType(Enum):
    DATA = "data"
    ACKNOWLEDGEMENT = "acknowledgement"
    HEARTBEAT = "heartbeat"
    BROADCAST = "broadcast"

@dataclass
class Message:
    msg_type: MessageType
    from_id: int
    to_id: int
    payload: Any
    timestamp: float
    size_bytes: int = 1024
    message_id: str = ""

    def __post_init__(self):
        if not self.message_id:
            self.message_id = f"{self.from_id}_{self.to_id}_{self.timestamp}"
