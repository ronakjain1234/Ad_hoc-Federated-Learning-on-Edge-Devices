from dataclasses import dataclass
from typing import Any, Dict
import time

from .config import BatteryConfig

@dataclass
class Message:
    msg_type: str
    sender: int
    receiver: int
    payload: Any
    timestamp: float
    size_bytes: int = 1024

class Device:
    """Represents a simulated edge device with a battery and simple comm accounting."""
    def __init__(self, device_id: int, battery: BatteryConfig):
        self.id = device_id
        self.battery = battery
        self.energy = battery.initial_energy
        self.online = True
        self.neighbors = set()
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.energy_used = 0.0
        self.offline_events = 0

    def _spend(self, amount: float):
        self.energy -= amount
        self.energy_used += amount
        if self.energy < self.battery.min_energy_threshold:
            if self.online:
                self.offline_events += 1
            self.online = False

    def recharge(self):
        self.energy = min(self.energy + self.battery.recharge_rate, self.battery.initial_energy)
        if self.energy >= self.battery.min_energy_threshold:
            self.online = True

    def idle(self):
        self._spend(self.battery.idle_cost_per_round)

    def send(self, to_id: int, payload: Any, size_bytes: int = 1024, msg_type: str = "model_update"):
        now = time.time()
        self.messages_sent += 1
        self.bytes_sent += size_bytes
        # cost: MB = bytes / (1024*1024)
        self._spend((size_bytes / (1024*1024)) * self.battery.send_cost_per_mb)
        return Message(msg_type=msg_type, sender=self.id, receiver=to_id, payload=payload, timestamp=now, size_bytes=size_bytes)

    def receive(self, msg: Message):
        self.messages_received += 1
        self.bytes_received += msg.size_bytes
        self._spend((msg.size_bytes / (1024*1024)) * self.battery.recv_cost_per_mb)

    def status(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "online": self.online,
            "energy": self.energy,
            "neighbors": len(self.neighbors),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "energy_used": self.energy_used,
            "offline_events": self.offline_events,
        }
