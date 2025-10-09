from config import BatteryConfig, NetworkConfig, Message, MessageType
import logging
import time


class Device:
    """Represents a device/node in the network."""

    def __init__(self, device_id: int, battery_config: BatteryConfig):
        self.id = device_id
        self.battery_config = battery_config
        self.energy = battery_config.initial_energy
        self.is_online = True
        self.neighbors = []

        self.message_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.total_energy_consumed = 0
        self.offline_count = 0

        self.logger = logging.getLogger(f"Device_{device_id}")

    
    def update_energy(self, cost: float):
        """update battery level"""
        self.energy -= cost
        self.total_energy_consumed += cost
        if self.energy <= self.battery_config.min_energy_threshold:
            if self.is_online:
                self.offline_count += 1
            self.is_online = False
            self.logger.info(f"Device {self.id} is offline due to low energy")
    
    def recharge(self):
        """recharge battery"""
        old_online = self.is_online
        self.energy = min(
            self.energy + self.battery_config.recharge_rate,
            self.battery_config.initial_energy
        )

        if self.energy >= self.battery_config.min_energy_threshold:
            self.is_online = True
            if not old_online:
                self.logger.info(f"Device {self.id} is back online")
    
    def send_message(self, to_id: int, msg_type: MessageType, payload, size_bytes: int = 1024):
        """send message to another device"""
        message = Message(
            msg_type=msg_type,
            from_id=self.id,
            to_id=to_id,
            payload=payload,
            timestamp=time.time(),
            size_bytes=size_bytes
        )
        self.messages_sent += 1
        self.bytes_sent += size_bytes
        self.update_energy(size_bytes * self.battery_config.send_cost_per_mb)
        self.logger.info(f"Device {self.id} sent message to {to_id} with type {msg_type}")
        return message
    
    def receive_message(self, message: Message):
        """receive message from another device"""
        self.messages_received += 1
        self.bytes_received += message.size_bytes
        self.update_energy(message.size_bytes * self.battery_config.receive_cost_per_mb)
        self.logger.info(f"Device {self.id} received message from {message.from_id} with type {message.msg_type}")
        return message
    
    def idle(self):
        """idle mode"""
        self.update_energy(self.battery_config.idle_cost_per_round)
        self.logger.info(f"Device {self.id} is in idle mode")
        return self.energy
    
    def get_status(self):
        """Get current device status"""
        return {
            'id': self.id,
            'online': self.is_online,
            'energy': self.energy,
            'neighbors': len(self.neighbors),
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'total_energy_consumed': self.total_energy_consumed,
            'offline_count': self.offline_count
        }


