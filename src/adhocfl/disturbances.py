# disturbances.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import random
import networkx as nx

from .config import DisturbanceConfig
from .network import SimNetwork

class DisturbanceManager:
    """
    Stateless (per-round) disturbance injector. Mutates the SimNetwork's
    ephemeral state: which devices/edges are considered up, and per-edge cost overrides.
    """
    def __init__(self, cfg: DisturbanceConfig, rng: random.Random, net: SimNetwork):
        self.cfg = cfg
        self.rng = rng
        self.net = net

        # Ephemeral (recomputed each round if reset_each_round)
        self.offline_devices: Set[int] = set()
        self.down_edges: Set[Tuple[int,int]] = set()
        self.bw_override: Dict[Tuple[int,int], float] = {}    # Mbps
        self.lat_override: Dict[Tuple[int,int], float] = {}   # ms

        # Gateways (fixed for the run unless provided explicitly)
        if cfg.gateways is not None and len(cfg.gateways) > 0:
            self.gateways = list(cfg.gateways)
        else:
            # auto-pick top-degree nodes (robust defaults)
            deg = sorted(self.net.graph.degree, key=lambda x: x[1], reverse=True)
            k = min(cfg.max_gateway_count, len(deg))
            self.gateways = [node for node, _ in deg[:k]]
        self.net.set_gateways(self.gateways)

    def _reset_ephemeral(self):
        self.offline_devices.clear()
        self.down_edges.clear()
        self.bw_override.clear()
        self.lat_override.clear()

    def step(self, round_idx: int):
        if self.cfg.reset_each_round:
            self._reset_ephemeral()

        # 1) Device availability
        for node in self.net.graph.nodes:
            if self.rng.random() < self.cfg.device_dropout_prob:
                self.offline_devices.add(node)

        # 2) Link failures and throttling/jitter
        for u, v, data in self.net.graph.edges(data=True):
            # Random drop
            if self.rng.random() < self.cfg.link_dropout_prob:
                self.down_edges.add((u, v))
                self.down_edges.add((v, u))  # undirected model
                continue

            # Throttle bandwidth multiplicatively
            lo, hi = self.cfg.throttle_bandwidth_factor
            if hi < lo: lo, hi = hi, lo
            factor = self.rng.uniform(lo, hi)
            base_bw = float(data.get("bandwidth_mbps", 10.0))
            self.bw_override[(u, v)] = max(1e-6, base_bw * factor)
            self.bw_override[(v, u)] = self.bw_override[(u, v)]

            # Add latency jitter
            jitter = self.cfg.latency_jitter_ms
            if jitter > 0:
                base_lat = float(data.get("latency_ms", 10.0))
                jittered = max(0.0, base_lat + self.rng.uniform(-jitter, jitter))
                self.lat_override[(u, v)] = jittered
                self.lat_override[(v, u)] = jittered

        # Push ephemeral state into network for this round
        self.net.apply_ephemeral_state(
            offline_devices=self.offline_devices,
            down_edges=self.down_edges,
            bw_override=self.bw_override,
            lat_override=self.lat_override,
        )

    # Gateway-aware delivery check (used by orchestrator)
    def plan_uplink(self, client_id: int, payload_bytes: int) -> Tuple[bool, Optional[List[int]], float, str]:
        """
        Returns (will_deliver, path, comm_time_s, reason)
        - If client has no path to any gateway -> reason="partition"
        - If battery below floor -> reason="low_battery"
        - If packet lost -> reason="packet_loss"
        - If comm_time exceeds budget -> reason="timeout"
        - Else -> will_deliver=True with path and comm_time_s
        """
        # Battery floor check: orchestrator should pass live device objects, but we only
        # know the network here. Ask the network for battery if available, else skip.
        dev = self.net.devices.get(client_id)
        if dev is not None and dev.energy < self.cfg.battery_floor:
            return (False, None, 0.0, "low_battery")

        # Path to nearest gateway
        path, comm_time_s = self.net.shortest_gateway_path_and_time(
            source=client_id,
            gateways=self.gateways,
            bytes_to_send=payload_bytes,
        )
        if path is None:
            return (False, None, 0.0, "partition")

        # Packet loss
        if self.rng.random() < self.cfg.packet_loss_prob:
            return (False, path, comm_time_s, "packet_loss")

        # Straggler timeout
        if comm_time_s > self.cfg.round_comm_budget_s:
            return (False, path, comm_time_s, "timeout")

        return (True, path, comm_time_s, "ok")
