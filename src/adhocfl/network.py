import random
from typing import Dict, Tuple
import networkx as nx

from .config import NetworkConfig, BatteryConfig
from .device import Device

class SimNetwork:
    def __init__(self, net_cfg: NetworkConfig, bat_cfg: BatteryConfig, rng: random.Random):
        self.cfg = net_cfg
        self.battery = bat_cfg
        self.rng = rng
        self.graph = self._build_graph()
        self.devices = {i: Device(i, self.battery) for i in range(self.cfg.n_devices)}
        for u, v in self.graph.edges():
            self.devices[u].neighbors.add(v)
            self.devices[v].neighbors.add(u)

    def _build_graph(self):
        n = self.cfg.n_devices
        if self.cfg.topology == "erdos_renyi":
            G = nx.erdos_renyi_graph(n, self.cfg.er_probability, seed=self.rng.randrange(1<<30))
        elif self.cfg.topology == "barabasi":
            G = nx.barabasi_albert_graph(n, self.cfg.ba_m, seed=self.rng.randrange(1<<30))
        elif self.cfg.topology == "watts_strogatz":
            G = nx.watts_strogatz_graph(n, self.cfg.ws_k, self.cfg.ws_p, seed=self.rng.randrange(1<<30))
        elif self.cfg.topology == "ring":
            G = nx.cycle_graph(n)
        elif self.cfg.topology == "star":
            G = nx.star_graph(n-1)
        else:
            raise ValueError(f"Unknown topology {self.cfg.topology}")
        if self.cfg.ensure_connected and not nx.is_connected(G):
            # connect components by adding edges between them (simple heuristic)
            comps = list(nx.connected_components(G))
            for i in range(len(comps)-1):
                u = next(iter(comps[i]))
                v = next(iter(comps[i+1]))
                G.add_edge(u, v)
        # Assign link attributes
        for u, v in G.edges():
            G.edges[u, v]["latency_ms"] = self.rng.uniform(*self.cfg.latency_ms)
            G.edges[u, v]["bandwidth_mbps"] = self.rng.uniform(*self.cfg.bandwidth_mbps)
        return G

    def sample_active_clients(self, k: int):
        candidates = [d for d in self.devices.values() if d.online]
        if len(candidates) <= k:
            return [d.id for d in candidates]
        return [d.id for d in self.rng.sample(candidates, k)]
