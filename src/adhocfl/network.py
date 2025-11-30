import random
from typing import Dict, Tuple, List, Set, Optional
import networkx as nx

from .config import NetworkConfig, BatteryConfig
from .device import Device


class SimNetwork:
    def __init__(self, net_cfg: NetworkConfig, bat_cfg: BatteryConfig, rng: random.Random):
        self.cfg = net_cfg
        self.battery = bat_cfg
        self.rng = rng

        # Base (static) graph with link attributes
        self.graph = self._build_graph()

        # Device objects mapped by id
        self.devices: Dict[int, Device] = {i: Device(i, self.battery) for i in range(self.cfg.n_devices)}
        for u, v in self.graph.edges():
            self.devices[u].neighbors.add(v)
            self.devices[v].neighbors.add(u)

        # === NEW === Ephemeral disturbance state (re-applied each round)
        self._offline_devices: Set[int] = set()
        self._down_edges: Set[Tuple[int, int]] = set()     # edges removed this round (undirected, store sorted)
        self._bw_override: Dict[Tuple[int, int], float] = {}  # (u,v) -> Mbps
        self._lat_override: Dict[Tuple[int, int], float] = {} # (u,v) -> ms

        # === NEW === Gateways (ids) for server routing
        self._gateways: List[int] = []

    def _build_graph(self):
        if self.cfg.graph_path:
            import pandas as pd

            # Load adjacency matrix exported from a previous run.
            # Typically produced by networkx.to_pandas_adjacency(G).
            A = pd.read_csv(self.cfg.graph_path, index_col=0)

            # Many writers save index as ints and columns as strings (or vice versa).
            # Normalize both to ints when possible so they align.
            try:
                A.index = A.index.astype(int)
            except ValueError:
                # If this fails, we just leave them as-is.
                pass

            try:
                A.columns = A.columns.astype(int)
            except ValueError:
                pass

            # Now they must match for from_pandas_adjacency to work.
            if not A.index.equals(A.columns):
                raise ValueError(
                    "Adjacency CSV invalid: columns must match indices for from_pandas_adjacency.\n"
                    f"First few index:   {list(A.index)[:5]}\n"
                    f"First few columns: {list(A.columns)[:5]}"
                )

            # Reconstruct the graph topology exactly from the adjacency matrix.
            G = nx.from_pandas_adjacency(A)

            # Sanity check: ensure expected number of devices.
            if self.cfg.n_devices is not None and G.number_of_nodes() != self.cfg.n_devices:
                raise ValueError(
                    f"Loaded graph has {G.number_of_nodes()} nodes but n_devices={self.cfg.n_devices}"
                )

            # Assign link attributes EXACTLY as in your original code.
            for u, v in G.edges():
                G.edges[u, v]["latency_ms"] = self.rng.uniform(*self.cfg.latency_ms)
                G.edges[u, v]["bandwidth_mbps"] = self.rng.uniform(*self.cfg.bandwidth_mbps)

            return G
        
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

    # === NEW === Disturbance API (called by DisturbanceManager)
    def set_gateways(self, gateways: List[int]) -> None:
        """Define which nodes act as gateways to the server."""
        self._gateways = list(gateways or [])

    def apply_ephemeral_state(
        self,
        offline_devices: Set[int],
        down_edges: Set[Tuple[int, int]],
        bw_override: Dict[Tuple[int, int], float],
        lat_override: Dict[Tuple[int, int], float],
    ) -> None:
        """Apply per-round failures/overrides."""
        # Normalize undirected edge tuples to sorted pairs
        self._offline_devices = set(offline_devices or set())
        self._down_edges = {tuple(sorted(e)) for e in (down_edges or set())}
        self._bw_override = dict(bw_override or {})
        self._lat_override = dict(lat_override or {})

    # === NEW === Helpers used by sampling/routing
    def is_device_online(self, node_id: int) -> bool:
        """Online if device.online is True AND it is not forced-offline this round."""
        d = self.devices.get(node_id)
        online_flag = True if (d is None) else bool(d.online)
        return (node_id not in self._offline_devices) and online_flag

    def active_graph(self) -> nx.Graph:
        """
        Return a working copy of the graph for this round:
        - removes down edges
        - removes offline nodes
        """
        H = self.graph.copy()
        if self._down_edges:
            H.remove_edges_from(self._down_edges)
        # Optionally remove nodes that are offline this round
        to_drop = [n for n in H.nodes if not self.is_device_online(n)]
        if to_drop:
            H.remove_nodes_from(to_drop)
        return H

    def _edge_cost_seconds(self, u: int, v: int, payload_bytes: int) -> float:
        """
        Time to ship payload across one hop (seconds) = latency + serialization time.
        """
        # Effective bandwidth (Mbps)
        if (u, v) in self._bw_override:
            bw = self._bw_override[(u, v)]
        elif (v, u) in self._bw_override:
            bw = self._bw_override[(v, u)]
        else:
            bw = float(self.graph[u][v].get("bandwidth_mbps", 10.0))
        bw = max(1e-6, bw)

        # Effective latency (ms)
        if (u, v) in self._lat_override:
            lat_ms = self._lat_override[(u, v)]
        elif (v, u) in self._lat_override:
            lat_ms = self._lat_override[(v, u)]
        else:
            lat_ms = float(self.graph[u][v].get("latency_ms", 10.0))

        # Convert
        serialize_s = payload_bytes / (bw * 125_000.0)  # 1 Mbps = 125,000 bytes/s
        return (lat_ms / 1000.0) + serialize_s

    def shortest_gateway_path_naive(
        self,
        source: int,
        gateways: Optional[List[int]],
        bytes_to_send: int,
    ) -> Tuple[Optional[List[int]], float]:
        """
        Naive BFS-based shortest path (by hop count) from source to any gateway.
        Uses original graph without removing edges/nodes, but checks if path contains
        offline nodes or down edges. Returns (path, num_hops). If unreachable, returns (None, 0.0).
        """
        # Use original graph (don't remove edges/nodes)
        G = self.graph
        if source not in G:
            return (None, 0.0)

        targets = list(gateways or self._gateways)
        targets = [g for g in targets if g in G]
        if not targets:
            return (None, 0.0)

        # BFS to find shortest valid path by hop count
        # Try all gateways and find the shortest valid path
        best_path: Optional[List[int]] = None
        best_hops = float("inf")
        
        for g in targets:
            try:
                # Use BFS (unweighted shortest path)
                path = nx.shortest_path(G, source=source, target=g)
                hops = len(path) - 1
                
                # Only consider if this is potentially better (shorter or equal)
                if hops <= best_hops:
                    # Check if path contains any offline nodes or down edges
                    path_valid = True
                    for i in range(len(path)):
                        node = path[i]
                        # Check if node is offline
                        if not self.is_device_online(node):
                            path_valid = False
                            break
                        # Check if edge is down (for edges in path)
                        if i < len(path) - 1:
                            u, v = path[i], path[i+1]
                            edge_tuple = tuple(sorted((u, v)))
                            if edge_tuple in self._down_edges:
                                path_valid = False
                                break
                    
                    # If path is valid and shorter (or equal but first found), use it
                    if path_valid and hops < best_hops:
                        best_hops = hops
                        best_path = path
            except nx.NetworkXNoPath:
                continue

        if best_path is None:
            return (None, 0.0)
        # Return path and number of hops (as "time" for compatibility)
        return (best_path, float(best_hops))

    def shortest_gateway_path_and_time(
        self,
        source: int,
        gateways: Optional[List[int]],
        bytes_to_send: int,
    ) -> Tuple[Optional[List[int]], float]:
        """
        Fastest path (seconds) from source to any gateway using the active graph.
        Returns (path, total_time_s). If unreachable, returns (None, 0.0).
        """
        H = self.active_graph()
        if source not in H:
            return (None, 0.0)

        targets = list(gateways or self._gateways)
        targets = [g for g in targets if g in H]
        if not targets:
            return (None, 0.0)

        # Build per-edge weights for this payload
        Hw = H.copy()
        for u, v in Hw.edges():
            Hw.edges[u, v]["weight"] = self._edge_cost_seconds(u, v, bytes_to_send)

        best_path: Optional[List[int]] = None
        best_cost = float("inf")
        for g in targets:
            try:
                cost, path = nx.single_source_dijkstra(Hw, source=source, target=g, weight="weight")
                if cost < best_cost:
                    best_cost, best_path = cost, path
            except nx.NetworkXNoPath:
                continue

        if best_path is None:
            return (None, 0.0)
        return (best_path, best_cost)

    # === UPDATED === Use disturbance-aware online check when sampling
    def sample_active_clients(self, k: int) -> List[int]:
        candidates = [d.id for d in self.devices.values() if self.is_device_online(d.id)]
        if len(candidates) <= k:
            return candidates
        return self.rng.sample(candidates, k)
