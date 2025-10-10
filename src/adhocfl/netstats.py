import os, json, csv
from typing import Dict, Any
import networkx as nx

def _largest_cc(G: nx.Graph):
    if nx.is_connected(G): return G
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(comps[0]).copy()

def compute_metrics(G: nx.Graph) -> Dict[str, Any]:
    H = _largest_cc(G)
    n, m = G.number_of_nodes(), G.number_of_edges()
    degs = [d for _, d in G.degree()]
    return {
        "nodes": n,
        "edges": m,
        "avg_degree": sum(degs)/max(n,1),
        "density": nx.density(G),
        "connected": nx.is_connected(G),
        "num_components": nx.number_connected_components(G),
        "lcc_size": H.number_of_nodes(),
        "avg_path_length_lcc": float(nx.average_shortest_path_length(H)) if H.number_of_nodes()>1 else 0.0,
        "diameter_lcc": int(nx.diameter(H)) if H.number_of_nodes()>1 else 0,
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "assortativity_degree": nx.degree_assortativity_coefficient(G) if m>0 else 0.0,
    }

def export_tables(G: nx.Graph, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # nodes.csv
    with open(os.path.join(out_dir, "nodes.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id","degree"])
        for n, d in G.degree(): w.writerow([n, d])
    # edges.csv (Gephi-friendly)
    with open(os.path.join(out_dir, "edges.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source","target","latency_ms","bandwidth_mbps"])
        for u, v, data in G.edges(data=True):
            w.writerow([u, v, data.get("latency_ms",""), data.get("bandwidth_mbps","")])
    # adjacency.csv
    import numpy as np, pandas as pd
    A = nx.to_numpy_array(G, dtype=int)
    pd.DataFrame(A, index=list(G.nodes()), columns=list(G.nodes()))\
      .to_csv(os.path.join(out_dir, "adjacency.csv"))

def draw_graph(G: nx.Graph, out_dir: str, title="network"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, seed=42)  # spring layout; avoid if n is very large
    nx.draw_networkx_nodes(G, pos, node_size=30)
    nx.draw_networkx_edges(G, pos, alpha=0.35)
    plt.title(title); plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "graph.png"), dpi=200); plt.close()
