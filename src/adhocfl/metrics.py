import csv, os, json
from typing import Dict, Any, List

class MetricsLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.train_csv = os.path.join(self.out_dir, "train_metrics.csv")
        self.eval_csv = os.path.join(self.out_dir, "eval_metrics.csv")
        with open(self.train_csv, "w", newline="") as f:
            csv.writer(f).writerow(["round", "selected_clients", "bytes_sent", "bytes_received", "energy_used"])
        with open(self.eval_csv, "w", newline="") as f:
            csv.writer(f).writerow(["round", "accuracy"])

    def log_round(self, rnd: int, selected_clients: int, bytes_sent: int, bytes_received: int, energy_used: float):
        with open(self.train_csv, "a", newline="") as f:
            csv.writer(f).writerow([rnd, selected_clients, bytes_sent, bytes_received, energy_used])

    def log_eval(self, rnd: int, accuracy: float):
        with open(self.eval_csv, "a", newline="") as f:
            csv.writer(f).writerow([rnd, accuracy])

    def dump_config(self, cfg: Dict[str, Any]):
        with open(os.path.join(self.out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
