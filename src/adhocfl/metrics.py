import csv, os, json
from typing import Dict, Any, List

class MetricsLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # Paths
        self.train_csv = os.path.join(self.out_dir, "train_metrics.csv")
        self.eval_csv  = os.path.join(self.out_dir, "eval_metrics.csv")

        # for compatibility with earlier code that referred to 'train_path'
        self.train_path = self.train_csv

        # Headers
        self.train_header = [
            "round",
            "selected_clients",
            "delivered_clients",
            "dropped_offline",
            "dropped_low_battery",
            "dropped_partition",
            "dropped_packet_loss",
            "dropped_timeout",
            "bytes_sent",
            "bytes_received",
            "uplink_time_s",
            "downlink_time_s",
        ]
        self.eval_header = ["round", "accuracy"]

        # Do NOT prewrite any header rows here; _write_csv handles that per file.

    # --- helpers ---
    def _write_csv(self, path: str, row: Dict[str, Any], header: List[str]) -> None:
        new_file = (not os.path.exists(path)) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if new_file:
                w.writeheader()
            # Ensure only known keys are written (extra keys ignored)
            safe_row = {h: row.get(h, "") for h in header}
            w.writerow(safe_row)

    # --- new disturbance-aware logging (used by orchestrator.run) ---
    def log_train(self, round: int, **kwargs):
        row = {h: kwargs.get(h, 0) for h in self.train_header}
        row["round"] = round
        self._write_csv(self.train_csv, row, header=self.train_header)

    # --- backward-compatible minimal logger (older code paths) ---
    def log_round(self, rnd: int, selected_clients: int, bytes_sent: int, bytes_received: int, energy_used: float):
        # Map old fields into the new header
        row = {
            "round": rnd,
            "selected_clients": selected_clients,
            "delivered_clients": selected_clients,  # best-effort default
            "dropped_offline": 0,
            "dropped_low_battery": 0,
            "dropped_partition": 0,
            "dropped_packet_loss": 0,
            "dropped_timeout": 0,
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "uplink_time_s": 0.0,
            "downlink_time_s": 0.0,
        }
        self._write_csv(self.train_csv, row, header=self.train_header)

    def log_eval(self, rnd: int = None, accuracy: float = None, **kwargs):
        # accept either 'rnd' or 'round' for robustness
        if rnd is None:
            rnd = kwargs.get("round")
        if rnd is None:
            raise ValueError("log_eval requires 'rnd' or 'round'")
        if accuracy is None:
            accuracy = kwargs.get("accuracy", 0.0)
        self._write_csv(self.eval_csv, {"round": rnd, "accuracy": accuracy}, header=self.eval_header)


    def dump_config(self, cfg: Dict[str, Any]):
        with open(os.path.join(self.out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
