import argparse, yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from adhocfl.config import Config, NetworkConfig, BatteryConfig, TrainingConfig, DatasetConfig, RunConfig
from adhocfl.orchestrator import run

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    cfg = Config(
        network=NetworkConfig(**y.get("network", {})),
        battery=BatteryConfig(**y.get("battery", {})),
        training=TrainingConfig(**y.get("training", {})),
        dataset=DatasetConfig(**y.get("dataset", {})),
        run=RunConfig(**y.get("run", {})),
    )
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", type=str, default="configs/baseline.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = run(cfg)
    print(f"Finished. Metrics and artifacts in: {out_dir}")

if __name__ == "__main__":
    main()
