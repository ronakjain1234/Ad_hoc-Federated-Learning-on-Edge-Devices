# Ad-Hoc FL Baseline (FedAvg on FEMNIST)

A minimal, modular baseline to simulate a network of edge devices, train FedAvg on FEMNIST (LEAF), and record metrics. Designed to be easily extended with dropouts, link failures, and battery logic.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# Set dataset.leaf_root in configs/baseline.yaml to your LEAF FEMNIST 'preprocess' output folder
python scripts/run_baseline.py --config configs/baseline.yaml
```
Outputs will be under `runs/<timestamp>_baseline_femnist/` with `train_metrics.csv`, `eval_metrics.csv`, and `config.json`.

## Structure
- `src/adhocfl/config.py` – dataclasses for all configs
- `src/adhocfl/device.py` – simple device + accounting
- `src/adhocfl/network.py` – topology builder (NetworkX)
- `src/adhocfl/models/cnn.py` – small CNN for 1x28x28
- `src/adhocfl/data/femnist.py` – loader for LEAF FEMNIST client splits
- `src/adhocfl/fedavg.py` – client train + FedAvg + eval
- `src/adhocfl/orchestrator.py` – end-to-end training loop, metrics
- `configs/baseline.yaml` – tune all knobs here
- `scripts/run_baseline.py` – CLI entry point

## Notes
- Baseline intentionally sets all dropout probabilities to 0 (clean run).
- Energy + bytes are accounted per-device so disturbances can later toggle via config.
- If you don't have LEAF ready, set `dataset.source: emnist` to use a torchvision fallback for local smoke tests.
