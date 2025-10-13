import random, os, time
from typing import List
import torch
import json

from .config import Config, NetworkConfig, BatteryConfig, TrainingConfig, DatasetConfig, RunConfig
from .network import SimNetwork
from .models.cnn import SimpleCNN
from .data.femnist import load_leaf_clients, FEMNISTClientDataset
from .fedavg import fedavg, train_one_client, evaluate
from .metrics import MetricsLogger
from .netstats import compute_metrics, export_tables, draw_graph


def build_clients_from_leaf(leaf_root: str):
    train_clients = load_leaf_clients(leaf_root, split="train")
    test_clients  = load_leaf_clients(leaf_root, split="test")

    # Pool all test shards safely by concatenation
    import numpy as np
    from .data.femnist import FEMNISTClientDataset

    imgs, labels = [], []
    for ds in test_clients.values():
        # ds.images: (n_i, 784), ds.labels: (n_i,)
        imgs.append(np.asarray(ds.images))
        labels.append(np.asarray(ds.labels))

    pooled_imgs   = np.vstack(imgs) if imgs else np.zeros((0, 784), dtype=np.uint8)
    pooled_labels = np.hstack(labels) if labels else np.zeros((0,), dtype=np.int64)
    pooled_test   = FEMNISTClientDataset(pooled_imgs, pooled_labels)

    return train_clients, pooled_test

def run(cfg: Config):
    rng = random.Random(cfg.training.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device == "cuda" else "cpu")

    net = SimNetwork(cfg.network, cfg.battery, rng)
    out_dir = os.path.join(cfg.run.out_dir, f"{int(time.time())}_{cfg.run.run_name}")
    metrics = MetricsLogger(out_dir)
    if cfg.run.export_network:
        gdir = os.path.join(out_dir, "network")
        os.makedirs(gdir, exist_ok=True)
        draw_graph(net.graph, gdir, title=f"{cfg.network.topology} (n={cfg.network.n_devices})")
        export_tables(net.graph, gdir)
        with open(os.path.join(gdir, "graph_metrics.json"), "w") as f:
            json.dump(compute_metrics(net.graph), f, indent=2)
    # Save config snapshot
    metrics.dump_config({
        "network": cfg.network.__dict__,
        "battery": cfg.battery.__dict__,
        "training": cfg.training.__dict__,
        "dataset": cfg.dataset.__dict__,
        "run": cfg.run.__dict__,
    })

    # Data
    if cfg.dataset.source == "leaf":
        assert cfg.dataset.leaf_root is not None, "Please set dataset.leaf_root to your LEAF FEMNIST preprocessed folder"
        train_clients, test_ds = build_clients_from_leaf(cfg.dataset.leaf_root)
        num_classes = 62
    else:
        # Fallback: EMNIST "byclass" via torchvision (optional dev-only)
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.EMNIST(root="./data", split="byclass", train=True, download=True, transform=transform)
        test = datasets.EMNIST(root="./data", split="byclass", train=False, download=True, transform=transform)
        # Partition EMNIST into pseudo-clients evenly
        per_client = len(train) // cfg.network.n_devices
        train_clients = {}
        for i in range(cfg.network.n_devices):
            idxs = list(range(i*per_client, (i+1)*per_client))
            images = [train[i][0].squeeze().numpy()*255 for i in idxs]
            labels = [int(train[i][1]) for i in idxs]
            from .data.femnist import FEMNISTClientDataset
            train_clients[str(i)] = FEMNISTClientDataset(images, labels)
        test_ds = test
        num_classes = 62

    global_model = SimpleCNN(num_classes=num_classes)
    # FedAvg rounds
    for rnd in range(1, cfg.training.rounds+1):
        selected_ids = net.sample_active_clients(cfg.training.clients_per_round)
        state_dicts = []
        bytes_sent = 0
        bytes_recv = 0
        energy_used = 0.0
        for cid in selected_ids:
            # Map from numeric id -> client key: use modulo for simplicity
            client_key = list(train_clients.keys())[cid % len(train_clients)]
            client_ds = train_clients[client_key]
            # Train locally
            sd = train_one_client(global_model, client_ds, cfg.training.local_epochs, cfg.training.batch_size, cfg.training.lr, device)
            state_dicts.append(sd)
            # Account a rough payload size: serialize tensors sizes
            size_bytes = sum(p.numel()*4 for p in sd.values())  # float32
            msg = net.devices[cid].send(to_id=-1, payload=None, size_bytes=size_bytes)  # to server
            bytes_sent += msg.size_bytes
            energy_used += net.devices[cid].energy_used

        # Aggregate
        new_state = fedavg(state_dicts)
        global_model.load_state_dict(new_state)

        # Broadcast back to selected clients (account only once)
        payload_size = sum(p.numel()*4 for p in global_model.state_dict().values())
        for cid in selected_ids:
            msg_back = net.devices[cid].send(to_id=cid, payload=None, size_bytes=payload_size, msg_type="global_model")
            bytes_recv += msg_back.size_bytes

        # Idle/recharge accounting for all devices (baseline, no dropouts)
        for d in net.devices.values():
            d.idle()
            d.recharge()

        metrics.log_round(rnd, len(selected_ids), bytes_sent, bytes_recv, energy_used)

        # Evaluate
        acc = evaluate(global_model, test_ds, cfg.training.batch_size, device)
        metrics.log_eval(rnd, acc)

    return out_dir
