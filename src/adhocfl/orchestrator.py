import random, os, time
from typing import List
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
import json

from .config import Config, NetworkConfig, BatteryConfig, TrainingConfig, DatasetConfig, RunConfig
from .network import SimNetwork
from .models.cnn import SimpleCNN
from .models.cnn_plus import CNNPlus
from .data.femnist import load_leaf_clients, FEMNISTClientDataset
from .fedavg import fedavg, train_one_client, evaluate, weighted_fedavg
from .metrics import MetricsLogger
from .netstats import compute_metrics, export_tables, draw_graph

from .data.cifar10 import load_cifar10_clients
from .models.cnn_cifar import CIFAR10Small, CIFAR10LiteRes, ResNet20CIFAR
from .disturbances import DisturbanceManager


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
    dist = DisturbanceManager(cfg.disturbances, rng, net)
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
        "disturbances": cfg.disturbances.__dict__,
    })

    # Data
    if cfg.dataset.source == "leaf":
        assert cfg.dataset.leaf_root is not None, "Please set dataset.leaf_root to your LEAF FEMNIST preprocessed folder"
        train_clients, test_ds = build_clients_from_leaf(cfg.dataset.leaf_root)
        num_classes = 62
        global_model = SimpleCNN(num_classes=num_classes)
    elif cfg.dataset.source == "cifar10":
        # Use n_devices to define how many clients to create so mapping stays 1:1
        cifar_root = getattr(cfg.dataset, "cifar10_root", "./data")
        train_clients, test_ds = load_cifar10_clients(
            root=cfg.dataset.cifar10_root,
            n_clients=cfg.network.n_devices,
            seed=cfg.training.seed,
            split_mode="iid",     # change to "dirichlet" if you want non-iid later
            alpha=10.0
        )
        num_classes = 10
        global_model = ResNet20CIFAR(num_classes=num_classes)
    else:
        # Fallback: EMNIST "byclass" via torchvision (optional dev-only)
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.EMNIST(root="./data", split="byclass", train=True, download=True, transform=transform)
        test = datasets.EMNIST(root="./data", split="byclass", train=False, download=True, transform=transform)
        
        # Partition EMNIST into pseudo-clients
        per_client = len(train) // cfg.network.n_devices
        train_clients = {}
        
        if cfg.dataset.emnist_iid:
            # IID: Sequential split (same label distribution across clients)
            for i in range(cfg.network.n_devices):
                idxs = list(range(i*per_client, (i+1)*per_client))
                # Convert to numpy arrays and flatten images to (n_samples, 784)
                images_list = [train[idx][0].squeeze().numpy()*255 for idx in idxs]
                images = np.array([img.flatten() for img in images_list])  # Shape: (n_samples, 784)
                labels = np.array([int(train[idx][1]) for idx in idxs])
                from .data.femnist import FEMNISTClientDataset
                train_clients[str(i)] = FEMNISTClientDataset(images, labels)
        else:
            # Non-IID: Grouped by label (each client gets one or few specific classes)
            # Sort by label to group samples by class
            train_data = [(train[i][0].squeeze().numpy()*255, int(train[i][1])) for i in range(len(train))]
            train_data.sort(key=lambda x: x[1])  # Sort by label
            
            num_classes = 62
            classes_per_client = max(1, num_classes // cfg.network.n_devices)
            samples_per_client = len(train) // cfg.network.n_devices
            
            client_idx = 0
            samples_assigned = 0
            
            for i in range(cfg.network.n_devices):
                images_list = []
                labels_list = []
                
                # Assign samples for this client
                target_samples = min(samples_per_client, len(train_data) - samples_assigned)
                for _ in range(target_samples):
                    if samples_assigned < len(train_data):
                        img, label = train_data[samples_assigned]
                        images_list.append(img)
                        labels_list.append(label)
                        samples_assigned += 1
                
                if images_list:
                    images = np.array([img.flatten() for img in images_list])
                    labels = np.array(labels_list)
                    from .data.femnist import FEMNISTClientDataset
                    train_clients[str(i)] = FEMNISTClientDataset(images, labels)
        
        test_ds = test
        num_classes = 62
        global_model = SimpleCNN(num_classes=num_classes)

    # FedAvg rounds
    for rnd in range(1, cfg.training.rounds+1):
        # 1) Apply disturbances BEFORE sampling if enabled
        if cfg.disturbances.enabled and cfg.disturbances.apply_before_sampling:
            dist.step(rnd)

        # 2) Sample online clients (this respects ephemeral offline state)
        selected_ids = net.sample_active_clients(cfg.training.clients_per_round)

        # Round accumulators
        delivered_ids: List[int] = []
        dropped = {"offline": 0, "low_battery": 0, "partition": 0, "packet_loss": 0, "timeout": 0}
        uplink_bytes = 0
        downlink_bytes = 0
        uplink_time_s = 0.0
        downlink_time_s = 0.0

        client_updates = []
        client_weights = []

        # 3) Local train (unchanged) -> then route-aware delivery gate
        for cid in selected_ids:
            # Map device -> client dataset shard
            client_key = list(train_clients.keys())[cid % len(train_clients)]
            client_ds = train_clients[client_key]

            sd, n = train_one_client(
                global_model,
                client_ds,
                cfg.training.local_epochs,      # keep your field name
                cfg.training.batch_size,
                cfg.training.lr,
                device
            )

            # Size of model update in bytes (float32-aware)
            bytes_up = sum(t.numel() * t.element_size() for t in sd.values())

            # Decide if this update actually reaches the server this round
            if cfg.disturbances.enabled:
                will, path, t_s, reason = dist.plan_uplink(client_id=cid, payload_bytes=bytes_up)
            else:
                will, path, t_s, reason = True, [cid], 0.0, "ok"

            if not will:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue

            # Count only delivered uplinks (simple, consistent)
            uplink_bytes += bytes_up
            uplink_time_s += t_s
            client_updates.append((sd, n))
            client_weights.append(n)
            delivered_ids.append(cid)

        # 4) Aggregate delivered updates (unchanged logic)
        if client_updates:
            new_state = weighted_fedavg(client_updates, skip_bn=True)  # your existing aggregator
            global_model.load_state_dict(new_state, strict=True)

            # Downlink: account broadcast of the new global to delivered clients
            bytes_down = sum(t.numel() * t.element_size() for t in global_model.state_dict().values())
            downlink_bytes += bytes_down * len(delivered_ids)

            if cfg.disturbances.enabled:
                for cid in delivered_ids:
                    path, t_s = net.shortest_gateway_path_and_time(
                        source=cid, gateways=None, bytes_to_send=bytes_down
                    )
                    if path is not None:
                        downlink_time_s += t_s

        # 5) Background device behavior (unchanged)
        for d in net.devices.values():
            d.idle()
            d.recharge()

        # 6) Log round metrics (uses the expanded header from step 4)
        metrics.log_train(
            round=rnd,
            selected_clients=len(selected_ids),
            delivered_clients=len(delivered_ids),
            dropped_offline=dropped["offline"],
            dropped_low_battery=dropped["low_battery"],
            dropped_partition=dropped["partition"],
            dropped_packet_loss=dropped["packet_loss"],
            dropped_timeout=dropped["timeout"],
            bytes_sent=uplink_bytes,
            bytes_received=downlink_bytes,
            uplink_time_s=uplink_time_s,
            downlink_time_s=downlink_time_s,
        )

        # 7) Evaluate as before
        acc = evaluate(global_model, test_ds, cfg.training.batch_size, device)
        metrics.log_eval(round=rnd, accuracy=acc)

    return out_dir
