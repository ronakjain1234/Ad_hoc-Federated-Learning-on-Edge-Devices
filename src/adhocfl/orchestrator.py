import random, os, time
from typing import List
import torch
torch.backends.cudnn.benchmark = True
import json

from .config import Config, NetworkConfig, BatteryConfig, TrainingConfig, DatasetConfig, RunConfig
from .network import SimNetwork
from .models.cnn import SimpleCNN
from .models.cnn_plus import CNNPlus
from .fedavg import fedavg, train_one_client, evaluate, weighted_fedavg
from .metrics import MetricsLogger
from .netstats import compute_metrics, export_tables, draw_graph

from .disturbances import DisturbanceManager
from .device import Message
import numpy as np
from torch.utils.data import Dataset

class EMNISTClientDataset(Dataset):
    """
    Simple per-client dataset wrapper for EMNIST.
    Expects 28x28 uint8 images and integer labels.
    """
    def __init__(self, images, labels):
        # images: list/array of (28,28) or (784,) uint8
        # labels: list/array of ints
        self.images = np.array(images, dtype=np.uint8)
        self.labels = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]

        # If flattened (784), reshape to (28, 28)
        if img.ndim == 1:
            img = img.reshape(28, 28)

        # Convert to float32 [0,1] and to tensor [1, 28, 28]
        img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        label = int(self.labels[idx])
        return img, label


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
        print("No longer supported: Use EMNIST")
    elif cfg.dataset.source == "cifar10":
        print("No longer supported: Use EMNIST")
    else:
        # Fallback: EMNIST "byclass" via torchvision (optional dev-only)
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.EMNIST(root="./data", split="byclass", train=True, download=True, transform=transform)
        test = datasets.EMNIST(root="./data", split="byclass", train=False, download=True, transform=transform)
        
        train_clients = {}
        num_classes = 62
        
        if cfg.dataset.non_iid:
            import numpy as np
            np.random.seed(cfg.training.seed)
            # Dirichlet non-IID over EMNIST "byclass"
            # - Each class's samples are distributed to clients according to
            #   a Dirichlet(alpha) draw.
            # - Each sample lives on exactly ONE client.
            # - alpha < 1 => more skewed; alpha > 1 => closer to IID.

            n_clients = cfg.network.n_devices
            # Make this configurable later if you want: cfg.dataset.dirichlet_alpha
            alpha = getattr(cfg.dataset, "dirichlet_alpha", cfg.dataset.dirichlet_alpha)

            # 1) Group data indices by class
            class_data = {c: [] for c in range(num_classes)}
            for idx in range(len(train)):
                _, label = train[idx]
                class_data[int(label)].append(idx)

            # 2) For each class, sample a Dirichlet distribution over clients
            #    and split that class's samples according to those proportions.
            client_indices = {i: [] for i in range(n_clients)}

            for class_id, idxs in class_data.items():
                if not idxs:
                    continue

                idxs_copy = list(idxs)
                rng.shuffle(idxs_copy)

                # Dirichlet over clients
                # shape: (n_clients,)
                probs = np.random.dirichlet(alpha * np.ones(n_clients))

                # Multinomial draw: how many samples of this class per client
                counts = np.random.multinomial(len(idxs_copy), probs)

                start = 0
                for client_id, count in enumerate(counts):
                    if count == 0:
                        continue
                    end = start + count
                    client_indices[client_id].extend(idxs_copy[start:end])
                    start = end

            # 3) Build per-client datasets
            for i in range(n_clients):
                idxs = client_indices[i]
                rng.shuffle(idxs)
                images = [train[idx][0].squeeze().numpy() * 255 for idx in idxs]
                labels = [int(train[idx][1]) for idx in idxs]
                train_clients[str(i)] = EMNISTClientDataset(images, labels)
        else:
            # IID: Partition EMNIST into pseudo-clients evenly
            per_client = len(train) // cfg.network.n_devices
            for i in range(cfg.network.n_devices):
                idxs = list(range(i*per_client, (i+1)*per_client))
                images = [train[idx][0].squeeze().numpy()*255 for idx in idxs]
                labels = [int(train[idx][1]) for idx in idxs]
                train_clients[str(i)] = EMNISTClientDataset(images, labels)
        
        test_ds = test
        global_model = SimpleCNN(num_classes=num_classes)

    # FedAvg rounds
    for rnd in range(1, cfg.training.rounds+1):
        # 1) Apply disturbances BEFORE sampling if enabled
        if cfg.disturbances.enabled and cfg.disturbances.apply_before_sampling:
            dist.step(rnd)
        # This couples the sampling strategy to the recovery mode as requested.
        use_smart = (cfg.disturbances.routing_mode == "dynamic")

        # 2) Sample online clients (this respects ephemeral offline state)
        selected_ids = net.sample_active_clients(cfg.training.clients_per_round, smart_sampling=use_smart)

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

            # The device "sends" the update. Target ID -1 implies server/gateway.
            net.devices[cid].send(to_id=-1, payload=None, size_bytes=bytes_up)

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
            
            # All clients who successfully participated must receive the new global model
            current_time = time.time()
            for cid in delivered_ids:
                # Create a dummy message representing the global model download
                msg = Message(
                    msg_type="global_model",
                    sender=-1, # -1 implies server
                    receiver=cid,
                    payload=None,
                    timestamp=current_time,
                    size_bytes=bytes_down
                )
                net.devices[cid].receive(msg)

            if cfg.disturbances.enabled:
                for cid in delivered_ids:
                    if cfg.disturbances.routing_mode == "naive":
                        path, t_s = net.shortest_gateway_path_naive(
                            source=cid, gateways=None, bytes_to_send=bytes_down
                        )
                    else:
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
