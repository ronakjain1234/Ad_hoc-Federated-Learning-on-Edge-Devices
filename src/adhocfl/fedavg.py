from typing import Dict, List, Tuple
import copy
from xml.parsers.expat import model
import torch
from torch.utils.data import DataLoader

def fedavg(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float] = None):
    if weights is None:
        weights = [1.0/len(state_dicts)] * len(state_dicts)
    agg = copy.deepcopy(state_dicts[0])
    for k in agg.keys():
        agg[k] = agg[k].float() * weights[0]
    for i in range(1, len(state_dicts)):
        for k in agg.keys():
            agg[k] += state_dicts[i][k].float() * weights[i]
    return agg

def weighted_fedavg(updates, skip_bn: bool = True):
    """
    updates: list[(state_dict, num_samples)]
    Returns a new (server) state_dict using sample-size–weighted averaging.
    If skip_bn=True (FedBN-style), we DO NOT aggregate BatchNorm params/buffers.
    """
    assert len(updates) > 0
    ref_sd, _ = updates[0]
    ref_keys = list(ref_sd.keys())

    # Identify BN keys in a state_dict (works without model class access)
    def is_bn_key(k: str) -> bool:
        if ("running_mean" in k) or ("running_var" in k) or ("num_batches_tracked" in k):
            return True                       # BN buffers
        parts = k.split(".")
        # common BN module names: bn, bn1, bn2, downsample.1, etc.
        if any(("bn" in p) for p in parts) and parts[-1] in ("weight", "bias"):
            return True                       # BN affine params
        return False

    agg_keys = [k for k in ref_keys if not (skip_bn and is_bn_key(k))]

    # Weighted sum for non-BN keys
    total = float(sum(n for _, n in updates)) + 1e-12
    out = {k: None for k in ref_keys}
    for sd, n in updates:
        w = n / total
        for k in agg_keys:
            v = sd[k].float()
            out[k] = (v * w) if out[k] is None else (out[k] + v * w)

    # For skipped BN keys, copy from the first client (any is fine)
    for k in ref_keys:
        if k not in agg_keys:
            out[k] = ref_sd[k].clone()
        else:
            out[k] = out[k].type_as(ref_sd[k])

    return out

def train_one_client(model, dataset, epochs, batch_size, lr, device):
    model = copy.deepcopy(model)
    model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2 if device.type == "cuda" else 0, pin_memory=(device.type == "cuda"), persistent_workers=(device.type == "cuda"))
    # ---- probe: how many optimizer steps will this client take per epoch?
    num_batches = len(loader)
    # Print occasionally to avoid spam; comment out after you tune
    print(f"[client] n={len(dataset)} bs={batch_size} steps/epoch={num_batches}")
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()
    # Return CPU state dict so we can aggregate on CPU regardless of client device
    # return {k: v.detach().cpu() for k, v in model.state_dict().items()}
    num_samples = len(dataset)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}, num_samples
    # return model.state_dict()


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2 if device.type == "cuda" else 0, pin_memory=(device.type == "cuda"), persistent_workers=(device.type == "cuda"))
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)
