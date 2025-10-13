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

def train_one_client(model, dataset, epochs, batch_size, lr, device):
    model = copy.deepcopy(model)
    model.to(device)
    model.train()
    
    # Use Adam optimizer with weight decay for better convergence
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.95)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       num_workers=2 if device.type == "cuda" else 0, 
                       pin_memory=(device.type == "cuda"))
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            total_loss += loss.item()
            num_batches += 1
        
        # Update learning rate
        scheduler.step()
        
        # Print progress for debugging (optional)
        if epoch == epochs - 1:  # Only on last epoch to reduce noise
            avg_loss = total_loss / max(num_batches, 1)
            print(f"  Client training - Final epoch loss: {avg_loss:.4f}")
    
    # Return CPU state dict so we can aggregate on CPU regardless of client device
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # return model.state_dict()


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2 if device.type == "cuda" else 0, pin_memory=(device.type == "cuda"))
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)
