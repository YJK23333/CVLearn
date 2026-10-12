import os
import random
import numpy as np
import torch


def set_seed(seed=48):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(device_str="cuda"):
    if(device_str == "cuda" and torch.cuda.is_available()):
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct, labels.size(0)

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":epoch,
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }, path)
    print(f"[save]{path}")

def load_checkpoint(model, path, device="cpu",optimizer=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[load]{path}(epoch{ckpt.get('epoch','?')})")
    return ckpt.get("epoch", 0)