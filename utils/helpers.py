"""utils/helpers.py — config loading, seeding, checkpointing."""

import os
import random
import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    epoch = state.get("epoch", 0)
    best  = state.get("best_metric", 0.0)
    print(f"[Checkpoint] Loaded from {path}  (epoch {epoch}, best {best:.4f})")
    return epoch, best


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
