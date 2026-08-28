from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from repdist.normalize import Normalizer


def latest_path(ckpt_dir: str | Path) -> Path:
    return Path(ckpt_dir) / "latest.pt"


def best_path(ckpt_dir: str | Path) -> Path:
    return Path(ckpt_dir) / "best.pt"


def step_path(ckpt_dir: str | Path, step: int) -> Path:
    return Path(ckpt_dir) / f"step_{step:07d}.pt"


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler_state: dict | None,
    normalizer: Normalizer,
    best_val: float,
    patience_left: int,
    rng_state: dict | None = None,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state,
        "normalizer": normalizer.state_dict(),
        "best_val": best_val,
        "patience_left": patience_left,
        "rng": rng_state or capture_rng(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def capture_rng() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
