from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from repdist.data import Normalizer

CHECKPOINT_FORMAT = "repdist-checkpoint-v2"


class CheckpointCompatibilityError(ValueError):
    pass


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
    model_spec: dict | None = None,
    schedule_spec: dict | None = None,
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
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_spec": model_spec or _model_spec(model),
        "schedule_spec": schedule_spec or {},
        "normalizer_fingerprint": normalizer.fingerprint(),
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
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda"]])


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    validate_checkpoint_compatibility(payload)
    return payload


def _model_spec(model: nn.Module) -> dict:
    return {
        "architecture": type(model).__name__,
        "data_dim": getattr(model, "data_dim", None),
        "hidden_dim": getattr(model, "hidden_dim", None)
        or getattr(getattr(model, "fc1", None), "out_features", None),
        "n_hidden_layers": getattr(model, "n_hidden_layers", 1),
        "time_embed_dim": getattr(
            getattr(getattr(model, "time_embed", None), "0", None), "dim", None
        ),
        "full_rank_skip": True,
        "latent_rank": getattr(model, "latent_rank", 0),
    }


def validate_checkpoint_compatibility(
    payload: dict,
    *,
    model_spec: dict | None = None,
    schedule_spec: dict | None = None,
    normalizer_fingerprint: str | None = None,
) -> None:
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("checkpoint payload is not a schema")
    if "checkpoint_format" not in payload:
        raise CheckpointCompatibilityError("schema-less checkpoint is rejected")
    if payload["checkpoint_format"] != CHECKPOINT_FORMAT:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint format: {payload['checkpoint_format']!r}"
        )
    required = {"model_spec", "schedule_spec", "normalizer_fingerprint"}
    missing = sorted(required - payload.keys())
    if missing:
        raise CheckpointCompatibilityError(
            f"checkpoint metadata is incomplete: missing {', '.join(missing)}"
        )
    for name, expected in (
        ("model_spec", model_spec),
        ("schedule_spec", schedule_spec),
    ):
        if expected is not None and payload.get(name) != expected:
            raise CheckpointCompatibilityError(f"checkpoint {name} is incompatible")
    if (
        normalizer_fingerprint is not None
        and payload.get("normalizer_fingerprint") != normalizer_fingerprint
    ):
        raise CheckpointCompatibilityError("checkpoint normalizer is incompatible")
    if "normalizer" not in payload:
        raise CheckpointCompatibilityError(
            "checkpoint metadata is incomplete: missing normalizer"
        )
    try:
        normalizer = Normalizer.from_state_dict(payload["normalizer"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError("checkpoint normalizer is invalid") from exc
    if not normalizer.matches_fingerprint(payload["normalizer_fingerprint"]):
        raise CheckpointCompatibilityError(
            "checkpoint normalizer fingerprint does not match payload"
        )
