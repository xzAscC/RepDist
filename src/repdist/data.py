from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


class HiddenStateStore:
    """Shard-backed store for train/val/test hidden states."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {
                "cursor": 0,
                "n_val": 0,
                "n_test": 0,
                "n_train": 0,
                "dim": None,
                "train_shards": [],
            }
        return json.loads(self.manifest_path.read_text())

    def _save_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    @property
    def manifest(self) -> dict:
        return self._load_manifest()

    def val_path(self) -> Path:
        return self.root / "val.pt"

    def test_path(self) -> Path:
        return self.root / "test.pt"

    def has_val_test(self) -> bool:
        return self.val_path().exists() and self.test_path().exists()

    def n_train(self) -> int:
        return int(self._load_manifest().get("n_train", 0))

    def dim(self) -> int | None:
        return self._load_manifest().get("dim")

    def save_split(self, name: str, hidden: Tensor, cursor: int) -> None:
        if name not in {"val", "test"}:
            raise ValueError(name)
        path = self.val_path() if name == "val" else self.test_path()
        torch.save(hidden.cpu().to(torch.float16), path)
        manifest = self._load_manifest()
        manifest[f"n_{name}"] = int(hidden.shape[0])
        manifest["dim"] = int(hidden.shape[1])
        manifest["cursor"] = int(cursor)
        self._save_manifest(manifest)

    def append_train(self, hidden: Tensor, cursor: int) -> Path:
        manifest = self._load_manifest()
        shard_id = len(manifest.get("train_shards", []))
        path = self.root / f"train_{shard_id:05d}.pt"
        torch.save(hidden.cpu().to(torch.float16), path)
        shards = list(manifest.get("train_shards", []))
        shards.append(path.name)
        manifest["train_shards"] = shards
        manifest["n_train"] = int(manifest.get("n_train", 0)) + int(hidden.shape[0])
        manifest["dim"] = int(hidden.shape[1])
        manifest["cursor"] = int(cursor)
        self._save_manifest(manifest)
        return path

    def load_split(self, name: str) -> Tensor:
        if name == "train":
            return self.load_train()
        path = self.val_path() if name == "val" else self.test_path()
        if not path.exists():
            raise FileNotFoundError(path)
        return torch.load(path, map_location="cpu", weights_only=True).float()

    def load_train(self) -> Tensor:
        manifest = self._load_manifest()
        shards = manifest.get("train_shards", [])
        if not shards:
            raise FileNotFoundError(f"no train shards in {self.root}")
        parts = [
            torch.load(self.root / name, map_location="cpu", weights_only=True).float()
            for name in shards
        ]
        return torch.cat(parts, dim=0)


@dataclass(frozen=True)
class Normalizer:
    """Training-set centering plus a global scalar scale."""

    mean: Tensor
    scale: float

    def encode(self, hidden: Tensor) -> Tensor:
        return (
            hidden - self.mean.to(device=hidden.device, dtype=hidden.dtype)
        ) / self.scale

    def decode(self, x: Tensor) -> Tensor:
        return x * self.scale + self.mean.to(device=x.device, dtype=x.dtype)

    def state_dict(self) -> dict[str, Tensor | float]:
        return {"mean": self.mean.detach().cpu().clone(), "scale": float(self.scale)}

    def fingerprint(self) -> str:
        mean = self.mean.detach().cpu().float().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(mean.shape)).encode())
        digest.update(mean.numpy().tobytes())
        digest.update(repr(float(self.scale)).encode())
        return digest.hexdigest()

    def matches_fingerprint(self, fingerprint: str) -> bool:
        return self.fingerprint() == fingerprint

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor | float]) -> Normalizer:
        mean = state["mean"]
        if not isinstance(mean, Tensor):
            raise TypeError("normalizer mean must be a tensor")
        return cls(mean=mean.float(), scale=float(state["scale"]))

    @classmethod
    def fit(cls, hidden: Tensor) -> Normalizer:
        mean = hidden.mean(dim=0)
        centered = hidden - mean
        scale = float(centered.pow(2).mean().sqrt().clamp_min(1e-8).item())
        return cls(mean=mean.cpu(), scale=scale)


class LatentPCA:
    """Train-only PCA map used to diffuse in a shared low-dimensional basis."""

    def __init__(self, mean: Tensor, components: Tensor, scale: Tensor) -> None:
        self.mean = mean.float()
        self.components = components.float()
        self.scale = scale.float().reshape(-1)

    @property
    def rank(self) -> int:
        return int(self.components.shape[1])

    @classmethod
    def fit(cls, hidden: Tensor, rank: int) -> LatentPCA:
        if rank < 1:
            raise ValueError("latent rank must be >= 1")
        mean = hidden.mean(dim=0)
        centered = hidden - mean
        q = min(rank, *centered.shape)
        _, _, v = torch.pca_lowrank(centered, q=q, center=False)
        components = v[:, :rank].contiguous()
        latent = centered @ components
        scale = latent.std(dim=0, unbiased=False).clamp_min(1e-8)
        return cls(mean.cpu(), components.cpu(), scale.cpu())

    def encode(self, hidden: Tensor) -> Tensor:
        mean = self.mean.to(hidden.device)
        components = self.components.to(hidden.device)
        scale = self.scale.to(hidden.device)
        return ((hidden - mean) @ components) / scale

    def decode(self, latent: Tensor) -> Tensor:
        components = self.components.to(latent.device)
        mean = self.mean.to(latent.device)
        scale = self.scale.to(latent.device)
        return (latent * scale) @ components.T + mean

    def state_dict(self) -> dict[str, Tensor]:
        return {"mean": self.mean, "components": self.components, "scale": self.scale}

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> LatentPCA:
        return cls(state["mean"], state["components"], state["scale"])
