from __future__ import annotations

import json
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
