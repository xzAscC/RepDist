from __future__ import annotations

import json
from pathlib import Path

from repdist.config import ExperimentConfig


def probe_layers(num_hidden_layers: int, current_layer: int = 16) -> list[int]:
    """First, 25%, current, 75%, and last transformer blocks (1-indexed after-block)."""
    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be >= 1")
    last = num_hidden_layers
    first = 1
    p25 = max(1, int(round(0.25 * num_hidden_layers)))
    p75 = max(1, int(round(0.75 * num_hidden_layers)))
    current = current_layer if 1 <= current_layer <= last else max(1, last // 2)
    ordered = [first, p25, current, p75, last]
    unique: list[int] = []
    for layer in ordered:
        if layer not in unique:
            unique.append(layer)
    return unique


def resolved_layers(cfg: ExperimentConfig) -> list[int]:
    if cfg.extract.layers:
        layers = [int(layer) for layer in cfg.extract.layers]
        unique: list[int] = []
        for layer in layers:
            if layer not in unique:
                unique.append(layer)
        if not unique:
            raise ValueError("extract.layers must not be empty when provided")
        return unique
    return [int(cfg.extract.layer)]


def layer_store_root(data_root: str | Path, layer: int, n_layers: int) -> Path:
    root = Path(data_root)
    if n_layers <= 1:
        return root
    return root / f"layer_{layer:02d}"


def _nest_layer(path: str | Path, layer_name: str) -> str:
    parsed = Path(path)
    if parsed.name == layer_name:
        return str(parsed)
    return str(parsed / layer_name)


def apply_layer_paths(cfg: ExperimentConfig, layer: int) -> ExperimentConfig:
    """Point train/eval paths at one extracted layer under the config's own roots."""
    cfg.extract.layer = int(layer)
    cfg.extract.layers = []
    layer_name = f"layer_{layer:02d}"
    cfg.paths.data = _nest_layer(cfg.paths.data, layer_name)
    cfg.paths.checkpoints = _nest_layer(cfg.paths.checkpoints, layer_name)
    cfg.paths.logs = _nest_layer(cfg.paths.logs, layer_name)
    cfg.paths.outputs = _nest_layer(cfg.paths.outputs, layer_name)
    normalizer = Path(cfg.paths.normalizer)
    if normalizer.parent.name != layer_name:
        cfg.paths.normalizer = str(normalizer.parent / layer_name / normalizer.name)
    return cfg


def load_layer_metrics(root: str | Path) -> dict[int, dict]:
    root = Path(root)
    metrics: dict[int, dict] = {}
    paths = list(root.glob("layer_*/outputs/metrics/eval.json")) + list(
        root.glob("layer_*/metrics/eval.json")
    )
    for path in sorted(set(paths)):
        layer_name = next(part for part in path.parts if part.startswith("layer_"))
        layer = int(layer_name.split("_")[1])
        if layer in metrics:
            continue
        metrics[layer] = json.loads(path.read_text())
        metrics[layer]["layer"] = layer
    return metrics
