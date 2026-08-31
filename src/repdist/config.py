from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, cast, get_type_hints

import yaml


@dataclass
class PathsConfig:
    data: str = "data/hidden_states"
    checkpoints: str = "checkpoints"
    logs: str = "logs"
    outputs: str = "outputs"
    normalizer: str = "data/hidden_states/normalizer.pt"


@dataclass
class ExtractConfig:
    model_name: str = "allenai/Olmo-3-7B-Think"
    dataset_name: str = "allenai/Dolci-Think-SFT-7B"
    dataset_split: str = "train"
    layer: int = 16
    layers: list[int] = field(default_factory=list)
    max_prompt_tokens: int = 1024
    batch_size: int = 2
    shard_size: int = 512
    n_train_initial: int = 10000
    n_val: int = 5000
    n_test: int = 5000
    n_train_stream_chunk: int = 5000
    max_train_samples: int = 50000
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    synthetic_dim: int = 32
    synthetic_rank: int = 8


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    cosine_s: float = 0.008
    beta_max: float = 0.999
    hidden_dim: int = 8196
    n_hidden_layers: int = 1
    time_embed_dim: int = 128
    min_snr_gamma: float = 0.0
    latent_rank: int = 0
    zero_init_output: bool = True
    lr: float = 2e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    max_steps: int = 50000
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_every: int = 500
    ckpt_every: int = 500
    log_every: int = 50
    patience: int = 8
    min_delta: float = 1e-4
    stream_on_plateau: bool = True
    diagnostic_sample_count: int = 16
    diagnostic_every: int = 500
    max_healthy_final_rms: float = 3.0
    hard_abort_rms: float = 25.0
    diagnostic_timesteps: list[int] = field(default_factory=lambda: [900, 950, 999])

    def __post_init__(self) -> None:
        if self.diagnostic_sample_count < 1:
            raise ValueError("diagnostic_sample_count must be >= 1")
        if self.diagnostic_every < 1:
            raise ValueError("diagnostic_every must be >= 1")
        if self.max_healthy_final_rms <= 0:
            raise ValueError("max_healthy_final_rms must be > 0")
        if self.hard_abort_rms <= self.max_healthy_final_rms:
            raise ValueError("hard_abort_rms must exceed max_healthy_final_rms")
        if not self.diagnostic_timesteps:
            raise ValueError("diagnostic_timesteps must not be empty")
        if self.n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")
        if self.min_snr_gamma < 0:
            raise ValueError("min_snr_gamma must be >= 0")
        if self.latent_rank < 0:
            raise ValueError("latent_rank must be >= 0")


@dataclass
class EvalConfig:
    n_projections: int = 128
    sample_batch_size: int = 256
    pca_rank: int = 32
    match_train_rms: bool = False


@dataclass
class ExperimentConfig:
    seed: int = 0
    device: str = "cuda"
    paths: PathsConfig = field(default_factory=PathsConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        return cast(ExperimentConfig, _from_dict(cls, raw))

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open() as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    def resolve_device(self) -> str:
        if self.device == "cuda":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


def _from_dict(cls: Any, raw: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    raw = raw or {}
    hints = get_type_hints(cls)
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        nested = hints.get(f.name, f.type)
        if isinstance(nested, type) and is_dataclass(nested):
            kwargs[f.name] = _from_dict(nested, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)
