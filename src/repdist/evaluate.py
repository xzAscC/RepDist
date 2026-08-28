from __future__ import annotations

import json
from pathlib import Path

import torch

from repdist.checkpoint import best_path, latest_path, load_checkpoint
from repdist.config import ExperimentConfig
from repdist.diffusion import sample
from repdist.metrics import (
    covariance_spectrum,
    effective_rank,
    pca_subspace_metrics,
    sample_standard_normal,
    sliced_wasserstein_2,
)
from repdist.model import NoisePredictor
from repdist.normalize import Normalizer
from repdist.schedule import CosineSchedule
from repdist.store import HiddenStateStore
from repdist.visualize import validate_eval_payload, write_figures


def _load_model(
    cfg: ExperimentConfig, ckpt: dict, dim: int, device: str
) -> NoisePredictor:
    model = NoisePredictor(
        dim,
        hidden_dim=cfg.diffusion.hidden_dim,
        time_embed_dim=cfg.diffusion.time_embed_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def evaluate(cfg: ExperimentConfig, ckpt_name: str = "best") -> dict:
    device = cfg.resolve_device()
    store = HiddenStateStore(cfg.paths.data)
    train = store.load_split("train")
    test = store.load_split("test")
    ckpt_file = best_path(cfg.paths.checkpoints)
    if ckpt_name == "latest" or not ckpt_file.exists():
        ckpt_file = latest_path(cfg.paths.checkpoints)
    ckpt = load_checkpoint(ckpt_file, map_location=device)
    normalizer = Normalizer.from_state_dict(ckpt["normalizer"])

    dim = train.shape[1]
    schedule = CosineSchedule(
        cfg.diffusion.timesteps,
        cosine_s=cfg.diffusion.cosine_s,
        beta_max=cfg.diffusion.beta_max,
        device=device,
    )
    model = _load_model(cfg, ckpt, dim, device)
    gen_x = sample(
        model,
        n=test.shape[0],
        dim=dim,
        schedule=schedule,
        device=device,
        batch_size=cfg.eval.sample_batch_size,
    )
    normalized_train = normalizer.encode(train)
    normalized_real = normalizer.encode(test)
    normalized_diffusion = gen_x.cpu()
    random = sample_standard_normal(
        n=test.shape[0], dim=dim, seed=cfg.seed + 7, device="cpu"
    )

    spec_real = covariance_spectrum(normalized_real)
    spec_diff = covariance_spectrum(normalized_diffusion)
    spec_random = covariance_spectrum(random)

    metrics = {
        "eval_schema_version": 2,
        "comparison_space": "training-normalized",
        "baseline": "standard normal N(0,I)",
        "n_test": int(test.shape[0]),
        "n_train": int(train.shape[0]),
        "dim": int(dim),
        "ckpt": str(ckpt_file),
        "step": int(ckpt["step"]),
        "d_eff_real": float(effective_rank(spec_real)),
        "d_eff_diffusion": float(effective_rank(spec_diff)),
        "d_eff_random": float(effective_rank(spec_random)),
        "swd_real_diffusion": float(
            sliced_wasserstein_2(
                normalized_real,
                normalized_diffusion,
                cfg.eval.n_projections,
                cfg.seed,
            )
        ),
        "swd_real_random": float(
            sliced_wasserstein_2(
                normalized_real, random, cfg.eval.n_projections, cfg.seed
            )
        ),
        "swd_real_real_split": float(
            sliced_wasserstein_2(
                normalized_real[: test.shape[0] // 2],
                normalized_real[test.shape[0] // 2 :],
                cfg.eval.n_projections,
                cfg.seed,
            )
        ),
        **pca_subspace_metrics(
            normalized_train,
            normalized_real,
            normalized_diffusion,
            random,
            rank=cfg.eval.pca_rank,
            n_projections=cfg.eval.n_projections,
            seed=cfg.seed,
        ),
    }

    out_dir = Path(cfg.paths.outputs)
    fig_dir = out_dir / "figures"
    met_dir = out_dir / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)
    (met_dir / "eval.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save(
        {
            "spectrum_real": spec_real,
            "spectrum_diffusion": spec_diff,
            "spectrum_random": spec_random,
            "normalized_train": normalized_train,
            "normalized_real": normalized_real,
            "normalized_diffusion": normalized_diffusion,
            "standard_normal": random,
        },
        met_dir / "eval_tensors.pt",
    )
    write_figures(
        train=normalized_train,
        real=normalized_real,
        generated=normalized_diffusion,
        random=random,
        spec_real=spec_real,
        spec_diff=spec_diff,
        spec_random=spec_random,
        metrics=metrics,
        fig_dir=fig_dir,
        log_path=Path(cfg.paths.logs) / "train.jsonl",
    )
    return metrics


def visualize_from_saved(cfg: ExperimentConfig) -> dict:
    met_dir = Path(cfg.paths.outputs) / "metrics"
    payload = torch.load(
        met_dir / "eval_tensors.pt", map_location="cpu", weights_only=False
    )
    metrics = json.loads((met_dir / "eval.json").read_text())
    validate_eval_payload(payload, metrics)
    write_figures(
        train=payload["normalized_train"],
        real=payload["normalized_real"],
        generated=payload["normalized_diffusion"],
        random=payload["standard_normal"],
        spec_real=payload["spectrum_real"],
        spec_diff=payload["spectrum_diffusion"],
        spec_random=payload["spectrum_random"],
        metrics=metrics,
        fig_dir=Path(cfg.paths.outputs) / "figures",
        log_path=Path(cfg.paths.logs) / "train.jsonl",
    )
    return metrics
