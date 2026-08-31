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
from repdist.latent import LatentPCA
from repdist.layers import load_layer_metrics
from repdist.visualize import (
    validate_eval_payload,
    write_figures,
    write_layer_comparison,
)


def _match_rms(samples: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    ref_rms = torch.sqrt(torch.mean(reference.square()).clamp(min=1e-8))
    sample_rms = torch.sqrt(torch.mean(samples.square()).clamp(min=1e-8))
    return samples * (ref_rms / sample_rms)


def _load_model(
    cfg: ExperimentConfig, ckpt: dict, dim: int, device: str
) -> NoisePredictor:
    model = NoisePredictor(
        dim,
        hidden_dim=cfg.diffusion.hidden_dim,
        time_embed_dim=cfg.diffusion.time_embed_dim,
        n_hidden_layers=cfg.diffusion.n_hidden_layers,
        zero_init_output=cfg.diffusion.zero_init_output,
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

    normalized_train = normalizer.encode(train)
    normalized_real = normalizer.encode(test)
    pca = None
    extra = ckpt.get("extra") or {}
    if extra.get("pca") is not None:
        pca = LatentPCA.from_state_dict(extra["pca"])
    elif cfg.diffusion.latent_rank:
        pca = LatentPCA.fit(normalized_train, cfg.diffusion.latent_rank)
    latent_dim = pca.rank if pca is not None else train.shape[1]
    ambient_dim = train.shape[1]
    schedule = CosineSchedule(
        cfg.diffusion.timesteps,
        cosine_s=cfg.diffusion.cosine_s,
        beta_max=cfg.diffusion.beta_max,
        device=device,
    )
    model = _load_model(cfg, ckpt, latent_dim, device)
    gen_x = sample(
        model,
        n=test.shape[0],
        dim=latent_dim,
        schedule=schedule,
        device=device,
        batch_size=cfg.eval.sample_batch_size,
    )
    normalized_diffusion = pca.decode(gen_x.cpu()) if pca is not None else gen_x.cpu()
    if normalized_diffusion.shape[1] != ambient_dim:
        raise RuntimeError(
            f"decoded diffusion dim {tuple(normalized_diffusion.shape)} != ambient {ambient_dim}"
        )
    if cfg.eval.match_train_rms:
        normalized_diffusion = _match_rms(normalized_diffusion, normalized_train)
    random = sample_standard_normal(
        n=test.shape[0], dim=ambient_dim, seed=cfg.seed + 7, device="cpu"
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
        "dim": int(ambient_dim),
        "ckpt": str(ckpt_file),
        "step": int(ckpt["step"]),
        "match_train_rms": bool(cfg.eval.match_train_rms),
        "layer": int(cfg.extract.layer),
        "latent_rank": int(pca.rank if pca is not None else 0),
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


def compare_layer_runs(run_root: str | Path, out_dir: str | Path) -> dict:
    layer_metrics = load_layer_metrics(run_root)
    if not layer_metrics:
        raise FileNotFoundError(f"no layer eval.json files under {run_root}")
    comparison = {
        "run_root": str(run_root),
        "layers": sorted(layer_metrics),
        "metrics": {str(k): v for k, v in sorted(layer_metrics.items())},
        "beats_random_swd": {
            str(k): v["swd_real_diffusion"] < v["swd_real_random"]
            for k, v in sorted(layer_metrics.items())
        },
        "beats_random_pca_swd": {
            str(k): v["pca_swd_real_diffusion"] < v["pca_swd_real_random"]
            for k, v in sorted(layer_metrics.items())
        },
        "beats_random_pca_cov": {
            str(k): v["pca_covariance_relative_error_diffusion"]
            < v["pca_covariance_relative_error_random"]
            for k, v in sorted(layer_metrics.items())
        },
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    lines = [
        "# Layer comparison",
        "",
        "| layer | d_eff real | d_eff diff | SWD diff | SWD random | PCA-SWD diff | PCA-SWD random | PCA cov err diff | beats SWD | beats PCA-SWD |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for layer, metrics in sorted(layer_metrics.items()):
        beat = metrics["swd_real_diffusion"] < metrics["swd_real_random"]
        beat_pca = metrics["pca_swd_real_diffusion"] < metrics["pca_swd_real_random"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(layer),
                    f"{metrics['d_eff_real']:.3g}",
                    f"{metrics['d_eff_diffusion']:.3g}",
                    f"{metrics['swd_real_diffusion']:.4f}",
                    f"{metrics['swd_real_random']:.4f}",
                    f"{metrics['pca_swd_real_diffusion']:.4f}",
                    f"{metrics['pca_swd_real_random']:.4f}",
                    f"{metrics['pca_covariance_relative_error_diffusion']:.4f}",
                    "yes" if beat else "no",
                    "yes" if beat_pca else "no",
                ]
            )
            + " |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    write_layer_comparison(layer_metrics, out)
    return comparison
