from __future__ import annotations

import json
from pathlib import Path

import torch

from repdist.checkpoint import load_checkpoint, step_path
from repdist.config import ExperimentConfig
from repdist.data import HiddenStateStore, LatentPCA, Normalizer
from repdist.ddpm import CosineSchedule, NoisePredictor, sample
from repdist.metrics import sample_standard_normal, sliced_wasserstein_2


def candidate_steps(ckpt_dir: str | Path) -> list[int]:
    """Sorted training steps that have a step_*.pt checkpoint."""
    return sorted(
        int(path.stem.split("_")[1]) for path in Path(ckpt_dir).glob("step_*.pt")
    )


def load_step_model(
    cfg: ExperimentConfig, step: int, device: str
) -> tuple[dict, NoisePredictor]:
    """Load one step checkpoint and its eval-ready model."""
    ckpt = load_checkpoint(step_path(cfg.paths.checkpoints, step), map_location=device)
    model = NoisePredictor(
        int(ckpt["model_spec"]["data_dim"]),
        hidden_dim=cfg.diffusion.hidden_dim,
        time_embed_dim=cfg.diffusion.time_embed_dim,
        n_hidden_layers=cfg.diffusion.n_hidden_layers,
        zero_init_output=cfg.diffusion.zero_init_output,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return ckpt, model


def _frozen_preprocessing(ckpt: dict) -> tuple[Normalizer, LatentPCA | None]:
    normalizer = Normalizer.from_state_dict(ckpt["normalizer"])
    extra = ckpt.get("extra") or {}
    pca = LatentPCA.from_state_dict(extra["pca"]) if extra.get("pca") else None
    return normalizer, pca


@torch.no_grad()
def select_best_swd(cfg: ExperimentConfig, steps: list[int] | None = None) -> dict:
    """Rank step checkpoints by sliced Wasserstein-2 against the val split.

    Sampling uses a fixed seed per candidate so the ranking reflects model
    quality, not sampling noise. Writes swd_selection.json under cfg.paths.logs.
    """
    device = cfg.resolve_device()
    store = HiddenStateStore(cfg.paths.data)
    val = store.load_split("val")
    steps = (
        sorted(steps) if steps is not None else candidate_steps(cfg.paths.checkpoints)
    )
    if not steps:
        raise FileNotFoundError(
            f"no step_*.pt checkpoints under {cfg.paths.checkpoints}"
        )
    missing = [s for s in steps if not step_path(cfg.paths.checkpoints, s).exists()]
    if missing:
        raise FileNotFoundError(f"missing step checkpoints: {missing}")

    schedule = CosineSchedule(
        cfg.diffusion.timesteps,
        cosine_s=cfg.diffusion.cosine_s,
        beta_max=cfg.diffusion.beta_max,
        device=device,
    )
    first_ckpt, _ = load_step_model(cfg, steps[0], device)
    normalizer, _ = _frozen_preprocessing(first_ckpt)
    x_val = normalizer.encode(val)
    rows: list[dict] = []
    for step in steps:
        ckpt, model = load_step_model(cfg, step, device)
        _, pca = _frozen_preprocessing(ckpt)
        latent_dim = pca.rank if pca is not None else val.shape[1]
        generator = torch.Generator(device=device).manual_seed(cfg.seed)
        gen = sample(
            model,
            n=x_val.shape[0],
            dim=latent_dim,
            schedule=schedule,
            device=device,
            batch_size=cfg.eval.sample_batch_size,
            generator=generator,
            center=cfg.diffusion.reverse_center,
        )
        generated = pca.decode(gen.cpu()) if pca is not None else gen.cpu()
        rows.append(
            {
                "step": step,
                "swd_val": float(
                    sliced_wasserstein_2(
                        x_val, generated, cfg.eval.n_projections, cfg.seed
                    )
                ),
                "ckpt": str(step_path(cfg.paths.checkpoints, step)),
            }
        )
    rows.sort(key=lambda row: row["swd_val"])

    random = sample_standard_normal(
        n=val.shape[0], dim=val.shape[1], seed=cfg.seed + 7, device="cpu"
    )
    swd_random = float(
        sliced_wasserstein_2(x_val, random, cfg.eval.n_projections, cfg.seed)
    )
    result = {
        "layer": int(cfg.extract.layer),
        "split": "val",
        "seed": int(cfg.seed),
        "n_projections": int(cfg.eval.n_projections),
        "n_candidates": len(rows),
        "candidates": rows,
        "best_step": rows[0]["step"],
        "best_swd_val": rows[0]["swd_val"],
        "swd_val_random": swd_random,
        "beats_random": rows[0]["swd_val"] < swd_random,
    }
    log_dir = Path(cfg.paths.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "swd_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
