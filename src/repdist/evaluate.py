from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from repdist.checkpoint import best_path, latest_path, load_checkpoint
from repdist.config import ExperimentConfig
from repdist.diffusion import sample
from repdist.metrics import (
    covariance_spectrum,
    effective_rank,
    pca_basis,
    project_pca,
    sample_gaussian,
    sliced_wasserstein_2,
)
from repdist.model import NoisePredictor
from repdist.normalize import Normalizer
from repdist.schedule import CosineSchedule
from repdist.store import HiddenStateStore


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
    generated = normalizer.decode(gen_x.cpu())
    gauss = sample_gaussian(train.mean(0), train, test.shape[0], seed=cfg.seed + 7)

    spec_real = covariance_spectrum(test)
    spec_diff = covariance_spectrum(generated)
    spec_gauss = covariance_spectrum(gauss)

    metrics = {
        "n_test": int(test.shape[0]),
        "n_train": int(train.shape[0]),
        "dim": int(dim),
        "ckpt": str(ckpt_file),
        "step": int(ckpt["step"]),
        "d_eff_real": float(effective_rank(spec_real)),
        "d_eff_diffusion": float(effective_rank(spec_diff)),
        "d_eff_gaussian": float(effective_rank(spec_gauss)),
        "swd_real_diffusion": float(
            sliced_wasserstein_2(test, generated, cfg.eval.n_projections, cfg.seed)
        ),
        "swd_real_gaussian": float(
            sliced_wasserstein_2(test, gauss, cfg.eval.n_projections, cfg.seed)
        ),
        "swd_real_real_split": float(
            sliced_wasserstein_2(
                test[: test.shape[0] // 2],
                test[test.shape[0] // 2 :],
                cfg.eval.n_projections,
                cfg.seed,
            )
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
            "spectrum_gaussian": spec_gauss,
        },
        met_dir / "spectra.pt",
    )

    k = min(64, spec_real.numel())
    plt.figure(figsize=(6, 4))
    plt.plot(spec_real[:k].numpy(), label="real")
    plt.plot(spec_diff[:k].numpy(), label="diffusion")
    plt.plot(spec_gauss[:k].numpy(), label="gaussian")
    plt.yscale("log")
    plt.xlabel("eigenvalue index")
    plt.ylabel("covariance eigenvalue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "spectrum.png", dpi=160)
    plt.close()

    mean, components, _ = pca_basis(train, rank=2)
    plt.figure(figsize=(6, 5))
    for name, tensor in (
        ("real", test),
        ("diffusion", generated),
        ("gaussian", gauss),
    ):
        xy = project_pca(tensor, mean, components)
        plt.scatter(xy[:, 0], xy[:, 1], s=6, alpha=0.35, label=name)
    plt.legend()
    plt.xlabel("PC1 (train basis)")
    plt.ylabel("PC2 (train basis)")
    plt.tight_layout()
    plt.savefig(fig_dir / "pca.png", dpi=160)
    plt.close()

    log_path = Path(cfg.paths.logs) / "train.jsonl"
    if log_path.exists():
        steps, train_loss, val_steps, val_loss = [], [], [], []
        for line in log_path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("split") == "train":
                steps.append(rec["step"])
                train_loss.append(rec["loss"])
            elif rec.get("split") == "val":
                val_steps.append(rec["step"])
                val_loss.append(rec["loss"])
        plt.figure(figsize=(6, 4))
        plt.plot(steps, train_loss, label="train", linewidth=1)
        plt.plot(val_steps, val_loss, label="val")
        plt.xlabel("step")
        plt.ylabel("diffusion MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "loss.png", dpi=160)
        plt.close()

    return metrics
