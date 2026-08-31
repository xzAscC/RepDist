from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from repdist.metrics import pca_basis, project_pca

REAL = "#2166ac"
DIFF = "#b2182b"
RANDOM = "#1b9e77"
SPLIT = "#7570b3"
TRAIN = "#999999"
RANDOM_LABEL = "standard normal N(0,I)"
DIFF_LINESTYLE = "-."
RANDOM_LINESTYLE = "--"
SCATTER_SIZE = 7
SCATTER_ALPHA = 0.4

REQUIRED_PAYLOAD_KEYS = {
    "normalized_train",
    "normalized_real",
    "normalized_diffusion",
    "standard_normal",
    "spectrum_real",
    "spectrum_diffusion",
    "spectrum_random",
}
REQUIRED_METRIC_KEYS = {
    "d_eff_real",
    "d_eff_diffusion",
    "d_eff_random",
    "swd_real_diffusion",
    "swd_real_random",
    "swd_real_real_split",
    "pca_rank",
    "pca_d_eff_real",
    "pca_d_eff_diffusion",
    "pca_d_eff_random",
    "pca_d_eff_real_split",
    "pca_swd_real_diffusion",
    "pca_swd_real_random",
    "pca_swd_real_real_split",
    "pca_covariance_relative_error_diffusion",
    "pca_covariance_relative_error_random",
    "pca_covariance_relative_error_real_split",
}


def validate_eval_payload(payload: dict, metrics: dict) -> None:
    legacy_keys = set(payload) | set(metrics)
    if any("gauss" in key.lower() for key in legacy_keys):
        raise ValueError(
            "legacy Gaussian eval payload is incompatible; rerun `repdist eval` "
            "to create normalized standard-normal metrics"
        )
    missing_payload = REQUIRED_PAYLOAD_KEYS - set(payload)
    missing_metrics = REQUIRED_METRIC_KEYS - set(metrics)
    schema_matches = (
        metrics.get("eval_schema_version") == 2
        and metrics.get("comparison_space") == "training-normalized"
        and metrics.get("baseline") == RANDOM_LABEL
    )
    if missing_payload or missing_metrics or not schema_matches:
        missing = sorted(missing_payload | missing_metrics)
        detail = f"; missing keys: {', '.join(missing)}" if missing else ""
        raise ValueError(
            "incompatible eval payload; rerun `repdist eval` to create normalized "
            f"standard-normal metrics{detail}"
        )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_spectrum(
    spec_real: Tensor, spec_diff: Tensor, spec_random: Tensor, path: Path
) -> None:
    k = min(128, spec_real.numel())
    idx = np.arange(1, k + 1)
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(idx, spec_real[:k].numpy(), color=REAL, lw=2, label="real $p_H$")
    plt.plot(
        idx,
        spec_diff[:k].numpy(),
        color=DIFF,
        lw=2,
        ls=DIFF_LINESTYLE,
        label="diffusion $p_\\theta$",
    )
    plt.plot(
        idx,
        spec_random[:k].numpy(),
        color=RANDOM,
        lw=2,
        ls=RANDOM_LINESTYLE,
        label=RANDOM_LABEL,
    )
    plt.yscale("log")
    plt.xlabel("eigenvalue index $i$")
    plt.ylabel(r"covariance eigenvalue $\lambda_i$")
    plt.title("Second-order geometry: covariance spectrum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_cumulative_spectrum(
    spec_real: Tensor, spec_diff: Tensor, spec_random: Tensor, path: Path
) -> None:
    def cdf(spec: Tensor) -> np.ndarray:
        c = spec.clamp_min(0).cumsum(0)
        return (c / c[-1].clamp_min(1e-12)).numpy()

    k = min(256, spec_real.numel())
    idx = np.arange(1, k + 1)
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(idx, cdf(spec_real)[:k], color=REAL, lw=2, label="real $p_H$")
    plt.plot(
        idx,
        cdf(spec_diff)[:k],
        color=DIFF,
        lw=2,
        ls=DIFF_LINESTYLE,
        label="diffusion $p_\\theta$",
    )
    plt.plot(
        idx,
        cdf(spec_random)[:k],
        color=RANDOM,
        lw=2,
        ls=RANDOM_LINESTYLE,
        label=RANDOM_LABEL,
    )
    plt.xlabel("eigenvalue index $i$")
    plt.ylabel("cumulative spectral mass")
    plt.title("Spectral concentration")
    plt.legend()
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_effective_rank(metrics: dict, path: Path) -> None:
    labels = ["real", "diffusion", "standard normal\nN(0,I)"]
    colors = [REAL, DIFF, RANDOM]
    panels = (
        (
            "full normalized space",
            [
                metrics["d_eff_real"],
                metrics["d_eff_diffusion"],
                metrics["d_eff_random"],
            ],
        ),
        (
            f"train-PCA subspace (rank {metrics['pca_rank']})",
            [
                metrics["pca_d_eff_real"],
                metrics["pca_d_eff_diffusion"],
                metrics["pca_d_eff_random"],
            ],
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for ax, (title, vals) in zip(axes, panels, strict=True):
        bars = ax.bar(labels, vals, color=colors, width=0.62)
        ax.set_title(title)
        ax.set_ylabel(r"effective rank $d_{\mathrm{eff}}$")
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    fig.suptitle(r"$d_{\mathrm{eff}}=(\sum\lambda_i)^2 / \sum\lambda_i^2$")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pca(
    train: Tensor, real: Tensor, diff: Tensor, random: Tensor, path: Path
) -> None:
    mean, components, _ = pca_basis(train, rank=2)
    panels = (
        ("real $p_H$", real, REAL),
        (r"diffusion $p_\theta$", diff, DIFF),
        (RANDOM_LABEL, random, RANDOM),
    )
    coords = [project_pca(t, mean, components).numpy() for _, t, _ in panels]
    all_xy = np.concatenate(coords, axis=0)
    lo = all_xy.min(axis=0)
    hi = all_xy.max(axis=0)
    pad = np.maximum(0.08 * (hi - lo), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharex=True, sharey=True)
    for ax, (title, _, color), xy in zip(axes, panels, coords, strict=True):
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            c=color,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("PC1 (train basis)")
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    axes[0].set_ylabel("PC2 (train basis)")
    fig.suptitle("Train-PCA view of normalized held-out and generated states", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_swd(metrics: dict, path: Path) -> None:
    labels = ["diffusion", "standard normal\nN(0,I)", "real split"]
    colors = [DIFF, RANDOM, SPLIT]
    panels = (
        (
            "full-space SWD",
            [
                metrics["swd_real_diffusion"],
                metrics["swd_real_random"],
                metrics["swd_real_real_split"],
            ],
            r"sliced Wasserstein $\mathrm{SWD}_2$",
        ),
        (
            f"PCA-rank-{metrics['pca_rank']} SWD",
            [
                metrics["pca_swd_real_diffusion"],
                metrics["pca_swd_real_random"],
                metrics["pca_swd_real_real_split"],
            ],
            r"sliced Wasserstein $\mathrm{SWD}_2$",
        ),
        (
            f"PCA-rank-{metrics['pca_rank']} covariance",
            [
                metrics["pca_covariance_relative_error_diffusion"],
                metrics["pca_covariance_relative_error_random"],
                metrics["pca_covariance_relative_error_real_split"],
            ],
            "covariance relative error",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))
    for ax, (title, vals, ylabel) in zip(axes, panels, strict=True):
        bars = ax.bar(labels, vals, color=colors, width=0.62)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3g}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    fig.suptitle("Normalized distribution matching (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_projections(
    real: Tensor, diff: Tensor, random: Tensor, path: Path, seed: int = 0
) -> None:
    g = torch.Generator(device=real.device).manual_seed(seed)
    directions = torch.randn(3, real.shape[1], generator=g)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharey=True)
    for ax, theta, k in zip(axes, directions, range(1, 4), strict=True):
        ax.hist(
            (real @ theta).numpy(),
            bins=40,
            density=True,
            histtype="step",
            color=REAL,
            lw=1.8,
            label="real",
        )
        ax.hist(
            (diff @ theta).numpy(),
            bins=40,
            density=True,
            histtype="step",
            color=DIFF,
            lw=1.8,
            ls=DIFF_LINESTYLE,
            label="diffusion",
        )
        ax.hist(
            (random @ theta).numpy(),
            bins=40,
            density=True,
            histtype="step",
            color=RANDOM,
            lw=1.8,
            ls=RANDOM_LINESTYLE,
            label=RANDOM_LABEL,
        )
        ax.set_xlabel(rf"$\theta_{k}^\top x$")
        ax.set_title(f"random projection {k}")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    fig.suptitle("Normalized one-dimensional projections used by SWD", y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_loss(log_path: Path, path: Path) -> None:
    if not log_path.exists():
        return
    steps, train_loss, val_steps, val_loss = [], [], [], []
    for line in log_path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("split") == "train":
            steps.append(rec["step"])
            train_loss.append(rec["loss"])
        elif rec.get("split") == "val":
            val_steps.append(rec["step"])
            val_loss.append(rec["loss"])
    if not steps:
        return
    plt.figure(figsize=(6.4, 4.0))
    train_marker = "o" if len(steps) == 1 else None
    val_marker = "o" if len(val_steps) == 1 else None
    plt.plot(
        steps,
        train_loss,
        color=TRAIN,
        lw=0.9,
        marker=train_marker,
        label="train",
    )
    plt.plot(
        val_steps,
        val_loss,
        color=REAL,
        lw=2.0,
        marker=val_marker,
        label="val",
    )
    plt.xlabel("step")
    plt.ylabel(r"diffusion loss $\mathcal{L}_{\mathrm{diff}}$")
    plt.title("Training and validation diffusion loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_figures(
    *,
    train: Tensor,
    real: Tensor,
    generated: Tensor,
    random: Tensor,
    spec_real: Tensor,
    spec_diff: Tensor,
    spec_random: Tensor,
    metrics: dict,
    fig_dir: Path,
    log_path: Path,
) -> list[Path]:
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        fig_dir / "spectrum.png",
        fig_dir / "spectrum_cumulative.png",
        fig_dir / "effective_rank.png",
        fig_dir / "pca.png",
        fig_dir / "swd.png",
        fig_dir / "projections.png",
        fig_dir / "loss.png",
    ]
    plot_spectrum(spec_real, spec_diff, spec_random, paths[0])
    plot_cumulative_spectrum(spec_real, spec_diff, spec_random, paths[1])
    plot_effective_rank(metrics, paths[2])
    plot_pca(train, real, generated, random, paths[3])
    plot_swd(metrics, paths[4])
    plot_projections(real, generated, random, paths[5])
    plot_loss(log_path, paths[6])
    return paths


def write_layer_comparison(layer_metrics: dict[int, dict], fig_dir: Path) -> list[Path]:
    if not layer_metrics:
        raise ValueError("layer_metrics must not be empty")
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted(layer_metrics)
    swd_diff = [layer_metrics[k]["swd_real_diffusion"] for k in layers]
    swd_rand = [layer_metrics[k]["swd_real_random"] for k in layers]
    deff_real = [layer_metrics[k]["d_eff_real"] for k in layers]
    deff_diff = [layer_metrics[k]["d_eff_diffusion"] for k in layers]
    deff_rand = [layer_metrics[k]["d_eff_random"] for k in layers]
    pca_err_diff = [
        layer_metrics[k]["pca_covariance_relative_error_diffusion"] for k in layers
    ]
    pca_err_rand = [
        layer_metrics[k]["pca_covariance_relative_error_random"] for k in layers
    ]

    swd_path = fig_dir / "swd_vs_layer.png"
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(layers, swd_diff, "o-", color=DIFF, label="diffusion")
    ax.plot(layers, swd_rand, "s--", color=RANDOM, label=RANDOM_LABEL)
    ax.set_xlabel("layer")
    ax.set_ylabel("sliced Wasserstein-2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(swd_path)
    plt.close(fig)

    deff_path = fig_dir / "deff_vs_layer.png"
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(layers, deff_real, "o-", color=REAL, label="real")
    ax.plot(layers, deff_diff, "s-.", color=DIFF, label="diffusion")
    ax.plot(layers, deff_rand, "^--", color=RANDOM, label=RANDOM_LABEL)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$d_{\mathrm{eff}}$")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(deff_path)
    plt.close(fig)

    pca_path = fig_dir / "pca_cov_error_vs_layer.png"
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(layers, pca_err_diff, "o-", color=DIFF, label="diffusion")
    ax.plot(layers, pca_err_rand, "s--", color=RANDOM, label=RANDOM_LABEL)
    ax.set_xlabel("layer")
    ax.set_ylabel("PCA-32 covariance relative error")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(pca_path)
    plt.close(fig)
    return [swd_path, deff_path, pca_path]
