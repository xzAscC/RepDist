from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import torch

from repdist.config import ExperimentConfig, apply_layer_paths, resolved_layers
from repdist.data import HiddenStateStore, LatentPCA, Normalizer
from repdist.ddpm import CosineSchedule
from repdist.metrics import (
    covariance_spectrum,
    effective_rank,
    pca_basis,
    sliced_wasserstein_2,
)
from repdist.select import load_step_model
from repdist.train import eval_loss


def family_metrics(
    real: torch.Tensor,
    gen: torch.Tensor,
    train: torch.Tensor,
    seed: int,
    n_projections: int,
    rank: int = 32,
) -> dict[str, float]:
    """Ambient-space geometry and distance of a sample family vs real states."""
    cov_real = torch.cov(real.T)
    cov_gen = torch.cov(gen.T)
    dcov = (cov_real - cov_gen).norm() / cov_real.norm().clamp_min(1e-12)
    torch.manual_seed(seed)
    _, v_real, _ = pca_basis(train, rank)
    _, v_gen, _ = pca_basis(gen, rank)
    k = min(v_real.shape[1], v_gen.shape[1])
    subsim = (v_real.T @ v_gen).square().sum() / k
    return {
        "subsim32": float(subsim),
        "dcov": float(dcov),
        "swd": float(sliced_wasserstein_2(real, gen, n_projections, seed)),
        "d_eff": float(effective_rank(covariance_spectrum(gen))),
    }


def gaussian_baselines(
    train: torch.Tensor, n: int, seed: int
) -> dict[str, torch.Tensor]:
    """Fitted, isotropic, and diagonal Gaussians of the training statistics."""
    mu = train.mean(dim=0)
    dim = train.shape[1]
    cov = torch.cov(train.T)
    ridge = 1e-6 * cov.diagonal().mean().clamp_min(1e-12)
    chol = torch.linalg.cholesky(cov + ridge * torch.eye(dim, device=cov.device))
    generator = torch.Generator(device=train.device).manual_seed(seed)
    eps = torch.randn(n, dim, generator=generator, device=train.device)
    sigma = (train - mu).square().mean().sqrt()
    return {
        "fitted": mu + eps @ chol.T,
        "isotropic": mu + sigma * eps,
        "diagonal": mu + eps * train.std(dim=0, unbiased=False),
    }


def _load_eval_tensors(root: Path, layer: int, device: str) -> dict[str, torch.Tensor]:
    path = root / f"layer_{layer:02d}" / "eval_tensors.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        key: value.to(device)
        for key, value in payload.items()
        if torch.is_tensor(value)
    }


def build_geom_metrics(
    run_roots: Mapping[str, str | Path],
    layers: list[int],
    main_run: str,
    seed: int,
    n_projections: int,
    device: str = "cpu",
    frozen: dict | None = None,
) -> dict:
    """Cross-run ambient-space metrics consumed by notebooks/figs/geom_metrics.json.

    Each run root must contain layer_XX/eval_tensors.pt (from `repdist eval`);
    real/train/N(0,I) references come from main_run's tensors. Families whose
    tensors are gone (legacy runs with deleted artifacts) fall back to their
    frozen entries verbatim when `frozen` (a previous geom_metrics.json) is
    provided.
    """
    if main_run not in run_roots:
        raise KeyError(f"main run {main_run!r} not among runs {sorted(run_roots)}")
    frozen = frozen or {}
    out_layers: dict[str, dict] = {}
    d_eff_real: dict[str, float] = {}
    d_eff_random: dict[str, float] = {}
    for layer in layers:
        key = str(layer)
        frozen_layer = frozen.get("layers", {}).get(key, {})
        tensors: dict[str, dict[str, torch.Tensor] | None] = {}
        for family, root in run_roots.items():
            tensors_path = Path(root) / f"layer_{layer:02d}" / "eval_tensors.pt"
            if tensors_path.exists():
                tensors[family] = _load_eval_tensors(Path(root), layer, device)
            else:
                entry = frozen_layer.get("diffusion", {}).get(family)
                if entry is None:
                    raise FileNotFoundError(
                        f"{tensors_path} is missing and no frozen entry exists "
                        f"for diffusion family {family!r}"
                    )
                tensors[family] = None
        main_tensors = tensors[main_run]
        if main_tensors is None:
            raise FileNotFoundError(
                f"main run {main_run!r} has no eval_tensors.pt for layer {layer}"
            )
        real = main_tensors["normalized_real"]
        train = main_tensors["normalized_train"]
        baselines = gaussian_baselines(train, real.shape[0], seed + 11)
        baselines["n01"] = main_tensors["standard_normal"]
        diffusion = {
            family: (
                frozen_layer["diffusion"][family]
                if t is None
                else family_metrics(
                    real, t["normalized_diffusion"], train, seed, n_projections
                )
            )
            for family, t in tensors.items()
        }
        baseline_metrics = {
            family: family_metrics(real, gen, train, seed, n_projections)
            for family, gen in baselines.items()
        }
        out_layers[key] = {
            "real_d_eff": float(effective_rank(covariance_spectrum(real))),
            "diffusion": diffusion,
            "baselines": baseline_metrics,
        }
        d_eff_real[key] = out_layers[key]["real_d_eff"]
        d_eff_random[key] = baseline_metrics["n01"]["d_eff"]
    return {
        "layers": out_layers,
        "d_eff_real": d_eff_real,
        "d_eff_random": d_eff_random,
    }


def write_loss_summary(cfg: ExperimentConfig, layers: list[int] | None = None) -> dict:
    """Test epsilon-MSE at each layer's SWD-selected checkpoint.

    Requires swd_selection.json from `repdist select-swd` per layer; writes
    loss_summary.json under the run's log root.
    """
    device = cfg.resolve_device()
    schedule = CosineSchedule(
        cfg.diffusion.timesteps,
        cosine_s=cfg.diffusion.cosine_s,
        beta_max=cfg.diffusion.beta_max,
        device=device,
    )
    layers = layers if layers is not None else resolved_layers(cfg)
    out: dict = {"layers": {}}
    for layer in layers:
        layer_cfg = apply_layer_paths(deepcopy(cfg), layer)
        selection_path = Path(layer_cfg.paths.logs) / "swd_selection.json"
        if not selection_path.exists():
            raise FileNotFoundError(f"run select-swd first: {selection_path}")
        best_step = int(json.loads(selection_path.read_text())["best_step"])
        ckpt, model = load_step_model(layer_cfg, best_step, device)
        normalizer = Normalizer.from_state_dict(ckpt["normalizer"])
        pca = None
        extra = ckpt.get("extra") or {}
        if extra.get("pca"):
            pca = LatentPCA.from_state_dict(extra["pca"])
        store = HiddenStateStore(layer_cfg.paths.data)
        x_test = normalizer.encode(store.load_split("test"))
        if pca is not None:
            x_test = pca.encode(x_test)
        test_loss = eval_loss(
            model,
            x_test,
            schedule,
            cfg.diffusion.batch_size,
            device,
            min_snr_gamma=cfg.diffusion.min_snr_gamma,
        )
        out["layers"][str(layer)] = {
            "best_step": best_step,
            "test_loss": float(test_loss),
        }
    log_dir = Path(cfg.paths.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "loss_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    return out
