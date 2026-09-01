import json
import math
from copy import deepcopy
from pathlib import Path

import torch

from repdist.config import ExperimentConfig, apply_layer_paths
from repdist.report import build_geom_metrics, family_metrics, write_loss_summary
from repdist.select import select_best_swd
from repdist.train import train

METRIC_KEYS = {"subsim32", "dcov", "swd", "d_eff"}


def _smoke_cfg(tmp_path: Path, max_steps: int) -> ExperimentConfig:
    cfg = ExperimentConfig.load("configs/smoke.yaml")
    cfg.paths.data = str(tmp_path / "data")
    cfg.paths.checkpoints = str(tmp_path / "checkpoints")
    cfg.paths.logs = str(tmp_path / "logs")
    cfg.paths.figs = str(tmp_path / "figs")
    cfg.paths.normalizer = str(tmp_path / "normalizer.pt")
    cfg.device = "cpu"
    cfg.diffusion.max_steps = max_steps
    cfg.diffusion.eval_every = max(max_steps // 2, 1)
    cfg.diffusion.ckpt_every = max(max_steps // 2, 1)
    cfg.diffusion.log_every = 1
    cfg.diffusion.warmup_steps = 1
    cfg.diffusion.diagnostic_sample_count = 2
    cfg.diffusion.diagnostic_every = 50
    return cfg


def _write_fake_run(root: Path, family: str, dim: int = 8, n: int = 48) -> None:
    g = torch.Generator().manual_seed(hash(family) % 2**31)
    train_t = torch.randn(n, dim, generator=g) * 2.0 + 0.5
    real = train_t[: n // 2] + 0.01 * torch.randn(n // 2, dim, generator=g)
    diffusion = real + 0.1 * torch.randn(n // 2, dim, generator=g)
    n01 = torch.randn(n // 2, dim, generator=g)
    layer_dir = root / "layer_16"
    layer_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "normalized_train": train_t,
            "normalized_real": real,
            "normalized_diffusion": diffusion,
            "standard_normal": n01,
        },
        layer_dir / "eval_tensors.pt",
    )


def test_family_metrics_matches_identical_distributions(tmp_path: Path):
    g = torch.Generator().manual_seed(0)
    train_t = torch.randn(64, 8, generator=g)
    real = train_t[:32]

    metrics = family_metrics(real, real, train_t, seed=0, n_projections=8)

    assert set(metrics) == METRIC_KEYS
    assert all(isinstance(v, float) and math.isfinite(v) for v in metrics.values())
    assert metrics["dcov"] < 1e-8
    assert metrics["subsim32"] > 0.95
    assert metrics["swd"] < 1e-6


def test_build_geom_metrics_matches_notebook_schema(tmp_path: Path):
    runs = {name: tmp_path / name for name in ("1024", "8196", "50k")}
    for name, root in runs.items():
        _write_fake_run(root, name)

    geom = build_geom_metrics(
        runs, layers=[16], main_run="50k", seed=0, n_projections=8
    )

    layer = geom["layers"]["16"]
    assert math.isfinite(layer["real_d_eff"])
    assert set(layer["diffusion"]) == {"1024", "8196", "50k"}
    assert set(layer["baselines"]) == {"n01", "fitted", "isotropic", "diagonal"}
    for group in ("diffusion", "baselines"):
        for entry in layer[group].values():
            assert set(entry) == METRIC_KEYS
            assert all(math.isfinite(v) for v in entry.values())
    assert set(geom["d_eff_real"]) == {"16"}
    assert set(geom["d_eff_random"]) == {"16"}


def test_build_geom_metrics_freezes_missing_families(tmp_path: Path):
    runs = {name: tmp_path / name for name in ("1024", "8196", "50k")}
    for name, root in runs.items():
        _write_fake_run(root, name)
    (runs["1024"] / "layer_16" / "eval_tensors.pt").unlink()
    frozen = {
        "layers": {
            "16": {
                "diffusion": {
                    "1024": {
                        "subsim32": 0.5,
                        "dcov": 0.9,
                        "swd": 1.5,
                        "d_eff": 7.0,
                    }
                }
            }
        }
    }

    geom = build_geom_metrics(
        runs, layers=[16], main_run="50k", seed=0, n_projections=8, frozen=frozen
    )

    assert (
        geom["layers"]["16"]["diffusion"]["1024"]
        == frozen["layers"]["16"]["diffusion"]["1024"]
    )
    assert set(geom["layers"]["16"]["diffusion"]) == {"1024", "8196", "50k"}

    try:
        build_geom_metrics(runs, layers=[16], main_run="50k", seed=0, n_projections=8)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing family without frozen entry must fail")


def test_write_loss_summary_uses_selected_step(tmp_path: Path):
    cfg = _smoke_cfg(tmp_path, max_steps=6)
    cfg.diffusion.ckpt_every = 3
    cfg.diffusion.step_ckpt_every = 3
    layer_cfg = apply_layer_paths(deepcopy(cfg), 16)
    train(layer_cfg, resume=False)
    selection = select_best_swd(layer_cfg)

    write_loss_summary(cfg, layers=[16])

    summary = json.loads((Path(cfg.paths.logs) / "loss_summary.json").read_text())
    entry = summary["layers"]["16"]
    assert entry["best_step"] == selection["best_step"]
    assert math.isfinite(entry["test_loss"])
