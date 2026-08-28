# pyright: reportMissingImports=false

from pathlib import Path

import pytest
import torch

from repdist.metrics import covariance_spectrum
from repdist.visualize import RANDOM_LABEL, validate_eval_payload, write_figures


def test_write_figures(tmp_path: Path):
    g = torch.Generator().manual_seed(0)
    train = torch.randn(80, 16, generator=g)
    real = torch.randn(40, 16, generator=g)
    gen = real + 0.1 * torch.randn(40, 16, generator=g)
    random = torch.randn(40, 16, generator=g)
    spec_real = covariance_spectrum(real)
    metrics = {
        "eval_schema_version": 2,
        "comparison_space": "training-normalized",
        "baseline": "standard normal N(0,I)",
        "d_eff_real": 8.0,
        "d_eff_diffusion": 9.0,
        "d_eff_random": 12.0,
        "swd_real_diffusion": 0.2,
        "swd_real_random": 0.8,
        "swd_real_real_split": 0.1,
        "pca_rank": 8,
        "pca_d_eff_real": 5.0,
        "pca_d_eff_diffusion": 5.5,
        "pca_d_eff_random": 7.5,
        "pca_d_eff_real_split": 4.8,
        "pca_swd_real_diffusion": 0.18,
        "pca_swd_real_random": 0.7,
        "pca_swd_real_real_split": 0.09,
        "pca_covariance_relative_error_diffusion": 0.12,
        "pca_covariance_relative_error_random": 0.6,
        "pca_covariance_relative_error_real_split": 0.08,
    }
    log_path = tmp_path / "train.jsonl"
    log_path.write_text(
        '{"step": 1, "split": "train", "loss": 1.0}\n'
        '{"step": 1, "split": "val", "loss": 1.1}\n'
    )
    paths = write_figures(
        train=train,
        real=real,
        generated=gen,
        random=random,
        spec_real=spec_real,
        spec_diff=covariance_spectrum(gen),
        spec_random=covariance_spectrum(random),
        metrics=metrics,
        fig_dir=tmp_path / "fig",
        log_path=log_path,
    )
    assert all(p.exists() for p in paths)
    assert RANDOM_LABEL == "standard normal N(0,I)"


def test_validate_eval_payload_rejects_legacy_gaussian_schema():
    payload = {"gaussian": torch.randn(4, 2), "spectrum_gaussian": torch.ones(2)}
    metrics = {"d_eff_gaussian": 2.0, "swd_real_gaussian": 0.5}

    with pytest.raises(ValueError, match="legacy Gaussian.*rerun.*eval"):
        validate_eval_payload(payload, metrics)
