from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from repdist.checkpoint import (
    CHECKPOINT_FORMAT,
    CheckpointCompatibilityError,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from repdist.data import Normalizer
from repdist.ddpm import NoisePredictor


def test_checkpoint_roundtrip(tmp_path: Path):
    model = NoisePredictor(8, hidden_dim=16, time_embed_dim=8)
    opt = AdamW(model.parameters(), lr=1e-3)
    hidden = torch.randn(20, 8)
    norm = Normalizer.fit(hidden)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        step=11,
        model=model,
        optimizer=opt,
        scheduler_state=None,
        normalizer=norm,
        best_val=0.5,
        patience_left=3,
    )
    payload = load_checkpoint(path)
    restored = NoisePredictor(8, hidden_dim=16, time_embed_dim=8)
    restored.load_state_dict(payload["model"])
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)
    assert payload["step"] == 11
    assert payload["best_val"] == 0.5
    assert payload["checkpoint_format"] == CHECKPOINT_FORMAT
    assert payload["model_spec"]["full_rank_skip"] is True
    assert payload["normalizer_fingerprint"] == norm.fingerprint()


def test_schema_less_checkpoint_is_rejected(tmp_path: Path):
    path = tmp_path / "old.pt"
    torch.save({"model": {}, "optimizer": {}}, path)

    with pytest.raises(CheckpointCompatibilityError, match="schema-less"):
        load_checkpoint(path)


def test_checkpoint_metadata_mismatch_is_rejected_before_loading():
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_spec": {"architecture": "NoisePredictor", "full_rank_skip": True},
        "schedule_spec": {"timesteps": 1000},
        "normalizer_fingerprint": "abc",
    }

    with pytest.raises(CheckpointCompatibilityError, match="model_spec"):
        validate_checkpoint_compatibility(
            payload,
            model_spec={"architecture": "NoisePredictor", "full_rank_skip": False},
            schedule_spec={"timesteps": 1000},
            normalizer_fingerprint="abc",
        )


def test_checkpoint_normalizer_fingerprint_must_match_embedded_state(tmp_path: Path):
    model = NoisePredictor(8, hidden_dim=16, time_embed_dim=8)
    opt = AdamW(model.parameters(), lr=1e-3)
    norm = Normalizer.fit(torch.randn(20, 8))
    path = tmp_path / "checkpoint-tampered.pt"
    save_checkpoint(
        path,
        step=1,
        model=model,
        optimizer=opt,
        scheduler_state=None,
        normalizer=norm,
        best_val=0.5,
        patience_left=3,
    )
    payload = torch.load(path, weights_only=False)
    payload["normalizer"]["scale"] = 99.0
    torch.save(payload, path)

    with pytest.raises(CheckpointCompatibilityError, match="does not match"):
        load_checkpoint(path)
