from pathlib import Path

import torch
from torch.optim import AdamW

from repdist.checkpoint import load_checkpoint, save_checkpoint
from repdist.model import NoisePredictor
from repdist.normalize import Normalizer


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
