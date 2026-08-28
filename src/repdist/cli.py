from __future__ import annotations

import argparse
from pathlib import Path

from repdist.config import ExperimentConfig
from repdist.evaluate import evaluate, visualize_from_saved
from repdist.extract import ensure_heldout, ensure_train
from repdist.store import HiddenStateStore
from repdist.train import train


def _config(path: str) -> ExperimentConfig:
    return ExperimentConfig.load(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="repdist")
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="extract hidden states")
    p_extract.add_argument("--train-only", action="store_true")

    p_train = sub.add_parser("train", help="train DDPM")
    p_train.add_argument("--no-resume", action="store_true")

    p_eval = sub.add_parser("eval", help="evaluate held-out matching")
    p_eval.add_argument("--ckpt", default="best", choices=["best", "latest"])

    sub.add_parser(
        "viz",
        help="write memo figures from saved normalized eval tensors or by re-running eval",
    )
    sub.add_parser("all", help="extract, train, evaluate")

    args = parser.parse_args(argv)
    cfg = _config(args.config)
    for rel in (
        cfg.paths.data,
        cfg.paths.checkpoints,
        cfg.paths.logs,
        cfg.paths.outputs,
    ):
        Path(rel).mkdir(parents=True, exist_ok=True)

    if args.cmd == "extract":
        store = HiddenStateStore(cfg.paths.data)
        device = cfg.resolve_device()
        if not args.train_only:
            ensure_heldout(cfg, store, device)
        ensure_train(cfg, store, device)
        print(store.manifest)
        return 0

    if args.cmd == "train":
        result = train(cfg, resume=not args.no_resume)
        print(result)
        return 0

    if args.cmd == "eval":
        metrics = evaluate(cfg, ckpt_name=args.ckpt)
        print(metrics)
        return 0

    if args.cmd == "viz":
        saved = Path(cfg.paths.outputs) / "metrics" / "eval_tensors.pt"
        if saved.exists():
            metrics = visualize_from_saved(cfg)
        else:
            metrics = evaluate(cfg, ckpt_name="best")
        print(metrics)
        print(f"figures -> {cfg.paths.outputs}/figures")
        return 0

    if args.cmd == "all":
        store = HiddenStateStore(cfg.paths.data)
        device = cfg.resolve_device()
        ensure_train(cfg, store, device)
        print(train(cfg, resume=True))
        print(evaluate(cfg, ckpt_name="best"))
        return 0

    raise AssertionError(args.cmd)
