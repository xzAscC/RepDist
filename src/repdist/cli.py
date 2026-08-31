from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from repdist.config import ExperimentConfig
from repdist.evaluate import compare_layer_runs, evaluate, visualize_from_saved
from repdist.extract import ensure_heldout, ensure_train, layer_stores
from repdist.layers import apply_layer_paths, resolved_layers
from repdist.store import HiddenStateStore
from repdist.train import train, train_with_lr_retry


def _load_cfg(path: str) -> ExperimentConfig:
    return ExperimentConfig.load(path)


def _layer_cfgs(cfg: ExperimentConfig, layer: int | None) -> list[ExperimentConfig]:
    if layer is not None:
        return [apply_layer_paths(deepcopy(cfg), layer)]
    if cfg.extract.layers:
        return [apply_layer_paths(deepcopy(cfg), item) for item in resolved_layers(cfg)]
    return [cfg]


def _ensure_dirs(cfg: ExperimentConfig) -> None:
    for rel in (
        cfg.paths.data,
        cfg.paths.checkpoints,
        cfg.paths.logs,
        cfg.paths.outputs,
    ):
        Path(rel).mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="repdist")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="restrict train/eval/viz to one extracted layer",
    )
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
    p_cmp = sub.add_parser(
        "compare-layers",
        help="aggregate per-layer eval.json files into comparison tables and figures",
    )
    p_cmp.add_argument("--root", default="outputs")
    p_cmp.add_argument("--out", default="results")

    args = parser.parse_args(argv)
    if args.cmd == "compare-layers":
        comparison = compare_layer_runs(args.root, args.out)
        print(comparison["beats_random_swd"])
        return 0

    cfg = _load_cfg(args.config)
    if args.cmd == "extract" and args.layer is not None:
        parser.error("--layer applies to train/eval/viz, not extract")

    if args.cmd == "extract":
        _ensure_dirs(cfg)
        store = HiddenStateStore(cfg.paths.data)
        device = cfg.resolve_device()
        if not args.train_only:
            ensure_heldout(cfg, store, device)
        ensure_train(cfg, store, device)
        print({k: v.manifest for k, v in layer_stores(cfg, store).items()})
        return 0

    if args.cmd == "train":
        result = None
        for layer_cfg in _layer_cfgs(cfg, args.layer):
            _ensure_dirs(layer_cfg)
            result = train_with_lr_retry(layer_cfg, resume=not args.no_resume)
            print(result)
        return 0

    if args.cmd == "eval":
        metrics = None
        for layer_cfg in _layer_cfgs(cfg, args.layer):
            _ensure_dirs(layer_cfg)
            metrics = evaluate(layer_cfg, ckpt_name=args.ckpt)
            print(metrics)
        return 0

    if args.cmd == "viz":
        metrics = None
        for layer_cfg in _layer_cfgs(cfg, args.layer):
            _ensure_dirs(layer_cfg)
            saved = Path(layer_cfg.paths.outputs) / "metrics" / "eval_tensors.pt"
            if saved.exists():
                metrics = visualize_from_saved(layer_cfg)
            else:
                metrics = evaluate(layer_cfg, ckpt_name="best")
            print(metrics)
            print(f"figures -> {layer_cfg.paths.outputs}/figures")
        return 0

    if args.cmd == "all":
        _ensure_dirs(cfg)
        store = HiddenStateStore(cfg.paths.data)
        device = cfg.resolve_device()
        ensure_train(cfg, store, device)
        for layer_cfg in _layer_cfgs(cfg, args.layer):
            _ensure_dirs(layer_cfg)
            print(train_with_lr_retry(layer_cfg, resume=True))
            print(evaluate(layer_cfg, ckpt_name="best"))
        return 0

    raise AssertionError(args.cmd)
