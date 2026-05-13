"""Seed-bagging wrapper around the existing stack.

Trains the same stacking pipeline N times with different random seeds and
averages the test/val probabilities.  Free variance reduction; typically
worth +0.2–0.5 pp on a tabular GBDT stack.

Usage::

    python -m src.training.bag_seeds run \\
        --features data/processed/features.parquet \\
        --val-basho 202311 --test-start 202401 \\
        --xgb-params runs/xgb_best_params.json \\
        --out-dir runs/bag_seeds_v1 --n-bags 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.features.structural import (
    KimariteMatchupTable,
    add_stage_features,
    symmetric_augment,
)
from src.training.train_struct import (
    BASE_MODEL_BUILDERS,
    CATEGORICAL_COLS,
    LABEL_COL,
    WEIGHT_COL,
    KFoldTargetEncoder,
    feature_cols,
    time_split,
    train_stack,
)

logger = logging.getLogger(__name__)


def _fit_bag(
    df: pd.DataFrame,
    val_basho: str,
    test_start: str,
    xgb_params_path: Path | None,
    seed: int,
):
    """One bag: fits the stack once with a given seed.  Returns (val_proba, test_proba, y_val, y_test)."""
    split = time_split(df, val_basho, test_start)
    tr_df = add_stage_features(split.train)
    va_df = add_stage_features(split.val)
    te_df = add_stage_features(split.test)
    km = KimariteMatchupTable().fit(tr_df)
    tr_df = km.transform(tr_df)
    va_df = km.transform(va_df)
    te_df = km.transform(te_df)

    te_enc = KFoldTargetEncoder(CATEGORICAL_COLS, random_state=seed)
    y_tr = tr_df[LABEL_COL].to_numpy()
    Xtr_full = te_enc.fit_transform(tr_df, y_tr)
    Xva_full = te_enc.transform(va_df)
    Xte_full = te_enc.transform(te_df)
    cols = feature_cols(Xtr_full)
    Xtr = Xtr_full[cols].fillna(-9999.0)
    Xva = Xva_full[cols].fillna(-9999.0)
    Xte = Xte_full[cols].fillna(-9999.0)
    w_tr = Xtr_full[WEIGHT_COL].to_numpy() if WEIGHT_COL in Xtr_full.columns else np.ones(len(Xtr))
    y_val = va_df[LABEL_COL].to_numpy()
    y_test = te_df[LABEL_COL].to_numpy()

    # Per-seed model param overrides
    model_params = {}
    if xgb_params_path is not None and Path(xgb_params_path).exists():
        with open(xgb_params_path) as f:
            xgb_p = json.load(f)
        xgb_p["random_state"] = seed
        model_params["xgb"] = xgb_p
    # LGBM / Cat seed perturbation
    model_params["lgbm"] = {"random_state": seed + 100}
    model_params["cat"] = {"random_seed": seed + 200}

    stack = train_stack(
        Xtr, y_tr, w_tr, Xva, Xte,
        base_models=("xgb", "lgbm", "cat"),
        model_params=model_params,
        meta="xgb",
        random_state=seed,
    )
    return stack.val_proba, stack.test_proba, y_val, y_test


def run(
    features_path: Path,
    val_basho: str,
    test_start: str,
    out_dir: Path,
    xgb_params_path: Path | None = None,
    n_bags: int = 5,
    seeds: list[int] | None = None,
) -> dict:
    df = pd.read_parquet(features_path)
    df["bashoId"] = df["bashoId"].astype(str)
    if seeds is None:
        seeds = list(range(42, 42 + n_bags))
    logger.info("Bagging %d seeds: %s", len(seeds), seeds)

    val_probs: list[np.ndarray] = []
    test_probs: list[np.ndarray] = []
    y_val = y_test = None
    per_bag = []
    for i, s in enumerate(seeds):
        logger.info("=== Bag %d/%d (seed=%d) ===", i + 1, len(seeds), s)
        vp, tp, yv, yt = _fit_bag(df, val_basho, test_start, xgb_params_path, s)
        val_probs.append(vp)
        test_probs.append(tp)
        y_val, y_test = yv, yt
        per_bag.append({
            "seed": s,
            "val_acc": float(accuracy_score(yv, vp > 0.5)),
            "test_acc": float(accuracy_score(yt, tp > 0.5)),
            "val_logloss": float(log_loss(yv, np.clip(vp, 1e-6, 1 - 1e-6))),
            "test_logloss": float(log_loss(yt, np.clip(tp, 1e-6, 1 - 1e-6))),
        })
        logger.info(
            "bag %d: val_acc=%.4f test_acc=%.4f",
            s, per_bag[-1]["val_acc"], per_bag[-1]["test_acc"],
        )

    val_mean = np.mean(val_probs, axis=0)
    test_mean = np.mean(test_probs, axis=0)

    out = {
        "per_bag": per_bag,
        "bagged_val": {
            "acc": float(accuracy_score(y_val, val_mean > 0.5)),
            "logloss": float(log_loss(y_val, np.clip(val_mean, 1e-6, 1 - 1e-6))),
            "auc": float(roc_auc_score(y_val, val_mean)),
        },
        "bagged_test": {
            "acc": float(accuracy_score(y_test, test_mean > 0.5)),
            "logloss": float(log_loss(y_test, np.clip(test_mean, 1e-6, 1 - 1e-6))),
            "auc": float(roc_auc_score(y_test, test_mean)),
        },
        "n_bags": len(seeds),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------- #
def _setup_logging(v: int) -> None:
    logging.basicConfig(
        level=logging.INFO if v else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_run(args: argparse.Namespace) -> int:
    out = run(
        Path(args.features), args.val_basho, args.test_start,
        Path(args.out_dir),
        xgb_params_path=Path(args.xgb_params) if args.xgb_params else None,
        n_bags=args.n_bags,
    )
    print(json.dumps(out, indent=2))
    return 0


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Seed-bagged stack")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--val-basho", required=True)
    r.add_argument("--test-start", required=True)
    r.add_argument("--out-dir", default="runs/bag_seeds_v1")
    r.add_argument("--xgb-params", default=None)
    r.add_argument("--n-bags", type=int, default=5)
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
