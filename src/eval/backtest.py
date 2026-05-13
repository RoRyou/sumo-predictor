"""Walk-forward backtest for Route A.

For each basho B in the backtest window:
    - train = all features with bashoId < B
    - test  = features with bashoId == B
    - fit a single XGBoost (fast) and report metrics
Aggregate metrics across all backtest basho.

This complements the static train/val/test split in train_struct.py.

CLI::

    python -m src.eval.backtest run \\
        --features data/processed/features.parquet \\
        --start 202301 --out runs/backtest_v1.parquet
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

from src.features.structural import add_stage_features
from src.training.train_struct import (
    CATEGORICAL_COLS,
    LABEL_COL,
    WEIGHT_COL,
    KFoldTargetEncoder,
    feature_cols,
    make_xgb,
)

logger = logging.getLogger(__name__)


def walk_forward(
    features: pd.DataFrame,
    start_basho: str,
    end_basho: str | None = None,
    min_train_basho: int = 6,
    xgb_params: dict | None = None,
) -> pd.DataFrame:
    """Walk forward basho-by-basho.  Returns one summary row per basho."""
    features = features.copy()
    features["bashoId"] = features["bashoId"].astype(str)
    all_basho = sorted(features["bashoId"].unique())
    if len(all_basho) < min_train_basho + 1:
        raise ValueError(f"Need at least {min_train_basho + 1} basho, got {len(all_basho)}")

    target_basho = [b for b in all_basho if b >= start_basho]
    if end_basho is not None:
        target_basho = [b for b in target_basho if b <= end_basho]

    rows: list[dict] = []
    for b in target_basho:
        tr = features[features["bashoId"] < b]
        te = features[features["bashoId"] == b]
        if len(tr) < 500 or len(te) == 0:
            continue
        # T15 stage features (deterministic, leak-free)
        tr = add_stage_features(tr)
        te = add_stage_features(te)
        y_tr = tr[LABEL_COL].to_numpy()
        y_te = te[LABEL_COL].to_numpy()
        w_tr = tr[WEIGHT_COL].to_numpy() if WEIGHT_COL in tr.columns else None

        te_enc = KFoldTargetEncoder(CATEGORICAL_COLS)
        Xtr = te_enc.fit_transform(tr, y_tr)
        Xte = te_enc.transform(te)
        cols = feature_cols(Xtr)
        Xtr = Xtr[cols].fillna(-9999.0)
        Xte = Xte[cols].fillna(-9999.0)

        m = make_xgb(xgb_params or {"n_estimators": 300, "learning_rate": 0.05})
        m.fit(Xtr, y_tr, sample_weight=w_tr)
        p = m.predict_proba(Xte)[:, 1]
        p_c = np.clip(p, 1e-6, 1 - 1e-6)
        row = {
            "basho": b,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "acc": float(accuracy_score(y_te, p > 0.5)),
            "logloss": float(log_loss(y_te, p_c)),
            "auc": float(roc_auc_score(y_te, p)) if len(np.unique(y_te)) > 1 else float("nan"),
            "east_win_rate": float(y_te.mean()),
            "pred_mean": float(p.mean()),
        }
        rows.append(row)
        logger.info(
            "basho %s | n_tr=%d n_te=%d acc=%.4f logloss=%.4f",
            b, row["n_train"], row["n_test"], row["acc"], row["logloss"],
        )
    return pd.DataFrame.from_records(rows)


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 1:
        level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_run(args: argparse.Namespace) -> int:
    features = pd.read_parquet(args.features)
    xgb_params = None
    if args.xgb_params and Path(args.xgb_params).exists():
        with open(args.xgb_params) as f:
            xgb_params = json.load(f)
        logger.info("Using XGB params: %s", xgb_params)
    df = walk_forward(features, args.start, args.end, xgb_params=xgb_params)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    summary = {
        "n_basho": int(len(df)),
        "macro_acc": float(df["acc"].mean()),
        "weighted_acc": float((df["acc"] * df["n_test"]).sum() / df["n_test"].sum())
        if len(df) else float("nan"),
        "macro_logloss": float(df["logloss"].mean()),
        "macro_auc": float(df["auc"].mean()),
        "acc_min": float(df["acc"].min()) if len(df) else float("nan"),
        "acc_max": float(df["acc"].max()) if len(df) else float("nan"),
    }
    print(json.dumps(summary, indent=2))
    logger.info("Wrote %s (%d rows)", out, len(df))
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward backtest")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Run walk-forward backtest")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--start", required=True, help="first basho to predict (e.g. 202301)")
    r.add_argument("--end", default=None, help="last basho to predict (inclusive)")
    r.add_argument("--out", default="runs/backtest.parquet")
    r.add_argument("--xgb-params", default=None, help="path to JSON of XGB best params")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
