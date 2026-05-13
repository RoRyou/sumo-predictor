"""Optuna tuning for the XGBoost base model.

Objective: minimize 5-fold OOF logloss on the TRAIN portion (everything before
``--val-basho``).  Search space matches the task spec:

    max_depth        ∈ [3, 8]
    n_estimators     ∈ [300, 1200]
    learning_rate    log [0.01, 0.1]
    subsample        [0.6, 1.0]
    colsample_bytree [0.6, 1.0]
    reg_lambda       log [0.1, 10]

Budget defaults: ``--max-trials 60``, ``--timeout 1800`` (30 min).

Writes the best params to ``runs/xgb_best_params.json``.

CLI::

    python -m src.training.tune_struct run \\
        --features data/processed/features.parquet \\
        --val-basho 202311 --test-start 202401 \\
        --out runs/xgb_best_params.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from src.features.structural import add_stage_features
from src.training.train_struct import (
    CATEGORICAL_COLS,
    LABEL_COL,
    WEIGHT_COL,
    KFoldTargetEncoder,
    feature_cols,
    make_xgb,
    time_split,
)

logger = logging.getLogger(__name__)


def _prepare_train_matrix(features_path: Path, val_basho: str, test_start: str):
    df = pd.read_parquet(features_path)
    df["bashoId"] = df["bashoId"].astype(str)
    split = time_split(df, val_basho, test_start)
    train = add_stage_features(split.train)
    te = KFoldTargetEncoder(CATEGORICAL_COLS)
    y_tr = train[LABEL_COL].to_numpy()
    Xtr = te.fit_transform(train, y_tr)
    cols = feature_cols(Xtr)
    X = Xtr[cols].fillna(-9999.0)
    w = Xtr[WEIGHT_COL].to_numpy() if WEIGHT_COL in Xtr.columns else np.ones(len(X))
    return X, y_tr, w, cols


def objective_factory(X, y, w, n_splits: int = 5, seed: int = 42):
    import optuna

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(X), dtype=float)
        for tr_idx, va_idx in kf.split(X):
            model = make_xgb(params)
            model.fit(X.iloc[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
            oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        return float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)))

    return objective


def run(features_path: Path, val_basho: str, test_start: str, out: Path,
        max_trials: int, timeout: int) -> dict:
    import optuna

    X, y, w, cols = _prepare_train_matrix(features_path, val_basho, test_start)
    logger.info("Tuning XGB on %d train rows × %d features", len(X), len(cols))
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective_factory(X, y, w),
        n_trials=max_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    logger.info("Best logloss=%.5f", study.best_value)
    logger.info("Best params=%s", study.best_params)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(study.best_params, f, indent=2)
    return study.best_params


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
    best = run(
        features_path=Path(args.features),
        val_basho=args.val_basho,
        test_start=args.test_start,
        out=Path(args.out),
        max_trials=args.max_trials,
        timeout=args.timeout,
    )
    print(json.dumps(best, indent=2))
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optuna tune XGBoost for Route A")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run Optuna study")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--val-basho", required=True)
    r.add_argument("--test-start", required=True)
    r.add_argument("--out", default="runs/xgb_best_params.json")
    r.add_argument("--max-trials", type=int, default=60)
    r.add_argument("--timeout", type=int, default=1800,
                   help="wall-clock seconds (default 1800 = 30 min)")
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
