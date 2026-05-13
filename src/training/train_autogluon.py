"""AutoGluon Tabular sanity-check trainer for the sumo bout predictor.

Trains a `best_quality` AutoGluon ensemble on the structural features and
reports val/test metrics. Used to gauge whether the current 60.36% test
plateau from `train_struct.py` is tuning-limited or data-limited.

CLI::

    python -m src.training.train_autogluon run \
        --features data/processed/features.parquet \
        --val-basho 202311 --test-start 202401 \
        --time-limit 14400 --out-dir runs/autogluon_v1
"""
from __future__ import annotations

import os

# IMPORTANT (macOS arm64): force lightgbm + its libomp to load BEFORE AutoGluon
# imports anything that links a second OpenMP runtime (scipy/sklearn/torch all
# can pull in a different libomp). Without this preload, the first LightGBM
# training step segfaults inside __kmp_suspend_initialize_thread. See run logs
# 2026-05-13 for the prior crashes that motivated this.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import lightgbm  # noqa: F401, E402  (preload to fix libomp double-init)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score  # noqa: E402

logger = logging.getLogger(__name__)


# Columns that must never become features.
ID_COLS = ["bashoId", "day", "matchNo", "eastId", "westId", "kimarite"]
LABEL_COL = "y"
WEIGHT_COL = "sample_weight"
CATEGORICAL_COLS = ["heya_A", "heya_B", "shusshin_A", "shusshin_B"]


def _split(df: pd.DataFrame, val_basho: int, test_start: int):
    basho = pd.to_numeric(df["bashoId"], errors="coerce").astype("Int64")
    train = df[basho < val_basho].copy()
    val = df[basho == val_basho].copy()
    test = df[basho >= test_start].copy()
    return train, val, test


def _strip_ids(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in ID_COLS if c in df.columns]
    return df.drop(columns=drop)


def _ensure_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "n": int(len(y_true)),
    }


def run(args: argparse.Namespace) -> None:
    from autogluon.tabular import TabularPredictor  # heavy import deferred

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features from %s", args.features)
    df = pd.read_parquet(args.features)
    logger.info("Loaded %d rows / %d cols", len(df), df.shape[1])

    train_df, val_df, test_df = _split(df, args.val_basho, args.test_start)
    logger.info(
        "Split sizes: train=%d  val(basho %d)=%d  test(basho >= %d)=%d",
        len(train_df), args.val_basho, len(val_df), args.test_start, len(test_df),
    )

    train_df = _ensure_categorical(_strip_ids(train_df))
    val_df = _ensure_categorical(_strip_ids(val_df))
    test_df = _ensure_categorical(_strip_ids(test_df))

    # AutoGluon can ingest sample weights via the `sample_weight` column name +
    # `sample_weight='sample_weight'` arg. The column is then NOT treated as a
    # feature.
    sw_kw = {}
    if WEIGHT_COL in train_df.columns:
        sw_kw = {"sample_weight": WEIGHT_COL, "weight_evaluation": False}

    predictor = TabularPredictor(
        label=LABEL_COL,
        problem_type="binary",
        eval_metric="log_loss",
        path=str(out_dir),
        **sw_kw,
    )

    logger.info("Starting AutoGluon fit: preset=best_quality, time_limit=%d", args.time_limit)
    # NOTE: on macOS arm64 we hit libomp double-init segfaults when AutoGluon
    # imports torch alongside LightGBM in the same process (PyTorch's bundled
    # libomp vs LightGBM's libomp). Excluding the torch-based learners avoids
    # the conflict; LightGBM, CatBoost, XGBoost, RF, XT and the linear KNN/LR
    # baselines still give a strong best_quality ensemble.
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        use_bag_holdout=True,
        presets=["best_quality"],
        time_limit=args.time_limit,
        verbosity=2,
        excluded_model_types=["NN_TORCH", "FASTAI"],
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    logger.info("AutoGluon training complete.")

    # Leaderboard
    lb = predictor.leaderboard(silent=True)
    lb_path = out_dir / "leaderboard.csv"
    lb.to_csv(lb_path, index=False)
    logger.info("Wrote leaderboard to %s", lb_path)

    # Predict on val + test (probability of class 1)
    def _proba(d: pd.DataFrame) -> np.ndarray:
        feat = d.drop(columns=[LABEL_COL] + ([WEIGHT_COL] if WEIGHT_COL in d.columns else []))
        proba = predictor.predict_proba(feat)
        # binary: dataframe with two columns; we want P(y=1)
        if isinstance(proba, pd.DataFrame):
            col = 1 if 1 in proba.columns else proba.columns[-1]
            return proba[col].to_numpy()
        return np.asarray(proba)

    metrics: dict = {}
    if len(val_df) > 0:
        y_val = val_df[LABEL_COL].to_numpy()
        p_val = _proba(val_df)
        metrics["val"] = _metrics(y_val, p_val)
        logger.info("Val: %s", metrics["val"])

    if len(test_df) > 0:
        y_test = test_df[LABEL_COL].to_numpy()
        p_test = _proba(test_df)
        metrics["test"] = _metrics(y_test, p_test)
        logger.info("Test: %s", metrics["test"])

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote metrics to %s", metrics_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train AutoGluon Tabular sanity-check model.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Run AutoGluon training")
    r.add_argument("--features", type=Path, required=True)
    r.add_argument("--val-basho", type=int, required=True)
    r.add_argument("--test-start", type=int, required=True)
    r.add_argument("--time-limit", type=int, default=14400)
    r.add_argument("--out-dir", type=Path, default=Path("runs/autogluon_v1"))
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
