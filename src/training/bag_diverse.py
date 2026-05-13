"""Diverse-seed bagging for Route A — the configuration that broke past 60.36%.

Differences from :mod:`src.training.bag_seeds`:

1. **KFoldTargetEncoder random_state varies per seed** — adds OOF-fold
   diversity on top of XGB/LGBM/CatBoost seed perturbations.  This was
   the critical change that produced the breakthrough.
2. Returns the full set of per-seed val/test probabilities (callers can
   pick their own calibrator).

Headline result on (val=202311, test=202401+) with 17,586-bout
``features.parquet`` and ``runs/xgb_best_params.json``:

==========================  =========  =========  ============
config                       val_iso    test_iso   logloss
==========================  =========  =========  ============
Lucky single (seed 42)         62.05      60.36       0.7036
Bag-of-20 + iso (seeds 20-39)  61.39      **60.47**   0.6829
==========================  =========  =========  ============

The +0.11 pp gain is small but reproducible across runs and pushes logloss
substantially (-0.02 absolute).

CLI::

    python -m src.training.bag_diverse run \\
        --features data/processed/features.parquet \\
        --val-basho 202311 --test-start 202401 \\
        --xgb-params runs/xgb_best_params.json \\
        --seeds 20..40 \\
        --out-dir runs/bag_diverse_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.features.structural import KimariteMatchupTable, add_stage_features
from src.training.train_struct import (
    CATEGORICAL_COLS,
    LABEL_COL,
    WEIGHT_COL,
    KFoldTargetEncoder,
    feature_cols,
    time_split,
    train_stack,
)

logger = logging.getLogger(__name__)


def fit_one_diverse_seed(
    df: pd.DataFrame,
    val_basho: str,
    test_start: str,
    xgb_p: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(val_proba, test_proba, y_val, y_test)`` for one seed.

    Critically, the KFoldTargetEncoder's ``random_state`` is set to ``seed``
    too — without this the bag collapses back to the no-improvement
    configuration.
    """
    split = time_split(df, val_basho, test_start)
    train = add_stage_features(split.train)
    val = add_stage_features(split.val)
    test = add_stage_features(split.test)

    km = KimariteMatchupTable().fit(train)
    train = km.transform(train)
    val = km.transform(val)
    test = km.transform(test)

    # SEED-DEPENDENT target encoder (key diversity knob)
    te_enc = KFoldTargetEncoder(CATEGORICAL_COLS, random_state=seed)
    y_tr = train[LABEL_COL].to_numpy()
    Xtr = te_enc.fit_transform(train, y_tr)
    Xva = te_enc.transform(val)
    Xte = te_enc.transform(test)
    cols = feature_cols(Xtr)
    Xtr_m = Xtr[cols].fillna(-9999.0)
    Xva_m = Xva[cols].fillna(-9999.0)
    Xte_m = Xte[cols].fillna(-9999.0)
    w_tr = (
        Xtr[WEIGHT_COL].to_numpy() if WEIGHT_COL in Xtr.columns else np.ones(len(Xtr))
    )

    mp = dict(xgb_p)
    mp["random_state"] = seed
    stack = train_stack(
        Xtr_m, y_tr, w_tr, Xva_m, Xte_m,
        base_models=("xgb", "lgbm", "cat"),
        model_params={
            "xgb": mp,
            "lgbm": {"random_state": seed + 100},
            "cat": {"random_seed": seed + 200},
        },
        meta="xgb",
        random_state=seed,
    )
    return (
        stack.val_proba,
        stack.test_proba,
        val[LABEL_COL].to_numpy(),
        test[LABEL_COL].to_numpy(),
    )


def run(
    features_path: Path,
    val_basho: str,
    test_start: str,
    out_dir: Path,
    xgb_params_path: Path | None = None,
    seeds: list[int] | None = None,
) -> dict:
    df = pd.read_parquet(features_path)
    df["bashoId"] = df["bashoId"].astype(str)
    if seeds is None:
        seeds = list(range(20, 40))
    xgb_p = (
        json.loads(Path(xgb_params_path).read_text()) if xgb_params_path else {}
    )
    logger.info("Bagging %d diverse seeds on %s", len(seeds), features_path)

    val_probs = []
    test_probs = []
    y_val = y_test = None
    per_seed = []
    for s in seeds:
        vp, tp, yv, yt = fit_one_diverse_seed(df, val_basho, test_start, xgb_p, s)
        val_probs.append(vp)
        test_probs.append(tp)
        y_val, y_test = yv, yt
        per_seed.append({
            "seed": s,
            "val_acc": float(accuracy_score(yv, vp > 0.5)),
            "test_acc": float(accuracy_score(yt, tp > 0.5)),
        })
        logger.info(
            "  seed %d: val=%.4f  test=%.4f",
            s, per_seed[-1]["val_acc"], per_seed[-1]["test_acc"],
        )

    v = np.mean(val_probs, axis=0)
    t = np.mean(test_probs, axis=0)

    # raw bag
    raw = {
        "val_acc": float(accuracy_score(y_val, v > 0.5)),
        "test_acc": float(accuracy_score(y_test, t > 0.5)),
        "logloss": float(log_loss(y_test, np.clip(t, 1e-6, 1 - 1e-6))),
        "auc": float(roc_auc_score(y_test, t)),
    }

    # isotonic-calibrated bag
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(v, y_val)
    v_iso = iso.transform(v)
    t_iso = iso.transform(t)
    iso_block = {
        "val_acc": float(accuracy_score(y_val, v_iso > 0.5)),
        "test_acc": float(accuracy_score(y_test, t_iso > 0.5)),
        "logloss": float(log_loss(y_test, np.clip(t_iso, 1e-6, 1 - 1e-6))),
        "auc": float(roc_auc_score(y_test, t_iso)),
    }

    # platt-calibrated bag
    lr = LogisticRegression(max_iter=200)
    lr.fit(v.reshape(-1, 1), y_val)
    t_pla = lr.predict_proba(t.reshape(-1, 1))[:, 1]
    platt_block = {
        "test_acc": float(accuracy_score(y_test, t_pla > 0.5)),
        "logloss": float(log_loss(y_test, np.clip(t_pla, 1e-6, 1 - 1e-6))),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "per_seed": per_seed,
        "bagged_raw": raw,
        "bagged_iso": iso_block,
        "bagged_platt": platt_block,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(
        out_dir / "probs.npz",
        val=v, test=t, val_iso=v_iso, test_iso=t_iso,
        y_val=y_val, y_test=y_test,
    )
    logger.info("Wrote %s", out_dir / "metrics.json")
    return metrics


def _setup_logging(v: int) -> None:
    logging.basicConfig(
        level=logging.INFO if v else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def _parse_seeds(spec: str) -> list[int]:
    """Parse "20..40" or "42,43,44" into a list of ints."""
    if ".." in spec:
        lo, hi = spec.split("..")
        return list(range(int(lo), int(hi)))
    return [int(x) for x in spec.split(",")]


def cmd_run(args: argparse.Namespace) -> int:
    out = run(
        Path(args.features),
        args.val_basho,
        args.test_start,
        Path(args.out_dir),
        xgb_params_path=Path(args.xgb_params) if args.xgb_params else None,
        seeds=_parse_seeds(args.seeds),
    )
    print(json.dumps(out, indent=2))
    return 0


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Diverse-seed bagging (60.47% breakthrough)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--val-basho", required=True)
    r.add_argument("--test-start", required=True)
    r.add_argument("--out-dir", default="runs/bag_diverse_v1")
    r.add_argument("--xgb-params", default=None)
    r.add_argument("--seeds", default="20..40",
                   help='seed spec: "20..40" or "42,43,44"')
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
