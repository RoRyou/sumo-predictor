"""Bag-of-N diverse-seed stack with MULTI-BASHO validation.

Train cutoff: bashoId < 202307
Val: bashoId in {202307, 202309, 202311} (≈900 rows, 3x current val size)
Test: bashoId >= 202401

For each seed:
  1. Train XGB+LGBM+CatBoost stack with KFoldTargetEncoder(random_state=seed)
  2. Predict on val (concatenated) and test
Mean across seeds; apply isotonic and Platt on the multi-basho val.
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
    CATEGORICAL_COLS, LABEL_COL, WEIGHT_COL,
    KFoldTargetEncoder, feature_cols, train_stack,
)

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_v4.parquet")
    p.add_argument("--train-end", default="202307",
                   help="train: bashoId < this (exclusive)")
    p.add_argument("--val-bashos", nargs="+", default=["202307", "202309", "202311"])
    p.add_argument("--test-start", default="202401")
    p.add_argument("--xgb-params", default="runs/xgb_best_params.json")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=20)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stderr)

    df = pd.read_parquet(args.features)
    df["bashoId"] = df["bashoId"].astype(str)

    train = df[df["bashoId"] < args.train_end].reset_index(drop=True)
    val = df[df["bashoId"].isin(args.val_bashos)].reset_index(drop=True)
    test = df[df["bashoId"] >= args.test_start].reset_index(drop=True)
    logger.info(f"train: {len(train)}  val: {len(val)} ({args.val_bashos})  test: {len(test)}")

    xgb_p = json.loads(Path(args.xgb_params).read_text()) if args.xgb_params else {}

    val_probs, test_probs = [], []
    per_seed = []
    for s in range(args.seed_start, args.seed_start + args.seeds):
        # Add stage features (T15)
        tr_s = add_stage_features(train.copy())
        va_s = add_stage_features(val.copy())
        te_s = add_stage_features(test.copy())

        # Kimarite matchup table (T7)
        km = KimariteMatchupTable().fit(tr_s)
        tr_s = km.transform(tr_s)
        va_s = km.transform(va_s)
        te_s = km.transform(te_s)

        # KFold TE with per-seed randomness
        te_enc = KFoldTargetEncoder(CATEGORICAL_COLS, random_state=s)
        y_tr = tr_s[LABEL_COL].to_numpy()
        Xtr = te_enc.fit_transform(tr_s, y_tr)
        Xva = te_enc.transform(va_s)
        Xte = te_enc.transform(te_s)
        cols = feature_cols(Xtr)
        Xtr_m = Xtr[cols].fillna(-9999.0)
        Xva_m = Xva[cols].fillna(-9999.0)
        Xte_m = Xte[cols].fillna(-9999.0)
        w_tr = Xtr[WEIGHT_COL].to_numpy() if WEIGHT_COL in Xtr.columns else np.ones(len(Xtr))

        mp = dict(xgb_p); mp["random_state"] = s
        stack = train_stack(
            Xtr_m, y_tr, w_tr, Xva_m, Xte_m,
            base_models=("xgb", "lgbm", "cat"),
            model_params={
                "xgb": mp,
                "lgbm": {"random_state": s + 100},
                "cat": {"random_seed": s + 200},
            },
            meta="xgb", random_state=s,
        )
        vp = stack.val_proba
        tp = stack.test_proba
        y_va = va_s[LABEL_COL].to_numpy()
        y_te = te_s[LABEL_COL].to_numpy()
        per_seed.append({'seed': s, 'val_acc': float(accuracy_score(y_va, vp>0.5)),
                          'test_acc': float(accuracy_score(y_te, tp>0.5))})
        logger.info(f"  seed {s}: val_acc={per_seed[-1]['val_acc']:.4f} test_acc={per_seed[-1]['test_acc']:.4f}")
        val_probs.append(vp)
        test_probs.append(tp)

    v = np.mean(val_probs, axis=0)
    t = np.mean(test_probs, axis=0)

    raw = {
        'val_acc': float(accuracy_score(y_va, v>0.5)),
        'test_acc': float(accuracy_score(y_te, t>0.5)),
        'val_auc': float(roc_auc_score(y_va, v)),
        'val_ll': float(log_loss(y_va, np.clip(v, 1e-6, 1-1e-6))),
        'logloss': float(log_loss(y_te, np.clip(t, 1e-6, 1-1e-6))),
        'auc': float(roc_auc_score(y_te, t)),
    }
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(v, y_va)
    v_iso = iso.transform(v); t_iso = iso.transform(t)
    iso_d = {
        'val_acc': float(accuracy_score(y_va, v_iso>0.5)),
        'test_acc': float(accuracy_score(y_te, t_iso>0.5)),
        'val_auc': float(roc_auc_score(y_va, v_iso)),
        'val_ll': float(log_loss(y_va, np.clip(v_iso, 1e-6, 1-1e-6))),
        'logloss': float(log_loss(y_te, np.clip(t_iso, 1e-6, 1-1e-6))),
        'auc': float(roc_auc_score(y_te, t_iso)),
    }
    lr = LogisticRegression(max_iter=200)
    lr.fit(v.reshape(-1, 1), y_va)
    v_pl = lr.predict_proba(v.reshape(-1, 1))[:, 1]
    t_pl = lr.predict_proba(t.reshape(-1, 1))[:, 1]
    pl_d = {
        'val_acc': float(accuracy_score(y_va, v_pl>0.5)),
        'test_acc': float(accuracy_score(y_te, t_pl>0.5)),
        'logloss': float(log_loss(y_te, np.clip(t_pl, 1e-6, 1-1e-6))),
    }

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'probs.npz',
             val=v, test=t, val_iso=v_iso, test_iso=t_iso, val_platt=v_pl, test_platt=t_pl,
             y_val=y_va, y_test=y_te,
             val_bashos=np.array(va_s['bashoId'].values, dtype='U10'))
    (out / 'metrics.json').write_text(json.dumps({
        'config': vars(args), 'per_seed': per_seed,
        'bagged_raw': raw, 'bagged_iso': iso_d, 'bagged_platt': pl_d,
    }, indent=2))
    logger.info(f"Saved to {out}")
    print(f"\nMulti-basho bag (n_val={len(v)}, train_end={args.train_end}):")
    print(f"  raw: val_acc={raw['val_acc']:.4f} val_auc={raw['val_auc']:.4f} test_acc={raw['test_acc']:.4f} test_auc={raw['auc']:.4f}")
    print(f"  iso: val_acc={iso_d['val_acc']:.4f} val_auc={iso_d['val_auc']:.4f} test_acc={iso_d['test_acc']:.4f} test_auc={iso_d['auc']:.4f}")


if __name__ == "__main__":
    main()
