"""Train LR / CB / MLP with multi-basho val for honest ensemble.
Each model trained on bashoId < 202307, val = {202307,202309,202311}, test = >=202401.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from category_encoders import TargetEncoder
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features_v4.parquet")
    ap.add_argument("--train-end", default="202307")
    ap.add_argument("--val-bashos", nargs="+", default=["202307","202309","202311"])
    ap.add_argument("--test-start", default="202401")
    ap.add_argument("--out-prefix", default="runs/aux_mb")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    df["bashoId"] = df["bashoId"].astype(str)
    train = df[df["bashoId"] < args.train_end].copy()
    val = df[df["bashoId"].isin(args.val_bashos)].copy()
    test = df[df["bashoId"] >= args.test_start].copy()
    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")

    DROP = {'bashoId','day','matchNo','eastId','westId','y','kimarite','sample_weight',
            'heya_A','heya_B','shusshin_A','shusshin_B'}
    CAT = ['heya_A','heya_B','shusshin_A','shusshin_B']
    NUM = [c for c in df.columns if c not in DROP and df[c].dtype != 'object']

    te = TargetEncoder(cols=CAT, smoothing=10)
    y_tr = train['y'].astype(int).values
    te.fit(train[CAT], y_tr)
    def X_te_(d):
        return pd.concat([d[NUM].reset_index(drop=True),
                          te.transform(d[CAT]).add_prefix('te__').reset_index(drop=True)],
                         axis=1).fillna(-9999.0)
    X_tr = X_te_(train); X_va = X_te_(val); X_te = X_te_(test)
    y_va = val['y'].astype(int).values
    y_te = test['y'].astype(int).values

    # Fill -9999 sentinel with column means for LR/MLP
    X_tr_clean = X_tr.copy(); X_va_clean = X_va.copy(); X_te_clean = X_te.copy()
    for col in X_tr_clean.columns:
        mean_val = X_tr_clean[col].replace(-9999.0, np.nan).mean()
        if not np.isnan(mean_val):
            for d in [X_tr_clean, X_va_clean, X_te_clean]:
                d.loc[d[col] == -9999.0, col] = mean_val

    scaler = StandardScaler().fit(X_tr_clean)
    X_tr_s = scaler.transform(X_tr_clean).astype(np.float32)
    X_va_s = scaler.transform(X_va_clean).astype(np.float32)
    X_te_s = scaler.transform(X_te_clean).astype(np.float32)

    # ===== LR =====
    print("\n=== LR ===")
    lr = LogisticRegression(C=0.0001, max_iter=3000, random_state=42)
    lr.fit(X_tr_s, y_tr, sample_weight=train['sample_weight'].values)
    p_va = lr.predict_proba(X_va_s)[:, 1]
    p_te = lr.predict_proba(X_te_s)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip'); iso.fit(p_va, y_va)
    v_iso = iso.transform(p_va); t_iso = iso.transform(p_te)
    print(f"LR-mb: raw val_acc={accuracy_score(y_va, p_va>0.5):.4f} val_auc={roc_auc_score(y_va, p_va):.4f} test_acc={accuracy_score(y_te, p_te>0.5):.4f}")
    print(f"       iso val_acc={accuracy_score(y_va, v_iso>0.5):.4f} val_auc={roc_auc_score(y_va, v_iso):.4f} test_acc={accuracy_score(y_te, t_iso>0.5):.4f}")
    np.savez(f"{args.out_prefix}_lr.npz", val=p_va, test=p_te, val_iso=v_iso, test_iso=t_iso, y_val=y_va, y_test=y_te)

    # ===== MLP (3-seed bag) =====
    print("\n=== MLP-3seed ===")
    val_probs, test_probs = [], []
    for seed in [42, 43, 44]:
        m = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.0001, max_iter=300,
                          random_state=seed, early_stopping=False, learning_rate_init=0.001)
        m.fit(X_tr_s, y_tr)
        val_probs.append(m.predict_proba(X_va_s)[:, 1])
        test_probs.append(m.predict_proba(X_te_s)[:, 1])
    p_va = np.mean(val_probs, axis=0); p_te = np.mean(test_probs, axis=0)
    iso = IsotonicRegression(out_of_bounds='clip'); iso.fit(p_va, y_va)
    v_iso = iso.transform(p_va); t_iso = iso.transform(p_te)
    print(f"MLP-mb: raw val_auc={roc_auc_score(y_va, p_va):.4f} test_acc={accuracy_score(y_te, p_te>0.5):.4f}")
    print(f"        iso val_auc={roc_auc_score(y_va, v_iso):.4f} test_acc={accuracy_score(y_te, t_iso>0.5):.4f}")
    np.savez(f"{args.out_prefix}_mlp.npz", val=p_va, test=p_te, val_iso=v_iso, test_iso=t_iso, y_val=y_va, y_test=y_te)

    # ===== MLP (10-seed bag, early stopping) =====
    print("\n=== MLP-10seed ===")
    val_probs, test_probs = [], []
    for seed in range(42, 52):
        m = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.0001, max_iter=300,
                          random_state=seed, early_stopping=True, validation_fraction=0.1,
                          learning_rate_init=0.001, batch_size=128)
        m.fit(X_tr_s, y_tr)
        val_probs.append(m.predict_proba(X_va_s)[:, 1])
        test_probs.append(m.predict_proba(X_te_s)[:, 1])
    p_va = np.mean(val_probs, axis=0); p_te = np.mean(test_probs, axis=0)
    iso = IsotonicRegression(out_of_bounds='clip'); iso.fit(p_va, y_va)
    v_iso = iso.transform(p_va); t_iso = iso.transform(p_te)
    print(f"MLP10-mb: raw val_auc={roc_auc_score(y_va, p_va):.4f} test_acc={accuracy_score(y_te, p_te>0.5):.4f}")
    print(f"          iso val_auc={roc_auc_score(y_va, v_iso):.4f} test_acc={accuracy_score(y_te, t_iso>0.5):.4f}")
    np.savez(f"{args.out_prefix}_mlp10.npz", val=p_va, test=p_te, val_iso=v_iso, test_iso=t_iso, y_val=y_va, y_test=y_te)

    # ===== CatBoost native cat =====
    print("\n=== CatBoost native cat ===")
    train2 = train.copy()
    val2 = val.copy()
    test2 = test.copy()
    for d in [train2, val2, test2]:
        d['eastId_cat'] = d['eastId'].astype(str)
        d['westId_cat'] = d['westId'].astype(str)
        d['heya_A_cat'] = d['heya_A'].astype(str)
        d['heya_B_cat'] = d['heya_B'].astype(str)
        d['shusshin_A_cat'] = d['shusshin_A'].astype(str)
        d['shusshin_B_cat'] = d['shusshin_B'].astype(str)
    CAT_COLS = ['eastId_cat','westId_cat','heya_A_cat','heya_B_cat','shusshin_A_cat','shusshin_B_cat']
    NUM_CB = [c for c in df.columns if c not in DROP and c not in CAT_COLS and df[c].dtype != 'object']
    features = NUM_CB + CAT_COLS
    X_tr_cb = train2[features].fillna(-9999.0)
    X_va_cb = val2[features].fillna(-9999.0)
    X_te_cb = test2[features].fillna(-9999.0)
    cat_indices = [features.index(c) for c in CAT_COLS]

    val_probs, test_probs = [], []
    for seed in [42, 43, 44, 45, 46]:
        m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.03,
                                l2_leaf_reg=3.0, random_seed=seed,
                                cat_features=cat_indices, verbose=0)
        m.fit(X_tr_cb, y_tr, sample_weight=train['sample_weight'].values)
        val_probs.append(m.predict_proba(X_va_cb)[:, 1])
        test_probs.append(m.predict_proba(X_te_cb)[:, 1])
    p_va = np.mean(val_probs, axis=0); p_te = np.mean(test_probs, axis=0)
    iso = IsotonicRegression(out_of_bounds='clip'); iso.fit(p_va, y_va)
    v_iso = iso.transform(p_va); t_iso = iso.transform(p_te)
    print(f"CB-mb: raw val_auc={roc_auc_score(y_va, p_va):.4f} test_acc={accuracy_score(y_te, p_te>0.5):.4f}")
    print(f"        iso val_auc={roc_auc_score(y_va, v_iso):.4f} test_acc={accuracy_score(y_te, t_iso>0.5):.4f}")
    np.savez(f"{args.out_prefix}_cb.npz", val=p_va, test=p_te, val_iso=v_iso, test_iso=t_iso, y_val=y_va, y_test=y_te)


if __name__ == "__main__":
    main()
