"""5-fold CV training of PoseTowerClassifier on REAL per-frame sequences.

Trains the Bi-LSTM × 2 + Attention Pool model from
:mod:`src.models.temporal` on the 83 aligned bouts whose per-frame
40-dim kinematic features were produced by
:mod:`src.features.extract_perframe_features`.

Saves out-of-fold (OOF) probabilities aligned with the row order of
``data/processed/pose_features_aligned.parquet`` so the downstream
hybrid script can splice them into the test-set probability vector.

CLI
---
::

    python -m src.training.train_pose_real run \\
        --perframe data/processed/pose_perframe.parquet \\
        --aligned data/processed/pose_features_aligned.parquet \\
        --out runs/pose_tower_oof.npy \\
        --T 75 --hidden 64 --embed-dim 64 --dropout 0.3 \\
        --epochs 50 --patience 8 --batch-size 8 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from src.features.kinematics import FEATURE_NAMES
from src.models.temporal import PoseTowerClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class BoutPoseDataset(Dataset):
    """One sample == one bout sequence (T_max, 40) + label."""

    def __init__(
        self,
        seqs: list[np.ndarray],
        labels: np.ndarray,
        T: int = 75,
    ) -> None:
        self.seqs = seqs
        self.labels = labels.astype(np.float32)
        self.T = T

    def __len__(self) -> int:
        return len(self.seqs)

    def _pad_or_truncate(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (seq (T, F), mask (T,) bool)."""
        T = self.T
        F = seq.shape[1]
        out = np.zeros((T, F), dtype=np.float32)
        mask = np.zeros(T, dtype=bool)
        n = min(seq.shape[0], T)
        if n > 0:
            # center-crop or take first T (we take first T for simplicity)
            out[:n] = seq[:n]
            mask[:n] = True
        return out, mask

    def __getitem__(self, idx: int):
        seq, mask = self._pad_or_truncate(self.seqs[idx])
        return (
            torch.from_numpy(seq),
            torch.from_numpy(mask),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_sequences(
    perframe_parquet: Path,
    aligned_parquet: Path,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Return (seqs aligned to row order of aligned_parquet, labels, uids).

    Bouts that produced 0 frames are represented by an empty array — the
    dataset will then return an all-zero (all-False mask) sample.
    """
    aligned = pd.read_parquet(aligned_parquet)
    aligned["bout_uid"] = (
        aligned["bashoId"].astype(str) + "_"
        + aligned["day"].astype(int).astype(str) + "_"
        + aligned["matchNo"].astype(int).astype(str)
    )
    pf = pd.read_parquet(perframe_parquet)

    # Per-frame parquet already has bout_uid (string)
    if "bout_uid" not in pf.columns:
        pf["bout_uid"] = (
            pf["bashoId"].astype(str) + "_"
            + pf["day"].astype(int).astype(str) + "_"
            + pf["matchNo"].astype(int).astype(str)
        )

    # Z-score each feature globally for stable LSTM training.
    feat_cols = list(FEATURE_NAMES)
    arr = pf[feat_cols].to_numpy(dtype=np.float32)
    # Replace NaN / inf with 0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0) + 1e-6
    arr_norm = (arr - mu) / sigma
    pf_norm = pd.DataFrame(arr_norm, columns=feat_cols)
    pf_norm["bout_uid"] = pf["bout_uid"].values
    pf_norm["frame_idx"] = pf.get("frame_idx", pd.Series(range(len(pf)))).values

    seqs: list[np.ndarray] = []
    labels: list[int] = []
    uids: list[str] = []
    for _, row in aligned.iterrows():
        uid = row["bout_uid"]
        g = pf_norm[pf_norm["bout_uid"] == uid].sort_values("frame_idx")
        if len(g) > 0:
            seq = g[feat_cols].to_numpy(dtype=np.float32)
        else:
            seq = np.zeros((0, len(feat_cols)), dtype=np.float32)
        seqs.append(seq)
        labels.append(int(row["y_east"]))
        uids.append(uid)
    return seqs, np.array(labels, dtype=np.int64), uids


# ---------------------------------------------------------------------
# Train / eval loop
# ---------------------------------------------------------------------
def train_one_fold(
    train_seqs: list[np.ndarray],
    train_y: np.ndarray,
    val_seqs: list[np.ndarray],
    val_y: np.ndarray,
    *,
    T: int,
    hidden: int,
    embed_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    device: torch.device,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Train one fold, return (val_probs, best_val_loss)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = BoutPoseDataset(train_seqs, train_y, T=T)
    val_ds = BoutPoseDataset(val_seqs, val_y, T=T)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PoseTowerClassifier(
        feat_dim=len(FEATURE_NAMES),
        hidden=hidden,
        embed_dim=embed_dim,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            opt.zero_grad()
            logit = model(x, mask=m)
            loss = bce(logit, y)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * x.size(0)
        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, m, y in val_loader:
                x, m, y = x.to(device), m.to(device), y.to(device)
                logit = model(x, mask=m)
                loss = bce(logit, y)
                va_loss += float(loss.item()) * x.size(0)
        va_loss /= len(val_ds)

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep % 5 == 0 or bad >= patience:
            logger.info(
                "    ep %02d tr_loss=%.4f va_loss=%.4f bad=%d best=%.4f",
                ep, tr_loss, va_loss, bad, best_val,
            )
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    probs = np.zeros(len(val_ds), dtype=np.float32)
    idx = 0
    with torch.no_grad():
        for x, m, y in val_loader:
            x, m = x.to(device), m.to(device)
            logit = model(x, mask=m)
            p = torch.sigmoid(logit).cpu().numpy()
            probs[idx:idx + p.shape[0]] = p
            idx += p.shape[0]
    return probs, best_val


def run(
    perframe_parquet: Path,
    aligned_parquet: Path,
    out_path: Path,
    *,
    T: int = 75,
    hidden: int = 64,
    embed_dim: int = 64,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    epochs: int = 50,
    patience: int = 8,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    seqs, y, uids = load_sequences(perframe_parquet, aligned_parquet)
    logger.info(
        "Loaded %d bouts; T stats: min=%d  max=%d  mean=%.1f",
        len(seqs),
        min(s.shape[0] for s in seqs),
        max(s.shape[0] for s in seqs),
        float(np.mean([s.shape[0] for s in seqs])),
    )
    logger.info("Label balance: %s", np.bincount(y).tolist())

    # CPU device — MPS LSTM has known bugs.
    device = torch.device("cpu")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_metrics = []
    for fold, (tr, va) in enumerate(skf.split(np.arange(len(y)), y)):
        logger.info("Fold %d  train=%d  val=%d", fold, len(tr), len(va))
        train_seqs = [seqs[i] for i in tr]
        val_seqs = [seqs[i] for i in va]
        probs, best_val = train_one_fold(
            train_seqs, y[tr], val_seqs, y[va],
            T=T, hidden=hidden, embed_dim=embed_dim, dropout=dropout,
            lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, epochs=epochs, patience=patience,
            device=device, seed=seed + fold,
        )
        oof[va] = probs
        acc = accuracy_score(y[va], probs > 0.5)
        ll = log_loss(y[va], np.clip(probs, 1e-6, 1 - 1e-6), labels=[0, 1])
        logger.info(
            "  fold %d done  val_acc=%.4f val_ll=%.4f best_va_loss=%.4f",
            fold, acc, ll, best_val,
        )
        fold_metrics.append({"fold": fold, "val_acc": float(acc), "val_logloss": float(ll)})

    acc = accuracy_score(y, oof > 0.5)
    try:
        auc = roc_auc_score(y, oof)
    except ValueError:
        auc = float("nan")
    ll = log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6), labels=[0, 1])
    summary = {
        "n_bouts": int(len(y)),
        "oof_acc": float(acc),
        "oof_auc": float(auc),
        "oof_logloss": float(ll),
        "majority_acc": float(max(y.mean(), 1 - y.mean())),
        "folds": fold_metrics,
        "T": T, "hidden": hidden, "embed_dim": embed_dim, "dropout": dropout,
        "lr": lr, "epochs": epochs, "batch_size": batch_size,
    }
    logger.info("OOF acc=%.4f  auc=%.4f  logloss=%.4f", acc, auc, ll)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, oof)
    (out_path.parent / (out_path.stem + "_meta.json")).write_text(json.dumps(summary, indent=2))
    return summary


def _cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )
    summary = run(
        Path(args.perframe),
        Path(args.aligned),
        Path(args.out),
        T=args.T, hidden=args.hidden, embed_dim=args.embed_dim,
        dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        n_splits=args.n_splits, seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="5-fold CV PoseTower training on real per-frame data")
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--perframe", required=True, type=Path)
    r.add_argument("--aligned", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--T", type=int, default=75)
    r.add_argument("--hidden", type=int, default=64)
    r.add_argument("--embed-dim", type=int, default=64)
    r.add_argument("--dropout", type=float, default=0.3)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--weight-decay", type=float, default=1e-4)
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--epochs", type=int, default=50)
    r.add_argument("--patience", type=int, default=8)
    r.add_argument("--n-splits", type=int, default=5)
    r.add_argument("--seed", type=int, default=42)
    r.set_defaults(func=_cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
