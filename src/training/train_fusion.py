"""Train Phase 3 two-tower fusion model.

Glues the trained Phase-1 stacked ensemble (Route A) to the Phase-2 pose
tower (Route B):

1. Reload structural features ``data/processed/features.parquet`` and the
   trained stacking pipeline (rebuilt on the fly — fast).
2. Load aligned pose features (one ``(T, F)`` array per bout) from
   ``data/processed/pose_features.parquet`` (this file is produced by a
   future ``src/features/pose.py`` batch-extract job; not present yet
   except for the smoke clip).
3. For each bout, build the input tuple ``(struct_x, pose_x)`` where
   ``struct_x`` includes the GBDT stacked logit plus key raw features.
4. Train :class:`~src.models.fusion.SumoFusionModel` with Lightning.

Until real aligned video data exists, this module exposes a
``smoke`` subcommand that builds a synthetic pose tensor (T=120, F=40) per
bout — useful to verify shapes/gradients flow end-to-end.

CLI::

    # End-to-end smoke (real struct + synthetic pose)
    python -m src.training.train_fusion smoke \\
        --features data/processed/features.parquet \\
        --val-basho 202311 --test-start 202401

    # Future: real fusion training (needs aligned pose features)
    python -m src.training.train_fusion fit \\
        --struct data/processed/features.parquet \\
        --pose   data/processed/pose_features.parquet
"""
from __future__ import annotations

# NOTE: PyTorch and XGBoost's OpenMP runtimes collide on macOS (both bundle
# their own libomp.dylib), causing a SIGSEGV when XGBoost trains *after*
# torch is imported.  Setting these env vars BEFORE torch/xgb import
# serialises the OMP scheduler and avoids the crash.  See:
#   - https://github.com/dmlc/xgboost/issues/1715
#   - https://github.com/pytorch/pytorch/issues/3146
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.models.fusion import SumoFusionModel
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


# ---------------------------------------------------------------------- #
# Key raw features fed alongside the stacked logit
# ---------------------------------------------------------------------- #
STRUCT_RAW_FEATURES = [
    "rank_diff",
    "height_diff",
    "weight_diff",
    "bmi_diff",
    "age_diff",
    "winrate_diff_10",
    "winrate_diff_30",
    "winrate_diff_90",
    "h2h_winrate",
    "h2h_count",
    "streak_A",
    "streak_B",
    "career_winrate_A",
    "career_winrate_B",
    "pushing_ratio_A",
    "pushing_ratio_B",
    "belt_ratio_A",
    "belt_ratio_B",
    "day_of_basho",
]


# ---------------------------------------------------------------------- #
# Stage-1 GBDT to produce a stacked logit per bout
# ---------------------------------------------------------------------- #
def gbdt_logits_for_split(
    train_df: pd.DataFrame,
    other_dfs: list[pd.DataFrame],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fit a single XGB on train, return logit probs on train + others.

    Lightweight stand-in for the full 3-base-model stack; the goal here is
    just to feed a strong scalar signal into the fusion model — adding the
    full stack later is straightforward (replace this function).
    """
    y_tr = train_df[LABEL_COL].to_numpy()
    w_tr = train_df[WEIGHT_COL].to_numpy() if WEIGHT_COL in train_df.columns else None

    te = KFoldTargetEncoder(CATEGORICAL_COLS)
    Xtr = te.fit_transform(train_df, y_tr)
    cols = feature_cols(Xtr)
    Xtr_m = Xtr[cols].fillna(-9999.0)

    model = make_xgb({"n_estimators": 500, "learning_rate": 0.05})
    model.fit(Xtr_m, y_tr, sample_weight=w_tr)
    p_tr = np.clip(model.predict_proba(Xtr_m)[:, 1], 1e-6, 1 - 1e-6)

    other_logits = []
    for o in other_dfs:
        if len(o) == 0:
            other_logits.append(np.zeros(0))
            continue
        Xo = te.transform(o)[cols].fillna(-9999.0)
        po = np.clip(model.predict_proba(Xo)[:, 1], 1e-6, 1 - 1e-6)
        other_logits.append(_logit(po))
    return _logit(p_tr), other_logits


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------- #
# Synthetic pose generator (for smoke tests / dry runs)
# ---------------------------------------------------------------------- #
def synthetic_pose_tensor(
    n: int, T: int = 120, F: int = 40, seed: int = 42
) -> np.ndarray:
    """Return shape ``(n, T, F)`` plausible pose-feature noise.

    Modelled as ``a + b*sin(omega*t) + eps`` per dim so the temporal model
    sees some structure it can fit (which lets us sanity-check that
    fusion isn't degenerate).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)[None, :, None]
    a = rng.standard_normal((n, 1, F))
    b = rng.standard_normal((n, 1, F)) * 0.3
    omega = rng.uniform(0.05, 0.3, size=(n, 1, F))
    eps = rng.standard_normal((n, T, F)) * 0.1
    return (a + b * np.sin(omega * t) + eps).astype(np.float32)


# ---------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------- #
class FusionDataset(Dataset):
    """Yields ``(struct_x, pose_x, y)``."""

    def __init__(
        self,
        struct_x: np.ndarray,
        pose_x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        assert len(struct_x) == len(pose_x) == len(y)
        self.struct_x = struct_x.astype(np.float32)
        self.pose_x = pose_x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.struct_x[i]),
            torch.from_numpy(self.pose_x[i]),
            torch.tensor(self.y[i]),
        )


# ---------------------------------------------------------------------- #
# Build struct features (raw + stacked logit) for each split
# ---------------------------------------------------------------------- #
def build_struct_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Return ``{'train': arr, 'val': arr, 'test': arr}`` with raw features + stacked logit."""
    used_raw = [c for c in STRUCT_RAW_FEATURES if c in train.columns]
    tr_logit, [va_logit, te_logit] = gbdt_logits_for_split(train, [val, test])

    def _pack(df: pd.DataFrame, logits: np.ndarray) -> np.ndarray:
        raw = df[used_raw].fillna(0.0).to_numpy(dtype=np.float32)
        return np.concatenate([raw, logits.reshape(-1, 1)], axis=1).astype(np.float32)

    return {
        "train": _pack(train, tr_logit),
        "val": _pack(val, va_logit),
        "test": _pack(test, te_logit),
        "used_raw": used_raw,
    }


# ---------------------------------------------------------------------- #
# Train loop
# ---------------------------------------------------------------------- #
def train_one_epoch(model, loader, opt, loss_fn, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for s, p, y in loader:
        s, p, y = s.to(device), p.to(device), y.to(device)
        opt.zero_grad()
        logits = model.forward_logits(s, p)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total += float(loss) * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    ys, ps = [], []
    for s, p, y in loader:
        s, p = s.to(device), p.to(device)
        prob = torch.sigmoid(model.forward_logits(s, p)).cpu().numpy()
        ps.append(prob)
        ys.append(y.numpy())
    if not ys:
        return {}
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    p_c = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, p > 0.5)),
        "logloss": float(log_loss(y, p_c)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


# ---------------------------------------------------------------------- #
# Smoke CLI: real struct + synthetic pose
# ---------------------------------------------------------------------- #
def cmd_smoke(args: argparse.Namespace) -> int:
    df = pd.read_parquet(args.features)
    df["bashoId"] = df["bashoId"].astype(str)
    sp = time_split(df, val_basho=args.val_basho, test_start=args.test_start)
    logger.info(repr(sp))

    feats = build_struct_features(sp.train, sp.val, sp.test)
    struct_dim = feats["train"].shape[1]
    logger.info("struct_dim = %d (raw=%d + stacked_logit=1)", struct_dim, len(feats["used_raw"]))

    # synthetic pose tensors (deterministic per split)
    pose_tr = synthetic_pose_tensor(len(sp.train), T=args.pose_T, F=args.pose_F, seed=42)
    pose_va = synthetic_pose_tensor(len(sp.val), T=args.pose_T, F=args.pose_F, seed=43)
    pose_te = synthetic_pose_tensor(len(sp.test), T=args.pose_T, F=args.pose_F, seed=44)

    y_tr = sp.train[LABEL_COL].to_numpy().astype(np.float32)
    y_va = sp.val[LABEL_COL].to_numpy().astype(np.float32)
    y_te = sp.test[LABEL_COL].to_numpy().astype(np.float32)

    # NOTE: torch MPS has known LSTM/segfault issues (see pytorch#92602);
    # default to CPU on macOS unless the caller forces otherwise.
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("device=%s", device)

    model = SumoFusionModel(
        struct_in_dim=struct_dim,
        pose_feat_dim=args.pose_F,
        struct_embed=32,
        pose_embed=128,
        hidden=64,
        dropout=0.3,
    ).to(device)

    tr_ds = FusionDataset(feats["train"], pose_tr, y_tr)
    va_ds = FusionDataset(feats["val"], pose_va, y_va)
    te_ds = FusionDataset(feats["test"], pose_te, y_te)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for ep in range(1, args.epochs + 1):
        loss_tr = train_one_epoch(model, tr_dl, opt, loss_fn, device)
        m_va = evaluate(model, va_dl, device)
        logger.info(
            "epoch %d  train_loss=%.4f  val_acc=%.4f  val_loss=%.4f",
            ep, loss_tr, m_va.get("acc", 0), m_va.get("logloss", 0),
        )
        history.append({"epoch": ep, "train_loss": loss_tr, **m_va})

    m_te = evaluate(model, te_dl, device)
    logger.info("[test] %s", m_te)
    out = {"history": history, "test": m_te, "struct_dim": struct_dim,
           "pose_T": args.pose_T, "pose_F": args.pose_F}

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, default=str))
        logger.info("Wrote %s", out_path)
    print(json.dumps(out, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------- #
# CLI plumbing
# ---------------------------------------------------------------------- #
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


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fusion training (Phase 3)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke", help="Smoke test: real struct + synthetic pose")
    s.add_argument("--features", default="data/processed/features.parquet")
    s.add_argument("--val-basho", required=True)
    s.add_argument("--test-start", required=True)
    s.add_argument("--epochs", type=int, default=4)
    s.add_argument("--batch-size", type=int, default=256)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--pose-T", type=int, default=120)
    s.add_argument("--pose-F", type=int, default=40)
    s.add_argument("--out", default="runs/fusion_smoke.json")
    s.add_argument("--device", default=None, help="force device (cpu/cuda/mps)")
    s.add_argument("-v", "--verbose", action="count", default=1)
    s.set_defaults(func=cmd_smoke)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
