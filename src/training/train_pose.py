"""Standalone training scaffold for the Pose Tower.

This is a **scaffold**, not a full training script.  Its job is to:

1. Lay out the ``PoseSequenceDataset`` that pairs a per-bout pose-features
   parquet with the bout outcomes parquet from Phase 1.
2. Build a Lightning module + Trainer with sensible defaults (binary
   cross-entropy, AdamW, cosine schedule).
3. Run a smoke verification (``smoke()``) that checks forward/backward
   pass on a synthetic batch -- useful before any real data exists.

Real training will be invoked by a separate driver once enough bouts have
been aligned to video clips.

Bout schema (from ``src/data/sumo_api.py``)::

    bashoId, day, matchNo, eastId, westId, winnerId, kimarite, ...

The dataset joins pose-features parquet on ``(bashoId, day, matchNo)``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.features.kinematics import FEATURE_DIM
from src.models.temporal import PoseTowerClassifier

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
BOUT_KEYS = ("bashoId", "day", "matchNo")


@dataclass
class BoutSample:
    bashoId: str
    day: int
    matchNo: int
    features: np.ndarray  # (T, F)
    label: int            # 1 if east won, 0 if west won


class PoseSequenceDataset(Dataset[BoutSample]):
    """Dataset of (per-bout pose feature sequence, east-wins label).

    Parameters
    ----------
    features_parquet
        Long-format parquet -- one row per *frame*, columns include
        ``bashoId, day, matchNo, t, <FEATURE_NAMES...>``.  ``t`` is the
        frame index inside the clip.
    bouts_parquet
        Phase 1's bout outcomes parquet (must contain
        ``bashoId, day, matchNo, eastId, westId, winnerId``).
    max_len
        Truncate / pad to this many frames.
    """

    def __init__(
        self,
        features_parquet: str | Path,
        bouts_parquet: str | Path,
        max_len: int = 150,
    ) -> None:
        import pandas as pd

        feats = pd.read_parquet(features_parquet)
        bouts = pd.read_parquet(bouts_parquet)

        for k in BOUT_KEYS:
            if k not in feats.columns or k not in bouts.columns:
                raise ValueError(f"missing required key column {k!r}")

        bouts = bouts.dropna(subset=["winnerId", "eastId", "westId"]).copy()
        bouts["label"] = (bouts["winnerId"] == bouts["eastId"]).astype(int)

        self._samples: list[BoutSample] = []
        feat_cols = [c for c in feats.columns if c not in (*BOUT_KEYS, "t")]
        if len(feat_cols) != FEATURE_DIM:
            logger.warning(
                "feature parquet has %d feature columns, expected %d",
                len(feat_cols),
                FEATURE_DIM,
            )

        grouped = feats.sort_values(["bashoId", "day", "matchNo", "t"]).groupby(
            list(BOUT_KEYS), sort=False
        )
        for (bashoId, day, matchNo), grp in grouped:
            row = bouts[
                (bouts["bashoId"] == bashoId)
                & (bouts["day"] == day)
                & (bouts["matchNo"] == matchNo)
            ]
            if row.empty:
                continue
            label = int(row["label"].iloc[0])
            arr = grp[feat_cols].to_numpy(dtype=np.float32)
            arr = _pad_or_trunc(arr, max_len)
            self._samples.append(
                BoutSample(
                    bashoId=str(bashoId),
                    day=int(day),
                    matchNo=int(matchNo),
                    features=arr,
                    label=label,
                )
            )
        logger.info("PoseSequenceDataset loaded %d bouts", len(self._samples))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self._samples[idx]
        return {
            "features": torch.from_numpy(s.features),
            "label": torch.tensor(s.label, dtype=torch.float32),
        }


def _pad_or_trunc(x: np.ndarray, T: int) -> np.ndarray:
    if x.shape[0] >= T:
        return x[:T]
    pad = np.zeros((T - x.shape[0], x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


# ----------------------------------------------------------------------
# Synthetic dataset (used by smoke / unit tests)
# ----------------------------------------------------------------------
class SyntheticPoseDataset(Dataset[BoutSample]):
    """Random Gaussian sequences with random binary labels."""

    def __init__(self, n: int = 32, T: int = 150, F: int = FEATURE_DIM, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.x = rng.standard_normal((n, T, F)).astype(np.float32)
        self.y = rng.integers(0, 2, size=n).astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.x[idx]),
            "label": torch.tensor(self.y[idx]),
        }


# ----------------------------------------------------------------------
# Lightning module
# ----------------------------------------------------------------------
try:
    import lightning as L  # type: ignore

    class PoseLightningModule(L.LightningModule):
        def __init__(
            self,
            feat_dim: int = FEATURE_DIM,
            hidden: int = 128,
            embed_dim: int = 128,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.save_hyperparameters()
            self.model = PoseTowerClassifier(
                feat_dim=feat_dim,
                hidden=hidden,
                embed_dim=embed_dim,
                dropout=dropout,
            )
            self.criterion = nn.BCEWithLogitsLoss()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        def _step(self, batch, stage: str) -> torch.Tensor:
            logits = self.model(batch["features"])
            loss = self.criterion(logits, batch["label"])
            self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=len(logits))
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == batch["label"]).float().mean()
            self.log(f"{stage}/acc", acc, prog_bar=True, batch_size=len(logits))
            return loss

        def training_step(self, batch, batch_idx):
            return self._step(batch, "train")

        def validation_step(self, batch, batch_idx):
            return self._step(batch, "val")

        def configure_optimizers(self):
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
            return {"optimizer": opt, "lr_scheduler": sched}
except ImportError:  # pragma: no cover
    L = None
    PoseLightningModule = None  # type: ignore[assignment]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
def smoke(n_steps: int = 3, batch_size: int = 4, T: int = 150) -> dict[str, float]:
    """Run a few train steps on a synthetic batch.  Returns initial/final loss."""
    torch.manual_seed(0)
    model = PoseTowerClassifier()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    ds = SyntheticPoseDataset(n=batch_size * 2, T=T)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    losses: list[float] = []
    model.train()
    for step, batch in enumerate(dl):
        if step >= n_steps:
            break
        logits = model(batch["features"])
        loss = crit(logits, batch["label"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    out = {
        "initial_loss": losses[0] if losses else float("nan"),
        "final_loss": losses[-1] if losses else float("nan"),
        "steps": float(len(losses)),
    }
    logger.info("Smoke: initial=%.4f final=%.4f", out["initial_loss"], out["final_loss"])
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.training.train_pose")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke", help="Run forward/backward smoke test")

    fit = sub.add_parser("fit", help="Run training on real pose features")
    fit.add_argument("--features", required=True)
    fit.add_argument("--bouts", required=True)
    fit.add_argument("--max-len", type=int, default=150)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--epochs", type=int, default=10)
    fit.add_argument("--lr", type=float, default=1e-3)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    if args.cmd == "smoke":
        out = smoke()
        print(out)
        return 0

    if args.cmd == "fit":
        if PoseLightningModule is None:
            raise SystemExit("lightning not installed; pip install lightning")
        ds = PoseSequenceDataset(args.features, args.bouts, max_len=args.max_len)
        n = len(ds)
        if n < 2:
            raise SystemExit(f"only {n} bouts available; need ≥2 for fit")
        split = max(1, int(n * 0.8))
        train_ds, val_ds = torch.utils.data.random_split(
            ds, [split, n - split],
            generator=torch.Generator().manual_seed(0),
        )
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size)
        module = PoseLightningModule(lr=args.lr)
        trainer = L.Trainer(  # type: ignore[attr-defined]
            max_epochs=args.epochs,
            accelerator="auto",
            log_every_n_steps=1,
        )
        trainer.fit(module, train_dl, val_dl)
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
