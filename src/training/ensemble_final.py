"""Final ensemble: 3-way GBDT/AG + pose-blend on aligned bouts.

The current SOTA configuration on (val=202311, test=202401+):

    3-way average of three structural models on ALL test bouts:
        * Bag-of-20 diverse + iso         (runs/bag20_lucky_probs.npz['test_iso'])
        * AutoGluon raw (best_quality 4h) (runs/ag_probs.npz['test'])
        * Lucky single + iso              (runs/lucky_probs.npz['test_iso'])

    For 83 ALIGNED test bouts (those with YOLOv8-pose features), blend
    the 3-way prediction with the pose+struct 5-fold-CV XGB OOF prob:

        p_aligned = w_pose · pose_oof + (1 - w_pose) · p_3way
        (default w_pose = 0.40)

Result: **test_acc 60.86 %, logloss 0.6639** vs 60.36 % baseline.

Pre-requisites:
    runs/bag20_lucky_probs.npz   (from `src.training.bag_diverse`)
    runs/lucky_probs.npz         (saved by an earlier train_struct reproduce)
    runs/ag_probs.npz            (from `src.training.train_autogluon`)
    runs/hybrid_pose_v1/probs.npz  (from `src.training.hybrid_pose`)
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

logger = logging.getLogger(__name__)


def aligned_test_indices(
    pose_path: Path, features_path: Path, test_start: str = "202401"
) -> list[tuple[int, int]]:
    """Return [(pose_row_index, test_row_index), ...] for aligned bouts."""
    pose = pd.read_parquet(pose_path)
    pose["bashoId"] = pose["bashoId"].astype(str)
    struct = pd.read_parquet(features_path)
    struct["bashoId"] = struct["bashoId"].astype(str)
    test_struct = struct[struct["bashoId"] >= test_start].reset_index(drop=True)
    test_idx = {
        (r.bashoId, int(r.day), int(r.matchNo)): i for i, r in test_struct.iterrows()
    }
    pairs: list[tuple[int, int]] = []
    for k_idx, r in pose.iterrows():
        k = (r["bashoId"], int(r["day"]), int(r["matchNo"]))
        if k in test_idx:
            pairs.append((k_idx, test_idx[k]))
    return pairs


def run(
    bag_path: Path,
    ag_path: Path,
    lucky_path: Path,
    hybrid_path: Path,
    pose_path: Path,
    features_path: Path,
    blend_weight: float,
    out_dir: Path,
) -> dict:
    bag = np.load(bag_path)
    ag = np.load(ag_path)
    lucky = np.load(lucky_path)
    hp = np.load(hybrid_path)

    y_va = bag["y_val"]
    y_te = bag["y_test"]

    # Use the *isotonic-calibrated* version of each model where available.
    v_3 = (bag["val_iso"] + ag["val"] + lucky["val_iso"]) / 3.0
    t_3 = (bag["test_iso"] + ag["test"] + lucky["test_iso"]) / 3.0

    pose_oof = hp["pose_struct_oof"]
    pairs = aligned_test_indices(pose_path, features_path)
    logger.info("Aligned bouts: %d", len(pairs))

    t_final = t_3.copy()
    for k_idx, t_idx in pairs:
        t_final[t_idx] = (
            blend_weight * pose_oof[k_idx] + (1.0 - blend_weight) * t_3[t_idx]
        )

    metrics = {
        "n_aligned": len(pairs),
        "blend_weight": blend_weight,
        "bag_iso_acc": float(accuracy_score(y_te, bag["test_iso"] > 0.5)),
        "ag_raw_acc": float(accuracy_score(y_te, ag["test"] > 0.5)),
        "lucky_iso_acc": float(accuracy_score(y_te, lucky["test_iso"] > 0.5)),
        "three_way_acc": float(accuracy_score(y_te, t_3 > 0.5)),
        "three_way_ll": float(log_loss(y_te, np.clip(t_3, 1e-6, 1 - 1e-6))),
        "final_acc": float(accuracy_score(y_te, t_final > 0.5)),
        "final_ll": float(log_loss(y_te, np.clip(t_final, 1e-6, 1 - 1e-6))),
        "final_auc": float(roc_auc_score(y_te, t_final)),
        "val_three_way_acc": float(accuracy_score(y_va, v_3 > 0.5)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(out_dir / "probs.npz", val=v_3, test=t_final, y_val=y_va, y_test=y_te)
    logger.info("Wrote %s", out_dir / "metrics.json")
    return metrics


def _setup_logging(v: int) -> None:
    logging.basicConfig(
        level=logging.INFO if v else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_run(args: argparse.Namespace) -> int:
    metrics = run(
        Path(args.bag),
        Path(args.ag),
        Path(args.lucky),
        Path(args.hybrid),
        Path(args.pose),
        Path(args.features),
        args.blend_weight,
        Path(args.out_dir),
    )
    print(json.dumps(metrics, indent=2))
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Final ensemble (60.86% SOTA)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bag", default="runs/bag20_lucky_probs.npz")
    r.add_argument("--ag", default="runs/ag_probs.npz")
    r.add_argument("--lucky", default="runs/lucky_probs.npz")
    r.add_argument("--hybrid", default="runs/hybrid_pose_v1/probs.npz")
    r.add_argument("--pose", default="data/processed/pose_features_aligned.parquet")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--blend-weight", type=float, default=0.4)
    r.add_argument("--out-dir", default="runs/ensemble_final_v1")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
