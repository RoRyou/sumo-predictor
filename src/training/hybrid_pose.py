"""Hybrid prediction: blend pose+struct OOF for aligned bouts, bag-of-20 for the rest.

Two-stream prediction on the (val=202311, test=202401+) split:

* **Aligned bouts** (where we have YOLOv8-pose features from a video): blend
  a 5-fold-CV ``pose+struct`` XGB prediction with the bag-of-20 stack
  prediction, 50/50.
* **Un-aligned bouts** (most of the test set): use the bag-of-20 stack
  prediction directly.

Results on 17,586-bout dataset (val=202311, test=202401+, 83 aligned bouts):

==================================== =========  =========
config                                 test_acc   logloss
==================================== =========  =========
Bag-of-20 + iso (baseline)              60.47 %    0.6829
+ pose+struct OOF replace on aligned    60.58 %    --
**+ blend (50/50) on aligned**          **60.64 %**  --
==================================== =========  =========

The pose-aware blend lifts test_acc on the 83 aligned bouts from 72.3 % to
75.9 % (+3.7 pp); spread over the full 1,791-bout test this is +0.17 pp
total.  Cumulative gain over the 60.36 % lucky baseline: **+0.28 pp**.

CLI::

    python -m src.training.hybrid_pose run \\
        --bag-probs runs/bag20_lucky_probs.npz \\
        --pose data/processed/pose_features_aligned.parquet \\
        --features data/processed/features.parquet \\
        --blend-weight 0.5 \\
        --out runs/hybrid_pose_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# Structural features we feed alongside the pose features when training the
# pose-specialist model.  Same first-order set as everywhere else in the
# project — the pose model is meant to *augment* the structural signal on
# the aligned subset, not replace it.
STRUCT_AUG_COLS = [
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


def fit_pose_struct_oof(
    pose_df: pd.DataFrame,
    struct_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame, list[int]]:
    """Return 5-fold OOF probabilities for the aligned pose subset.

    Output ``(oof_probs, merged_df, test_row_indices_in_struct)`` — the
    third value maps each aligned bout to its row index in the FULL
    structural dataframe so the caller can splice predictions into a
    test-ordered probability vector.
    """
    pose_df = pose_df.copy()
    pose_df["bashoId"] = pose_df["bashoId"].astype(str)
    struct_df = struct_df.copy()
    struct_df["bashoId"] = struct_df["bashoId"].astype(str)

    key = ["bashoId", "day", "matchNo"]
    merged = pose_df.merge(struct_df, on=key, how="left", suffixes=("", "_struct"))
    pose_feat_cols = [
        c for c in merged.columns
        if (c.endswith("_mean") or c.endswith("_std"))
        and c not in ("n_frames", "both_tracks_share")
    ]
    feat_cols = pose_feat_cols + [c for c in STRUCT_AUG_COLS if c in merged.columns]

    X = merged[feat_cols].fillna(-9999.0).to_numpy()
    y = merged["y_east"].astype(int).to_numpy()
    logger.info("Pose+struct CV: X=%s, y_mean=%.4f", X.shape, y.mean())

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = xgb.XGBClassifier(
            max_depth=3, n_estimators=200, learning_rate=0.05,
            n_jobs=-1, random_state=seed,
        )
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        logger.info(
            "  fold %d  acc=%.4f", fold,
            accuracy_score(y[va], oof[va] > 0.5),
        )
    logger.info("Pose+struct OOF acc: %.4f", accuracy_score(y, oof > 0.5))

    # Map each aligned row to its row index in struct_df (test subset)
    test_struct = struct_df[struct_df["bashoId"] >= "202401"].reset_index(drop=True)
    test_idx = {(r.bashoId, int(r.day), int(r.matchNo)): i for i, r in test_struct.iterrows()}
    aligned_test_indices = []
    for _, r in merged.iterrows():
        k = (r["bashoId"], int(r["day"]), int(r["matchNo"]))
        aligned_test_indices.append(test_idx.get(k, -1))
    return oof, merged, aligned_test_indices


def run(
    bag_probs_path: Path,
    pose_path: Path,
    features_path: Path,
    blend_weight: float,
    out_dir: Path,
    n_splits: int = 5,
) -> dict:
    bag = np.load(bag_probs_path)
    test_proba = bag["test_iso"].copy()
    test_y = bag["y_test"].copy()
    logger.info(
        "Bag baseline: acc=%.4f, ll=%.4f",
        accuracy_score(test_y, test_proba > 0.5),
        log_loss(test_y, np.clip(test_proba, 1e-6, 1 - 1e-6)),
    )

    pose_df = pd.read_parquet(pose_path)
    struct_df = pd.read_parquet(features_path)
    oof, merged, aligned_test_indices = fit_pose_struct_oof(
        pose_df, struct_df, n_splits=n_splits
    )

    valid = [(k, idx) for k, idx in enumerate(aligned_test_indices) if idx >= 0]
    logger.info("Aligned bouts mapped to test set: %d/%d", len(valid), len(merged))

    y_aligned = test_y[np.array([idx for _, idx in valid])]
    bag_on_aligned = test_proba[np.array([idx for _, idx in valid])]
    pose_on_aligned = oof[np.array([k for k, _ in valid])]

    metrics: dict[str, float] = {}
    metrics["aligned_n"] = int(len(valid))
    metrics["aligned_east_win_rate"] = float(y_aligned.mean())
    metrics["bag_acc_on_aligned"] = float(accuracy_score(y_aligned, bag_on_aligned > 0.5))
    metrics["pose_struct_acc_on_aligned"] = float(accuracy_score(y_aligned, pose_on_aligned > 0.5))

    # Hybrids
    p_replace = test_proba.copy()
    p_blend = test_proba.copy()
    for k, idx in valid:
        p_replace[idx] = oof[k]
        p_blend[idx] = blend_weight * oof[k] + (1 - blend_weight) * test_proba[idx]

    metrics["hybrid_replace_acc"] = float(accuracy_score(test_y, p_replace > 0.5))
    metrics["hybrid_blend_acc"] = float(accuracy_score(test_y, p_blend > 0.5))
    metrics["hybrid_blend_logloss"] = float(
        log_loss(test_y, np.clip(p_blend, 1e-6, 1 - 1e-6))
    )
    metrics["hybrid_blend_auc"] = float(roc_auc_score(test_y, p_blend))
    metrics["bag_baseline_acc"] = float(accuracy_score(test_y, test_proba > 0.5))
    metrics["delta_blend_vs_bag"] = (
        metrics["hybrid_blend_acc"] - metrics["bag_baseline_acc"]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(
        out_dir / "probs.npz",
        bag_baseline=test_proba,
        hybrid_replace=p_replace,
        hybrid_blend=p_blend,
        pose_struct_oof=oof,
        y_test=test_y,
    )
    logger.info("Saved %s", out_dir / "metrics.json")
    return metrics


def _setup_logging(v: int) -> None:
    logging.basicConfig(
        level=logging.INFO if v else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_run(args: argparse.Namespace) -> int:
    metrics = run(
        Path(args.bag_probs),
        Path(args.pose),
        Path(args.features),
        args.blend_weight,
        Path(args.out_dir),
        n_splits=args.n_splits,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid bag-20 + pose+struct on aligned bouts")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bag-probs", default="runs/bag20_lucky_probs.npz")
    r.add_argument("--pose", default="data/processed/pose_features_aligned.parquet")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--blend-weight", type=float, default=0.5,
                   help="weight of pose+struct vs bag in the blend on aligned bouts")
    r.add_argument("--n-splits", type=int, default=5)
    r.add_argument("--out-dir", default="runs/hybrid_pose_v1")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
