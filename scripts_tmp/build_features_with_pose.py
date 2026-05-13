"""Phase B: build features_with_pose.parquet and run two evaluations.

1. Concatenate every parquet in data/processed/pose_per_segment/ into a
   single pose_features_aligned.parquet.  Dedupe by
   (bashoId, day, matchNo) so multiple visual segments → one row.
   When duplicates exist we keep the one with highest `both_tracks_share`.

2. Merge those rows onto data/processed/features.parquet by
   (bashoId, day, matchNo).  The 80 pose feature columns will be NaN
   for the vast majority of un-aligned rows.

3. Pose-only XGB on the small aligned subset (with y_east as label).
   80 cols, y is the bout-aligned label.  Hold out a stratified 20%
   for test; train on the rest.  Report acc vs 50% baseline.

4. (caller will use the new parquet to retrain the v4 stack.)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


ROOT = Path("/Users/liang.lu/project/sumo_pred")
POSE_DIR = ROOT / "data/processed/pose_per_segment"
FEATURES_PATH = ROOT / "data/processed/features.parquet"
ALIGNED_OUT = ROOT / "data/processed/pose_features_aligned.parquet"
MERGED_OUT = ROOT / "data/processed/features_with_pose.parquet"
REPORT_DIR = ROOT / "reports/pose_phase2"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ---- 1) concatenate per-video parquets ----
    paths = sorted(POSE_DIR.glob("*.parquet"))
    logger.info("found %d per-video parquets", len(paths))
    if not paths:
        raise SystemExit("no per-video parquets found")

    parts = []
    per_video_stats: dict = {}
    for p in paths:
        d = pd.read_parquet(p)
        n_total = len(d)
        n_aligned = (d["winnerId"].notna().sum()
                     if "winnerId" in d.columns else 0)
        per_video_stats[p.stem] = {"segments": int(n_total),
                                   "aligned": int(n_aligned)}
        if n_total > 0:
            parts.append(d)
    big = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    logger.info("concatenated total rows=%d", len(big))

    # Only keep rows whose alignment succeeded (winnerId not null).
    if "winnerId" in big.columns:
        aligned = big[big["winnerId"].notna()].copy()
    else:
        aligned = big.iloc[0:0].copy()
    logger.info("aligned rows (any) = %d", len(aligned))

    # Dedupe by bout key — keep the row with highest both_tracks_share
    if not aligned.empty:
        aligned["bashoId"] = aligned["bashoId"].astype(str)
        aligned = aligned.sort_values("both_tracks_share", ascending=False)
        aligned = aligned.drop_duplicates(
            subset=["bashoId", "day", "matchNo"], keep="first"
        ).reset_index(drop=True)
    logger.info("unique aligned bouts = %d", len(aligned))
    aligned.to_parquet(ALIGNED_OUT, index=False)
    logger.info("wrote %s", ALIGNED_OUT)

    # Breakdown by basho
    by_basho = (aligned.groupby("bashoId").size().to_dict()
                if not aligned.empty else {})
    logger.info("aligned bouts by basho: %s", by_basho)

    # ---- 2) merge onto structural features ----
    feats = pd.read_parquet(FEATURES_PATH)
    feats["bashoId"] = feats["bashoId"].astype(str)
    pose_feat_cols = [c for c in aligned.columns
                      if c.endswith("_mean") or c.endswith("_std")]
    extra_pose_cols = ["both_tracks_share", "n_frames"]

    keep_pose = pose_feat_cols + extra_pose_cols
    # Make sure all are floats, no metadata other than the merge keys.
    if not aligned.empty:
        pose_compact = aligned[
            ["bashoId", "day", "matchNo"] + keep_pose
        ].copy()
        # cast keys to expected types
        pose_compact["day"] = pose_compact["day"].astype(int)
        pose_compact["matchNo"] = pose_compact["matchNo"].astype(int)
    else:
        # Empty placeholder with right columns so the merge still runs
        pose_compact = pd.DataFrame(columns=["bashoId", "day", "matchNo"] + keep_pose)

    # Rename pose feature columns with pose_ prefix to avoid collisions.
    rename_map = {c: f"pose_{c}" for c in keep_pose}
    pose_compact = pose_compact.rename(columns=rename_map)

    merged = feats.merge(
        pose_compact, on=["bashoId", "day", "matchNo"], how="left", validate="m:1"
    )
    has_pose_mask = merged[f"pose_{keep_pose[0]}"].notna()
    n_with_pose = int(has_pose_mask.sum())
    logger.info("merged %d / %d rows have pose features", n_with_pose, len(merged))
    merged.to_parquet(MERGED_OUT, index=False)
    logger.info("wrote %s", MERGED_OUT)

    # ---- 3) pose-only XGB on the aligned subset ----
    # Use a stratified split for the tiny sample.
    pose_cols_renamed = [f"pose_{c}" for c in keep_pose]
    aligned_only = merged[has_pose_mask].copy().reset_index(drop=True)
    logger.info("pose-only subset size=%d  east_win_rate=%.3f",
                len(aligned_only),
                float(aligned_only["y"].mean()) if len(aligned_only) else float("nan"))

    pose_xgb_result = {}
    if len(aligned_only) >= 12 and aligned_only["y"].nunique() > 1:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, log_loss
        import xgboost as xgb

        y = aligned_only["y"].astype(int).to_numpy()
        Xp = aligned_only[pose_cols_renamed].fillna(0.0).to_numpy(dtype=float)
        # tiny dataset — use leave-one-out style 5-fold CV to estimate accuracy
        kf = StratifiedKFold(n_splits=min(5, int(min(y.sum(), (y == 0).sum()))),
                              shuffle=True, random_state=42)
        oof = np.zeros(len(y), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(Xp, y)):
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                learning_rate=0.05,
                max_depth=3,
                n_estimators=200,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
                eval_metric="logloss",
            )
            model.fit(Xp[tr_idx], y[tr_idx])
            oof[va_idx] = model.predict_proba(Xp[va_idx])[:, 1]
        acc = accuracy_score(y, oof > 0.5)
        ll = log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))
        pose_xgb_result = {
            "n": int(len(y)),
            "n_pos": int(y.sum()),
            "cv_oof_acc": float(acc),
            "cv_oof_logloss": float(ll),
            "majority_class_baseline": float(max(y.mean(), 1 - y.mean())),
        }
    else:
        pose_xgb_result = {
            "n": int(len(aligned_only)),
            "skipped": True,
            "reason": "too few aligned bouts / single-class label",
        }
    logger.info("pose-only XGB OOF result: %s", pose_xgb_result)

    # ---- 4) save summary report ----
    summary = {
        "per_video": per_video_stats,
        "total_segments": int(sum(v["segments"] for v in per_video_stats.values())),
        "total_aligned_segment_rows": int(sum(v["aligned"] for v in per_video_stats.values())),
        "unique_aligned_bouts": int(len(aligned)),
        "by_basho": by_basho,
        "n_with_pose_in_merged": n_with_pose,
        "pose_only_xgb": pose_xgb_result,
        "n_pose_feature_cols": int(len(pose_cols_renamed)),
        "merged_path": str(MERGED_OUT),
        "aligned_path": str(ALIGNED_OUT),
    }
    summary_path = REPORT_DIR / "phaseB_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
