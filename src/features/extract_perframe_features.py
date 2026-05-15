"""Per-frame kinematic feature extraction for aligned sumo bouts.

Unlike :mod:`src.features.extract_bout_features` (which aggregates each
bout segment to a single mean/std row), this module emits ONE ROW PER
FRAME so a temporal model (Bi-LSTM, Transformer, etc.) can consume the
raw 40-dim kinematic signal.

It re-uses the existing pipeline:

* :class:`src.features.pose.PoseExtractor`        – YOLOv8-pose
* :class:`src.features.tracking.TwoRikishiTracker` – ID lock per segment
* :func:`src.features.kinematics.compute_features` – 40-dim features

Input: ``data/processed/pose_features_aligned.parquet`` which already
carries the resolved ``[video_id, bashoId, day, matchNo, t_start, t_end,
y_east]`` per aligned bout.

Output (one long-format parquet, one row per frame):

    [video_id, bashoId, day, matchNo, y_east, t,
     frame_idx, A_com_x_n, ..., reserved_3, both_tracks_any]

CLI
---
::

    python -m src.features.extract_perframe_features run \\
        --aligned-parquet data/processed/pose_features_aligned.parquet \\
        --videos-dir data/videos/aligned \\
        --out data/processed/pose_perframe.parquet \\
        --target-fps 15
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.extract_bout_features import (
    _read_segment_frames,
    _run_pose_pipeline,
    both_tracks_share,
)
from src.features.kinematics import FEATURE_NAMES

logger = logging.getLogger(__name__)


def extract_perframe_for_bout(
    video_path: Path,
    t_start: float,
    t_end: float,
    target_fps: float,
    pose_extractor=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(feats (T,40), kp_seq (T,2,17,3), eff_fps)`` for a bout."""
    frames, fps = _read_segment_frames(
        video_path, t_start, t_end, target_fps=target_fps
    )
    if frames.shape[0] < 3:
        return (
            np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0, 2, 17, 3), dtype=np.float32),
            fps,
        )
    kp_seq, feats = _run_pose_pipeline(frames, fps, extractor=pose_extractor)
    return feats, kp_seq, fps


def run(
    aligned_parquet: Path,
    videos_dir: Path,
    out_path: Path,
    target_fps: float = 15.0,
) -> Path:
    """Loop over aligned bouts, extract per-frame features, write parquet."""
    aligned = pd.read_parquet(aligned_parquet)
    logger.info("Loaded %d aligned bouts from %s", len(aligned), aligned_parquet)

    # cache pose extractor across videos (loading YOLO is slow)
    from src.features.pose import PoseExtractor
    pose_extractor = PoseExtractor()

    all_rows: list[pd.DataFrame] = []
    stats = {
        "n_bouts": 0,
        "n_frames_total": 0,
        "n_zero_frame_bouts": 0,
        "id_lock_per_bout": [],  # both_tracks_share per bout
    }
    for i, row in aligned.iterrows():
        video_id = str(row["video_id"])
        video_path = videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            logger.warning("Missing video %s; skipping", video_path)
            continue
        t0 = float(row["t_start"])
        t1 = float(row["t_end"])
        logger.info(
            "[%d/%d] %s  %.2fs-%.2fs  bouts=%s_d%s_m%s",
            i + 1, len(aligned), video_id, t0, t1,
            row.get("bashoId"), row.get("day"), row.get("matchNo"),
        )
        feats, kp_seq, eff_fps = extract_perframe_for_bout(
            video_path, t0, t1, target_fps=target_fps,
            pose_extractor=pose_extractor,
        )
        T = feats.shape[0]
        if T == 0:
            stats["n_zero_frame_bouts"] += 1
            stats["id_lock_per_bout"].append(0.0)
            continue
        id_lock = both_tracks_share(kp_seq)
        stats["id_lock_per_bout"].append(id_lock)
        stats["n_bouts"] += 1
        stats["n_frames_total"] += T

        # build long-format df: one row per frame
        df = pd.DataFrame(feats, columns=FEATURE_NAMES)
        df.insert(0, "frame_idx", np.arange(T, dtype=np.int32))
        df.insert(1, "t", df["frame_idx"] / eff_fps + t0)
        df["video_id"] = video_id
        df["bashoId"] = str(row["bashoId"])
        df["day"] = int(row["day"])
        df["matchNo"] = int(row["matchNo"])
        df["y_east"] = int(row["y_east"])
        df["both_tracks_share"] = float(id_lock)
        df["bout_uid"] = f"{row['bashoId']}_{int(row['day'])}_{int(row['matchNo'])}"
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No per-frame rows produced!")
    out = pd.concat(all_rows, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info(
        "Wrote %d rows (= total frames) for %d bouts to %s",
        len(out), out["bout_uid"].nunique(), out_path,
    )
    logger.info(
        "Mean T per bout: %.1f  zero-frame bouts: %d  mean ID-lock: %.3f",
        np.mean([len(g) for _, g in out.groupby("bout_uid")]),
        stats["n_zero_frame_bouts"],
        float(np.mean(stats["id_lock_per_bout"])) if stats["id_lock_per_bout"] else 0.0,
    )
    return out_path


def _cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )
    run(
        Path(args.aligned_parquet),
        Path(args.videos_dir),
        Path(args.out),
        target_fps=args.target_fps,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-frame kinematic feature extractor")
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument(
        "--aligned-parquet", required=True, type=Path,
        help="parquet containing the resolved aligned bouts with t_start/t_end",
    )
    r.add_argument(
        "--videos-dir", required=True, type=Path,
        help="directory holding {video_id}.mp4",
    )
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--target-fps", type=float, default=15.0)
    r.set_defaults(func=_cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
