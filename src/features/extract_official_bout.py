"""Per-video pose feature extraction for Japan Sumo Association official clips.

These are single-bout videos (60-180s) with the bout itself usually in the
final 30-60 seconds (the earlier portion is ring entry / shikiri ritual).

Pipeline:
1. Read frames from t_start..t_end (default: full video).
2. YOLOv8-pose + TwoRikishiTracker.
3. Detect "fight start" via motion energy peak (both tracks moving).
4. Compute 40-dim kinematic features over the fight segment.
5. Aggregate (mean, std) → 1 row.

CLI::

    python -m src.features.extract_official_bout run \\
        --video data/videos/official/202401_d13_tobizaru_ura.mp4 \\
        --video-id 202401_d13_tobizaru_ura \\
        --bashoId 202401 --day 13 --matchNo 7 \\
        --eastId 1234 --westId 5678 \\
        --out data/processed/official_pose/<video_id>.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.features.kinematics import FEATURE_NAMES, compute_features
from src.features.pose import PoseExtractor

logger = logging.getLogger(__name__)


def find_fight_window(frames_pose: list, fps: float, min_dur: float = 3.0) -> tuple[int, int]:
    """Pick the high-motion window using both-tracks displacement.

    Returns (start_idx, end_idx) inclusive bounds in frames list.
    """
    if len(frames_pose) < 30:
        return (0, len(frames_pose) - 1)

    # Build per-frame motion = sum of |Δkeypoint| across both tracks
    motions = []
    prev = None
    for fp in frames_pose:
        cur_kp = None
        if fp.get("track_a") is not None and fp.get("track_b") is not None:
            kpa_xy = fp["track_a"][:, :2]
            kpb_xy = fp["track_b"][:, :2]
            cur_kp = np.concatenate([kpa_xy, kpb_xy], axis=0)
        if cur_kp is None or prev is None:
            motions.append(0.0)
        else:
            motions.append(float(np.linalg.norm(cur_kp - prev, axis=1).mean()))
        prev = cur_kp
    motions = np.array(motions)
    # Smooth with 0.5s window
    w = max(1, int(fps * 0.5))
    kernel = np.ones(w) / w
    motion_smooth = np.convolve(motions, kernel, mode="same")
    # Find peak window: 3-15s around argmax
    peak = int(motion_smooth.argmax())
    half = max(int(fps * 1.5), int(min_dur * fps / 2))
    start = max(0, peak - half)
    end = min(len(frames_pose) - 1, peak + half)
    return start, end


def extract_one_video(
    video_path: Path,
    fps_target: float = 10.0,
    save_perframe: bool = False,
) -> dict | None:
    """Extract aggregate pose features for one single-bout video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open %s", video_path)
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = nframes_total / src_fps if nframes_total > 0 else 0
    sample_every = max(1, int(round(src_fps / fps_target)))
    actual_fps = src_fps / sample_every

    pose = PoseExtractor()

    frames_pose = []
    fi = 0
    sampled = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % sample_every == 0:
            try:
                people = pose.detect_frame(frame)
                a_kp, b_kp = PoseExtractor._fallback_two(people)  # each (17, 3)
                a_has = float(a_kp[:, 2].sum()) > 0
                b_has = float(b_kp[:, 2].sum()) > 0
                frames_pose.append({
                    "frame_idx": fi,
                    "track_a": a_kp if a_has else None,
                    "track_b": b_kp if b_has else None,
                })
                sampled += 1
            except Exception:
                frames_pose.append({"frame_idx": fi, "track_a": None, "track_b": None})
        fi += 1
    cap.release()

    if sampled == 0:
        logger.error("No frames sampled from %s", video_path)
        return None

    # Detect fight window
    fs, fe = find_fight_window(frames_pose, actual_fps)
    logger.info("Video %s: %d frames sampled, fight window=[%d, %d] (%.1fs)",
                video_path.name, sampled, fs, fe, (fe - fs) / actual_fps)
    frames_fight = frames_pose[fs:fe + 1]

    # Build (T, 2, 17, 3) tensor for the fight window — missing tracks zero-filled.
    T = len(frames_fight)
    kp_seq = np.zeros((T, 2, 17, 3), dtype=np.float32)
    both_tracks_count = 0
    for t, fp in enumerate(frames_fight):
        if fp["track_a"] is not None:
            kp_seq[t, 0] = fp["track_a"]
        if fp["track_b"] is not None:
            kp_seq[t, 1] = fp["track_b"]
        if fp["track_a"] is not None and fp["track_b"] is not None:
            both_tracks_count += 1

    both_tracks_share = both_tracks_count / max(1, T)

    if both_tracks_count == 0:
        logger.warning("No valid both-track frames in fight window for %s", video_path.name)
        agg = {f"{k}_mean": np.nan for k in FEATURE_NAMES}
        agg.update({f"{k}_std": np.nan for k in FEATURE_NAMES})
    else:
        feats = compute_features(kp_seq)  # (T, F)
        # Restrict aggregation to frames where both tracks exist
        valid_mask = np.array([
            (fp["track_a"] is not None and fp["track_b"] is not None)
            for fp in frames_fight
        ], dtype=bool)
        valid_feats = feats[valid_mask].astype(np.float64)
        means = valid_feats.mean(axis=0)
        stds = valid_feats.std(axis=0)
        agg = {}
        for i, k in enumerate(FEATURE_NAMES):
            agg[f"{k}_mean"] = float(means[i])
            agg[f"{k}_std"] = float(stds[i])

    return {
        "video_path": str(video_path),
        "duration": duration,
        "n_frames_sampled": sampled,
        "fight_start_frame": fs,
        "fight_end_frame": fe,
        "fight_duration_s": (fe - fs) / actual_fps,
        "both_tracks_share": both_tracks_share,
        "n_frames_both_tracks": both_tracks_count,
        **agg,
    }


def _setup_logging(v: int) -> None:
    logging.basicConfig(
        level=logging.INFO if v else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_run(args: argparse.Namespace) -> int:
    t0 = time.time()
    feats = extract_one_video(Path(args.video), fps_target=args.fps_target)
    elapsed = time.time() - t0
    if feats is None:
        print(json.dumps({"error": "extract failed"}))
        return 1
    feats["video_id"] = args.video_id
    feats["bashoId"] = args.basho_id
    feats["day"] = args.day
    feats["matchNo"] = args.matchNo
    feats["eastId"] = args.east_id
    feats["westId"] = args.west_id
    feats["wall_time_s"] = elapsed
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([feats]).to_parquet(args.out, index=False)
    print(json.dumps({k: v for k, v in feats.items() if not k.endswith("_mean") and not k.endswith("_std")}, indent=2))
    logger.info("Saved %s (elapsed %.1fs)", args.out, elapsed)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Process multiple videos listed in an alignment parquet."""
    aln = pd.read_parquet(args.alignment)
    if args.bashos:
        aln = aln[aln["basho_id"].isin(args.bashos)]
    if args.limit:
        aln = aln.head(args.limit)
    logger.info("Processing %d videos", len(aln))

    rows = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(aln.itertuples(index=False)):
        video_path = Path(args.videos_dir) / f"{row.video_id}.mp4"
        if not video_path.exists():
            logger.warning("Missing video: %s", video_path)
            continue
        t0 = time.time()
        feats = extract_one_video(video_path, fps_target=args.fps_target)
        elapsed = time.time() - t0
        if feats is None:
            continue
        feats["video_id"] = row.video_id
        feats["bashoId"] = row.basho_id
        feats["day"] = row.day
        feats["matchNo"] = row.matchNo
        feats["eastId"] = row.eastId
        feats["westId"] = row.westId
        feats["wall_time_s"] = elapsed
        rows.append(feats)
        logger.info("[%d/%d] %s done in %.1fs (both_tracks=%.2f)",
                    i + 1, len(aln), row.video_id, elapsed, feats["both_tracks_share"])

    if rows:
        df = pd.DataFrame(rows)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-video pose extraction for official bout clips")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Single video")
    r.add_argument("--video", required=True)
    r.add_argument("--video-id", required=True)
    r.add_argument("--basho-id", required=True)
    r.add_argument("--day", type=int, required=True)
    r.add_argument("--matchNo", type=int, required=True)
    r.add_argument("--east-id", type=int, required=True)
    r.add_argument("--west-id", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--fps-target", type=float, default=10.0)
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="Process all videos in an alignment parquet")
    b.add_argument("--alignment", required=True)
    b.add_argument("--videos-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--out-dir", default="data/processed/official_pose")
    b.add_argument("--bashos", nargs="+", default=None)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--fps-target", type=float, default=10.0)
    b.add_argument("-v", "--verbose", action="count", default=1)
    b.set_defaults(func=cmd_batch)

    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
