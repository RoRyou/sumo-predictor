"""Per-bout pose-feature extraction for highlight-reel videos.

For each detected segment we:

1. Read the video frames inside ``[t_start, t_end]`` at a target fps.
2. Run the existing :class:`src.features.pose.PoseExtractor` (+ a fresh
   :class:`src.features.tracking.TwoRikishiTracker` per segment so ID
   locks don't leak across cuts).
3. Compute the 40-dim kinematic features with
   :func:`src.features.kinematics.compute_features`.
4. Aggregate to a single row: mean + std of every feature over the
   segment, plus a ``both_tracks_share`` quality metric.

CLI
---
::

    python -m src.features.extract_bout_features run \
        --video data/videos/smoke.mp4 \
        --segments-out reports/smoke_segments.json \
        --features-out data/processed/pose_features_smoke_segments.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.features.caption_ocr import (
    CaptionDetector,
    Segment,
    SegmentBuilder,
    segments_to_json,
)
from src.features.kinematics import FEATURE_NAMES, compute_features
from src.features.scene_cuts import detect_visual_cuts, merge_segments

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Frame slicing
# ----------------------------------------------------------------------
def _read_segment_frames(
    video_path: str | Path,
    t_start: float,
    t_end: float,
    target_fps: float = 15.0,
) -> tuple[np.ndarray, float]:
    """Decode frames in ``[t_start, t_end)`` at approx ``target_fps``.

    Returns ``(frames (T,H,W,3) uint8, effective_fps)``.
    """
    import av  # type: ignore

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    src_fps = float(stream.average_rate) if stream.average_rate else 30.0
    stride = max(1, int(round(src_fps / max(target_fps, 1e-3))))
    eff_fps = src_fps / stride

    frames: list[np.ndarray] = []
    for i, frame in enumerate(container.decode(stream)):
        t = i / src_fps
        if t < t_start:
            continue
        if t >= t_end:
            break
        if i % stride != 0:
            continue
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8), eff_fps
    return np.stack(frames, axis=0), eff_fps


# ----------------------------------------------------------------------
# Pose runner (lazy import so tests can monkey-patch easily)
# ----------------------------------------------------------------------
def _run_pose_pipeline(
    frames: np.ndarray, fps: float, extractor: Any | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(kp_seq (T,2,17,3), features (T, FEATURE_DIM))``."""
    from src.features.kinematics import KinematicsConfig
    from src.features.pose import PoseExtractor
    from src.features.tracking import TwoRikishiTracker

    if extractor is None:
        extractor = PoseExtractor()
    tracker = TwoRikishiTracker(frame_rate=fps)
    kp_seq = extractor.extract(frames, tracker=tracker)
    feats = compute_features(kp_seq, KinematicsConfig(fps=fps))
    return kp_seq, feats


# ----------------------------------------------------------------------
# Quality metric
# ----------------------------------------------------------------------
def both_tracks_share(kp_seq: np.ndarray, conf_thresh: float = 0.2) -> float:
    """Fraction of frames in which BOTH track slots have any keypoint
    above ``conf_thresh``.

    Used as a sanity gauge: under highlight-reel cuts the unsegmented
    share was 0.3%; per-segment we expect ≥10x higher (=> ≥3-5%).
    """
    if kp_seq.size == 0:
        return 0.0
    # max conf per person per frame
    pmax = kp_seq[:, :, :, 2].max(axis=2)  # (T, 2)
    both = ((pmax[:, 0] >= conf_thresh) & (pmax[:, 1] >= conf_thresh)).mean()
    return float(both)


# ----------------------------------------------------------------------
# Per-segment aggregation
# ----------------------------------------------------------------------
def aggregate_segment_row(
    video_id: str,
    segment_idx: int,
    segment: dict,
    features: np.ndarray,
    kp_seq: np.ndarray,
) -> dict[str, Any]:
    """Compute mean/std aggregates + metadata for one segment."""
    row: dict[str, Any] = {
        "video_id": video_id,
        "segment_idx": int(segment_idx),
        "t_start": float(segment["t_start"]),
        "t_end": float(segment["t_end"]),
        "winner_name": segment.get("winner_name"),
        "score": segment.get("score"),
        "dominant_caption": segment.get("dominant_caption"),
        "n_frames": int(features.shape[0]),
        "both_tracks_share": both_tracks_share(kp_seq),
    }
    if features.shape[0] == 0:
        for name in FEATURE_NAMES:
            row[f"{name}_mean"] = 0.0
            row[f"{name}_std"] = 0.0
        return row
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    for j, name in enumerate(FEATURE_NAMES):
        row[f"{name}_mean"] = float(means[j])
        row[f"{name}_std"] = float(stds[j])
    return row


# ----------------------------------------------------------------------
# End-to-end driver
# ----------------------------------------------------------------------
def detect_segments(
    video_path: str | Path,
    *,
    use_ocr: bool = True,
    sample_fps: float = 2.0,
    sim_thresh: float = 0.85,
    visual_threshold: float = 27.0,
    min_dur: float = 3.0,
) -> tuple[list[dict], dict[str, Any]]:
    """Run caption-OCR + visual scene-cut and return merged segments.

    Returns
    -------
    segments
        list of dicts (asdict of :class:`Segment`-like).
    meta
        ``{"ocr_segments": [...], "visual_segments": [...]}`` for debug.
    """
    from src.features.caption_ocr import _video_duration

    duration = _video_duration(video_path)

    ocr_segments: list[Segment] = []
    if use_ocr:
        try:
            detector = CaptionDetector(video_path, sample_fps=sample_fps)
            samples = detector.scan()
            builder = SegmentBuilder(sim_thresh=sim_thresh)
            ocr_segments = builder.build(samples, video_end=duration)
            logger.info("OCR yielded %d raw caption segments", len(ocr_segments))
        except Exception as exc:  # pragma: no cover
            logger.warning("Caption OCR failed (%s); falling back to visual-only", exc)
            ocr_segments = []

    visuals = detect_visual_cuts(video_path, threshold=visual_threshold)

    cap_ranges = [(s.t_start, s.t_end) for s in ocr_segments] if ocr_segments else None
    merged = merge_segments(
        video_path,
        caption_segments=cap_ranges,
        visual_segments=visuals,
        threshold=visual_threshold,
        min_dur=min_dur,
    )

    # Attach the dominant caption / winner / score from the overlapping
    # OCR segment (the one with maximum temporal overlap).
    def _overlap_winner(t0: float, t1: float) -> tuple[str, str | None, str | None]:
        if not ocr_segments:
            return "", None, None
        best_ov = 0.0
        best_idx = -1
        for i, s in enumerate(ocr_segments):
            ov = max(0.0, min(t1, s.t_end) - max(t0, s.t_start))
            if ov > best_ov:
                best_ov = ov
                best_idx = i
        if best_idx < 0:
            return "", None, None
        s = ocr_segments[best_idx]
        return s.dominant_caption, s.winner_name, s.score

    out_segments: list[dict] = []
    for c in merged:
        cap, win, score = _overlap_winner(c.t_start, c.t_end)
        out_segments.append(
            {
                "t_start": c.t_start,
                "t_end": c.t_end,
                "dominant_caption": cap,
                "winner_name": win,
                "score": score,
                "source": c.source,
            }
        )

    meta = {
        "ocr_segments": [asdict(s) for s in ocr_segments],
        "visual_segments": [{"t_start": a, "t_end": b} for a, b in visuals],
        "duration": duration,
    }
    return out_segments, meta


def run(
    video_path: str | Path,
    segments_out: str | Path,
    features_out: str | Path,
    *,
    use_ocr: bool = True,
    target_fps: float = 15.0,
    visual_threshold: float = 27.0,
    min_dur: float = 3.0,
    video_id: str | None = None,
    pose_extractor: Any | None = None,
) -> tuple[Path, Path]:
    """End-to-end: segment → pose → aggregated parquet."""
    import pandas as pd

    video_path = Path(video_path)
    if video_id is None:
        video_id = video_path.stem

    segments, meta = detect_segments(
        video_path,
        use_ocr=use_ocr,
        visual_threshold=visual_threshold,
        min_dur=min_dur,
    )
    logger.info("Final merged segments (>= %.1fs): %d", min_dur, len(segments))

    payload = {"video_id": video_id, "segments": segments, "meta": meta}
    Path(segments_out).parent.mkdir(parents=True, exist_ok=True)
    Path(segments_out).write_text(json.dumps(payload, indent=2))

    rows: list[dict] = []
    for idx, seg in enumerate(segments):
        logger.info(
            "Processing segment %d/%d  %.2fs-%.2fs  winner=%s",
            idx + 1,
            len(segments),
            seg["t_start"],
            seg["t_end"],
            seg.get("winner_name"),
        )
        frames, fps = _read_segment_frames(
            video_path, seg["t_start"], seg["t_end"], target_fps=target_fps
        )
        if frames.shape[0] < 3:
            logger.warning("  too few frames (%d); skipping pose", frames.shape[0])
            kp_seq = np.zeros((0, 2, 17, 3), dtype=np.float32)
            feats = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        else:
            kp_seq, feats = _run_pose_pipeline(frames, fps, extractor=pose_extractor)
        row = aggregate_segment_row(video_id, idx, seg, feats, kp_seq)
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(features_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_out, index=False)
    logger.info("Wrote %d rows to %s", len(df), features_out)
    return Path(segments_out), Path(features_out)


# ----------------------------------------------------------------------
# Overlay visualisation
# ----------------------------------------------------------------------
def save_segments_overlay(
    video_path: str | Path,
    segments: list[dict],
    visual_cuts: list[tuple[float, float]],
    out_path: str | Path,
) -> Path:
    """Render a timeline plot of merged segments + visual cuts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.features.caption_ocr import _video_duration

    duration = _video_duration(video_path)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 3)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["merged", "visual", "OCR/winner"])
    ax.set_xlabel("time (s)")
    ax.set_title(f"Segment overlay -- {Path(video_path).name}")

    for v0, v1 in visual_cuts:
        ax.axvspan(v0, v1, ymin=0.34, ymax=0.66, alpha=0.3, color="tab:orange")
    for s in segments:
        ax.axvspan(
            s["t_start"], s["t_end"], ymin=0.02, ymax=0.32, alpha=0.4, color="tab:blue"
        )
        label = s.get("winner_name") or "?"
        ax.text(
            (s["t_start"] + s["t_end"]) / 2,
            2.5,
            label,
            ha="center",
            fontsize=9,
            color="black",
        )
        ax.axvline(s["t_start"], color="black", lw=0.6, alpha=0.5)
    ax.axvline(duration, color="black", lw=0.6, alpha=0.5)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seg_path, feat_path = run(
        args.video,
        args.segments_out,
        args.features_out,
        use_ocr=not args.no_ocr,
        target_fps=args.fps,
        visual_threshold=args.scene_threshold,
        min_dur=args.min_dur,
    )
    # save overlay
    if args.overlay:
        data = json.loads(Path(seg_path).read_text())
        visuals = [
            (m["t_start"], m["t_end"])
            for m in data.get("meta", {}).get("visual_segments", [])
        ]
        save_segments_overlay(args.video, data["segments"], visuals, args.overlay)
    logger.info("Done; segments=%s features=%s", seg_path, feat_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-bout pose feature extractor")
    sp = p.add_subparsers(dest="cmd", required=True)
    run_p = sp.add_parser("run", help="Detect segments and extract pose features")
    run_p.add_argument("--video", required=True, type=Path)
    run_p.add_argument("--segments-out", required=True, type=Path)
    run_p.add_argument("--features-out", required=True, type=Path)
    run_p.add_argument("--fps", type=float, default=15.0, help="pose-pipeline target fps")
    run_p.add_argument("--scene-threshold", type=float, default=27.0)
    run_p.add_argument("--min-dur", type=float, default=3.0)
    run_p.add_argument("--no-ocr", action="store_true", help="skip caption OCR (visual-only)")
    run_p.add_argument("--overlay", type=Path, default=None)
    run_p.set_defaults(func=_cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
