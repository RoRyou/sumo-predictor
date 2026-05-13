"""Visual scene-cut detection + merge with caption-driven segments.

A thin wrapper around `PySceneDetect <https://www.scenedetect.com>`_
that gives us the *visual* cut signal (raw RGB-content delta), which we
combine with the caption-OCR signal in :mod:`src.features.caption_ocr`.

Merge logic
-----------
A "bout segment" boundary is the union of:

* a visual cut from PySceneDetect, AND
* a caption-change boundary from
  :class:`src.features.caption_ocr.SegmentBuilder`.

Boundaries closer than ``merge_eps`` seconds are collapsed; segments
shorter than ``min_dur`` (default 3.0 s) are dropped to filter the
intro/outro/transition shots.

CLI
---
::

    python -m src.features.scene_cuts scan --video data/videos/smoke.mp4
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@dataclass
class CutSegment:
    """A merged segment with provenance info."""

    t_start: float
    t_end: float
    source: str  # "visual" | "caption" | "merged"

    def duration(self) -> float:
        return self.t_end - self.t_start


# ----------------------------------------------------------------------
# Visual cuts via PySceneDetect
# ----------------------------------------------------------------------
def detect_visual_cuts(
    video_path: str | Path,
    threshold: float = 27.0,
    min_scene_len: float = 0.5,
) -> list[tuple[float, float]]:
    """Return list of ``(t_start, t_end)`` for every visual scene.

    Parameters
    ----------
    video_path
        mp4 path.
    threshold
        Content-detector threshold; lower => more cuts.  PySceneDetect's
        default is 27.0.
    min_scene_len
        Minimum scene length in seconds (PySceneDetect will reject
        scenes shorter than this even with a content change).
    """
    from scenedetect import ContentDetector, SceneManager, open_video  # type: ignore

    video = open_video(str(video_path))
    sm = SceneManager()
    # min_scene_len is expressed in frames in older scenedetect API; pass
    # via frames count when possible.
    try:
        sm.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=int(min_scene_len * 30))
        )
    except TypeError:
        sm.add_detector(ContentDetector(threshold=threshold))

    sm.detect_scenes(video=video, show_progress=False)
    scenes = sm.get_scene_list()
    out: list[tuple[float, float]] = []
    for start, end in scenes:
        # newer scenedetect exposes `.seconds`; older versions have
        # `get_seconds()`.  Try the new API first.
        try:
            s0 = float(start.seconds)
            s1 = float(end.seconds)
        except AttributeError:  # pragma: no cover
            s0 = float(start.get_seconds())
            s1 = float(end.get_seconds())
        out.append((s0, s1))
    if not out:
        # Whole video as one scene -- happens for short clips
        # with no content cuts found.
        out = [(0.0, _video_duration(video_path))]
    return out


def _video_duration(path: str | Path) -> float:
    import av  # type: ignore

    c = av.open(str(path))
    try:
        if c.duration is not None:
            return float(c.duration) / av.time_base
        stream = c.streams.video[0]
        return float(stream.frames) / float(stream.average_rate or 30.0)
    finally:
        c.close()


# ----------------------------------------------------------------------
# Merge logic
# ----------------------------------------------------------------------
def _dedupe_boundaries(
    times: Iterable[float], merge_eps: float
) -> list[float]:
    s = sorted(times)
    out: list[float] = []
    for t in s:
        if out and (t - out[-1]) <= merge_eps:
            continue
        out.append(t)
    return out


def merge_segments(
    video_path: str | Path,
    caption_segments: list[tuple[float, float]] | None,
    visual_segments: list[tuple[float, float]] | None = None,
    *,
    threshold: float = 27.0,
    merge_eps: float = 0.5,
    min_dur: float = 3.0,
) -> list[CutSegment]:
    """Combine caption-change and visual-cut boundaries.

    Parameters
    ----------
    video_path
        Source mp4 (used to read the duration if neither cut list covers
        the tail of the video).
    caption_segments
        Output of caption-OCR segment builder, as ``(t_start, t_end)``.
        Pass ``None`` to fall back to visual-only segmentation.
    visual_segments
        Pre-computed visual scene list, or ``None`` to call
        :func:`detect_visual_cuts` ourselves.
    threshold
        Forwarded when ``visual_segments`` is ``None``.
    merge_eps
        Collapse boundaries within this distance (s).
    min_dur
        Drop final merged segments shorter than this (filters intros).
    """
    if visual_segments is None:
        visual_segments = detect_visual_cuts(video_path, threshold=threshold)

    duration = _video_duration(video_path)

    # Collect boundary timestamps from both signals.
    boundaries: list[float] = [0.0, duration]
    for t0, t1 in visual_segments:
        boundaries.append(t0)
        boundaries.append(t1)
    if caption_segments:
        for t0, t1 in caption_segments:
            boundaries.append(t0)
            boundaries.append(t1)

    boundaries = _dedupe_boundaries(boundaries, merge_eps)
    # Build segments between consecutive boundaries
    raw: list[CutSegment] = []
    for i in range(len(boundaries) - 1):
        raw.append(
            CutSegment(
                t_start=float(boundaries[i]),
                t_end=float(boundaries[i + 1]),
                source="merged" if caption_segments else "visual",
            )
        )
    # Drop short segments
    return [s for s in raw if s.duration() >= min_dur]


def scan(
    video_path: str | Path,
    *,
    threshold: float = 27.0,
    min_dur: float = 3.0,
) -> list[CutSegment]:
    """Convenience: visual-only segmentation, filtered by ``min_dur``."""
    visuals = detect_visual_cuts(video_path, threshold=threshold)
    return [
        CutSegment(t_start=a, t_end=b, source="visual")
        for (a, b) in visuals
        if (b - a) >= min_dur
    ]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _cmd_scan(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    segs = scan(args.video, threshold=args.threshold, min_dur=args.min_dur)
    logger.info("Detected %d visual segments (>= %.1fs):", len(segs), args.min_dur)
    for i, s in enumerate(segs):
        logger.info(
            "  seg %02d  %.2fs-%.2fs  dur=%.2fs", i, s.t_start, s.t_end, s.duration()
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visual scene-cut detector")
    sp = p.add_subparsers(dest="cmd", required=True)
    scan_p = sp.add_parser("scan", help="Scan a video for visual scene cuts")
    scan_p.add_argument("--video", required=True, type=Path)
    scan_p.add_argument("--threshold", type=float, default=27.0)
    scan_p.add_argument("--min-dur", type=float, default=3.0)
    scan_p.set_defaults(func=_cmd_scan)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
