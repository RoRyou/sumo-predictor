"""Caption-OCR-driven segmentation for sumo highlight reels.

Sumo highlight reels (e.g. "SUMO PRIME TIME") are montages: they cut
between many different bouts every 10-30 seconds.  At the end of each
bout the broadcast overlays a caption such as ``Winner FUJINOKAWA 1-0``.
A *change* in that caption is a perfect cut marker, and the text itself
maps each segment to a winner.

This module is intentionally pure-Python (easyocr + opencv) and CPU-only:
no Tesseract dependency.

Pipeline
--------
1. :class:`CaptionDetector` samples frames at a configurable fps (default
   2 fps), OCRs the lower third of each frame, and returns
   ``list[(t, text)]`` of the dominant caption string per sample.
2. :class:`SegmentBuilder` groups consecutive frames with similar
   captions (``difflib.SequenceMatcher`` ratio > 0.85) into segments.
   Returns :class:`Segment` records with timing + parsed winner/score.

CLI
---
::

    python -m src.features.caption_ocr scan \
        --video data/videos/smoke.mp4 --out reports/smoke_segments.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------
_LATIN_RE = re.compile(r"[A-Za-z]")
_WINNER_RE = re.compile(
    r"winner\s+([A-Z][A-Za-z'\-]{2,})",
    re.IGNORECASE,
)
_SCORE_RE = re.compile(r"\b(\d{1,2})\s*[-_/]\s*(\d{1,2})\b")


def _latin_ratio(s: str) -> float:
    if not s:
        return 0.0
    letters = _LATIN_RE.findall(s)
    return len(letters) / max(1, len(s))


def _normalise_text(s: str) -> str:
    """Lowercase, collapse whitespace, strip non-alnum-edges -- for grouping."""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 \-]+", "", s)
    return s.strip()


def parse_winner(text: str) -> str | None:
    """Pull the rikishi name out of ``Winner X`` style captions."""
    m = _WINNER_RE.search(text)
    if not m:
        return None
    name = m.group(1).upper()
    # easyocr often misreads -- accept anything 3+ chars
    if len(name) < 3:
        return None
    return name


def parse_score(text: str) -> str | None:
    m = _SCORE_RE.search(text)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


@dataclass
class CaptionSample:
    """One OCR sample taken at time ``t`` (seconds)."""

    t: float
    text: str
    raw_lines: list[str] = field(default_factory=list)


class CaptionDetector:
    """Sample frames at ``sample_fps`` and OCR the lower-third caption strip.

    Parameters
    ----------
    video_path
        Path to mp4.
    sample_fps
        How often to sample frames for OCR.  2 fps is a good default
        (captions linger > 0.5 s so we hit every transition).
    band
        ``(y_top_frac, y_bot_frac)`` -- vertical slice of the frame to OCR.
        Default lower third: ``(0.66, 1.0)``.
    languages
        easyocr language codes.  English only by default -- the captions
        we care about (``Winner X``, scoreboard) are Latin.
    gpu
        Forwarded to easyocr.  ``False`` keeps everything on CPU.
    min_chars
        Minimum length of the joined OCR result to be considered a
        non-empty caption.
    min_latin_ratio
        Reject samples where < this fraction of characters are letters
        (filters out junk like "11 / 11").
    """

    def __init__(
        self,
        video_path: str | Path,
        sample_fps: float = 2.0,
        band: tuple[float, float] = (0.66, 1.0),
        languages: list[str] | None = None,
        gpu: bool = False,
        min_chars: int = 4,
        min_latin_ratio: float = 0.5,
    ) -> None:
        self.video_path = Path(video_path)
        self.sample_fps = sample_fps
        self.band = band
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.min_chars = min_chars
        self.min_latin_ratio = min_latin_ratio
        self._reader: Any | None = None

    # ------------------------------------------------------------------
    def _get_reader(self) -> Any:
        if self._reader is None:
            import easyocr  # type: ignore

            logger.info(
                "Loading easyocr reader (langs=%s, gpu=%s)", self.languages, self.gpu
            )
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return self._reader

    # ------------------------------------------------------------------
    def _iter_samples(self) -> Any:
        """Yield ``(t_seconds, rgb_frame_band)`` at ``sample_fps``.

        Uses PyAV so it works whether or not ffmpeg is on PATH.
        """
        import av  # type: ignore

        container = av.open(str(self.video_path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        src_fps = float(stream.average_rate) if stream.average_rate else 30.0
        stride = max(1, int(round(src_fps / max(self.sample_fps, 1e-3))))
        y_top_frac, y_bot_frac = self.band

        for i, frame in enumerate(container.decode(stream)):
            if i % stride != 0:
                continue
            img = frame.to_ndarray(format="rgb24")
            H = img.shape[0]
            y0 = int(H * y_top_frac)
            y1 = int(H * y_bot_frac)
            band = img[y0:y1]
            t = i / src_fps
            yield t, band
        container.close()

    # ------------------------------------------------------------------
    def _ocr_band(self, band: np.ndarray) -> tuple[str, list[str]]:
        reader = self._get_reader()
        # easyocr returns list[(bbox, text, conf)]
        results = reader.readtext(band, detail=1, paragraph=False)
        lines = []
        for _bbox, text, conf in results:
            if conf is None or conf < 0.3:
                continue
            text = text.strip()
            if not text:
                continue
            lines.append(text)
        joined = " ".join(lines)
        return joined, lines

    # ------------------------------------------------------------------
    def scan(self, progress_every: int = 20) -> list[CaptionSample]:
        """OCR every sampled frame and return a list of :class:`CaptionSample`."""
        out: list[CaptionSample] = []
        for k, (t, band) in enumerate(self._iter_samples()):
            text, lines = self._ocr_band(band)
            if (
                len(text) >= self.min_chars
                and _latin_ratio(text) >= self.min_latin_ratio
            ):
                out.append(CaptionSample(t=float(t), text=text, raw_lines=lines))
            else:
                out.append(CaptionSample(t=float(t), text="", raw_lines=lines))
            if progress_every and (k + 1) % progress_every == 0:
                logger.info(
                    "OCR %d samples; latest t=%.2fs text=%r", k + 1, t, text[:60]
                )
        return out


# ----------------------------------------------------------------------
# Segment building
# ----------------------------------------------------------------------
@dataclass
class Segment:
    """A contiguous range of video over which the caption is stable.

    Attributes
    ----------
    t_start, t_end
        Inclusive-exclusive seconds in the source video.
    dominant_caption
        The most common caption string seen in the segment (raw OCR).
    winner_name
        Parsed rikishi name (uppercase) if the dominant caption matched
        ``Winner X``; otherwise ``None``.
    score
        Parsed bout score ``"k-m"`` if found.
    n_samples
        Number of OCR samples grouped into the segment.
    """

    t_start: float
    t_end: float
    dominant_caption: str
    winner_name: str | None = None
    score: str | None = None
    n_samples: int = 0

    def duration(self) -> float:
        return self.t_end - self.t_start


class SegmentBuilder:
    """Group consecutive :class:`CaptionSample` into bout-level segments.

    Two captions are deemed "the same caption" when
    ``SequenceMatcher.ratio() > sim_thresh``.  Empty captions are treated
    as "no signal" -- they extend the active segment but don't start a new
    one on their own.

    Parameters
    ----------
    sim_thresh
        Similarity ratio threshold (0..1).  0.85 is the spec default.
    min_segment_dur
        Drop segments shorter than this (filters single-frame noise).
    """

    def __init__(self, sim_thresh: float = 0.85, min_segment_dur: float = 0.0) -> None:
        self.sim_thresh = sim_thresh
        self.min_segment_dur = min_segment_dur

    @staticmethod
    def _similar(a: str, b: str, thresh: float) -> bool:
        if not a or not b:
            return False
        return SequenceMatcher(None, _normalise_text(a), _normalise_text(b)).ratio() >= thresh

    def build(
        self, samples: list[CaptionSample], video_end: float | None = None
    ) -> list[Segment]:
        if not samples:
            return []
        groups: list[list[CaptionSample]] = []
        anchor: str | None = None
        for s in samples:
            if anchor is None:
                if not s.text:
                    # nothing yet; seed an "empty" segment so timing covers
                    # the full video
                    groups.append([s])
                    anchor = ""
                    continue
                groups.append([s])
                anchor = s.text
                continue

            if not s.text:
                # extend current group with no signal
                groups[-1].append(s)
                continue

            if anchor == "" or not self._similar(anchor, s.text, self.sim_thresh):
                groups.append([s])
                anchor = s.text
            else:
                groups[-1].append(s)
                # refresh anchor to the most-recent good text so drift
                # accumulates gracefully
                anchor = s.text

        # Convert groups -> segments using midpoints between samples
        segments: list[Segment] = []
        ts = [g[0].t for g in groups]
        for i, group in enumerate(groups):
            t_start = ts[i] if i == 0 else (ts[i] + ts[i - 1]) / 2.0
            if i + 1 < len(groups):
                t_end = (ts[i] + ts[i + 1]) / 2.0
            else:
                t_end = video_end if video_end is not None else (group[-1].t + 0.5)

            texts = [s.text for s in group if s.text]
            if texts:
                # most common
                dom = max(set(texts), key=texts.count)
            else:
                dom = ""

            seg = Segment(
                t_start=float(t_start),
                t_end=float(t_end),
                dominant_caption=dom,
                winner_name=parse_winner(dom) if dom else None,
                score=parse_score(dom) if dom else None,
                n_samples=len(group),
            )
            segments.append(seg)

        if self.min_segment_dur > 0:
            segments = [s for s in segments if s.duration() >= self.min_segment_dur]
        return segments


# ----------------------------------------------------------------------
# Serialisation
# ----------------------------------------------------------------------
def segments_to_json(segments: list[Segment], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(s) for s in segments]
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def segments_from_json(in_path: str | Path) -> list[Segment]:
    data = json.loads(Path(in_path).read_text())
    return [Segment(**row) for row in data]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
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


def _cmd_scan(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    detector = CaptionDetector(
        video_path=args.video,
        sample_fps=args.fps,
    )
    samples = detector.scan()
    duration = _video_duration(args.video)
    builder = SegmentBuilder(sim_thresh=args.sim, min_segment_dur=args.min_dur)
    segments = builder.build(samples, video_end=duration)
    out_path = segments_to_json(segments, args.out)
    logger.info(
        "Wrote %d segments (of %d samples) to %s", len(segments), len(samples), out_path
    )
    for i, s in enumerate(segments):
        logger.info(
            "  seg %02d  %.2fs-%.2fs  winner=%s  score=%s  cap=%r",
            i,
            s.t_start,
            s.t_end,
            s.winner_name,
            s.score,
            s.dominant_caption[:60],
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Caption-OCR scene segmenter")
    sp = p.add_subparsers(dest="cmd", required=True)
    scan = sp.add_parser("scan", help="Scan a video for caption-driven segments")
    scan.add_argument("--video", required=True, type=Path)
    scan.add_argument("--out", required=True, type=Path)
    scan.add_argument("--fps", type=float, default=2.0, help="OCR sampling fps")
    scan.add_argument("--sim", type=float, default=0.85, help="similarity threshold")
    scan.add_argument(
        "--min-dur", type=float, default=0.0, help="drop segments shorter than this (s)"
    )
    scan.set_defaults(func=_cmd_scan)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
