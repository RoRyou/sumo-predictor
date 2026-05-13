"""Video acquisition + frame extraction helpers for Phase 2 (Route B).

This module provides thin wrappers around ``yt-dlp`` (to pull short clips
from YouTube) and ``imageio``/``PyAV`` (to read frames at a target fps).

It is intentionally light on validation -- Phase 2 only needs a single
smoke-test clip.  Callers are expected to be batch driver scripts later.

Example
-------
>>> from src.data.video_loader import download_clip, read_frames
>>> path = download_clip(
...     "https://www.youtube.com/watch?v=XYZ",
...     start="00:00:10", end="00:00:55",
...     out_path="data/videos/smoke.mp4",
... )
>>> frames, fps = read_frames(path, target_fps=15)  # (T, H, W, 3) uint8
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# yt-dlp wrappers
# ----------------------------------------------------------------------
def _yt_dlp_bin() -> str:
    exe = shutil.which("yt-dlp")
    if exe is None:
        # fall back to the python module entrypoint
        return "python -m yt_dlp"
    return exe


def download_clip(
    url: str,
    out_path: str | Path,
    start: str | None = None,
    end: str | None = None,
    fmt: str = "mp4",
    height_cap: int = 720,
    overwrite: bool = False,
) -> Path:
    """Download a clip (optionally a single time-range slice) via yt-dlp.

    Parameters
    ----------
    url
        YouTube URL.
    out_path
        Destination file path (parent dirs created).  Extension forced to ``.mp4``.
    start, end
        ``HH:MM:SS`` strings.  Both required for slicing; if either is None
        the whole video is downloaded.
    fmt
        Preferred container.  Default ``mp4`` for ffmpeg compatibility.
    height_cap
        Pick the smallest stream at or above this many pixels of height
        when one exists; otherwise the best available.
    overwrite
        If False and the file exists, skip the download.

    Returns
    -------
    Path
        Absolute path to the downloaded file.
    """
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        logger.info("Reusing existing clip at %s", out_path)
        return out_path

    sel = (
        f"bestvideo[height<={height_cap}][ext={fmt}]+bestaudio[ext=m4a]/"
        f"best[height<={height_cap}]/best"
    )

    cmd: list[str] = _yt_dlp_bin().split() + [
        "-f",
        sel,
        "--merge-output-format",
        fmt,
        "-o",
        str(out_path),
        "--no-playlist",
        "--no-warnings",
        "--quiet",
        "--no-progress",
    ]
    if start and end:
        cmd += ["--download-sections", f"*{start}-{end}"]
    cmd.append(url)

    logger.info("yt-dlp: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (rc={res.returncode}): "
            f"stdout={res.stdout[-400:]!r} stderr={res.stderr[-400:]!r}"
        )
    if not out_path.exists():
        # yt-dlp sometimes adds an extra extension; try to find sibling.
        siblings = list(out_path.parent.glob(out_path.stem + ".*"))
        if siblings:
            siblings[0].rename(out_path)
        else:
            raise RuntimeError(f"yt-dlp returned 0 but {out_path} not found")
    return out_path


# ----------------------------------------------------------------------
# Frame extraction
# ----------------------------------------------------------------------
def read_frames(
    path: str | Path,
    target_fps: float = 15.0,
    max_frames: int | None = None,
) -> tuple[np.ndarray, float]:
    """Decode a video to (T,H,W,3) uint8 RGB at approximately ``target_fps``.

    Implementation prefers PyAV; falls back to imageio if PyAV is missing.

    Returns
    -------
    frames : np.ndarray (T, H, W, 3) uint8 RGB
    fps : float
        Effective output fps after subsampling.
    """
    path = Path(path)
    try:
        import av  # type: ignore
    except ImportError:
        av = None

    if av is not None:
        return _read_frames_pyav(path, target_fps, max_frames)
    return _read_frames_imageio(path, target_fps, max_frames)


def _read_frames_pyav(
    path: Path, target_fps: float, max_frames: int | None
) -> tuple[np.ndarray, float]:
    import av  # type: ignore

    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    src_fps = float(stream.average_rate) if stream.average_rate else 30.0
    stride = max(1, int(round(src_fps / max(target_fps, 1e-3))))
    eff_fps = src_fps / stride

    out: list[np.ndarray] = []
    for i, frame in enumerate(container.decode(stream)):
        if i % stride != 0:
            continue
        out.append(frame.to_ndarray(format="rgb24"))
        if max_frames is not None and len(out) >= max_frames:
            break
    container.close()
    if not out:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8), 0.0
    return np.stack(out, axis=0), eff_fps


def _read_frames_imageio(
    path: Path, target_fps: float, max_frames: int | None
) -> tuple[np.ndarray, float]:
    import imageio.v3 as iio  # type: ignore

    meta = iio.immeta(str(path), plugin="pyav") if hasattr(iio, "immeta") else {}
    src_fps = float(meta.get("fps", 30.0))
    stride = max(1, int(round(src_fps / max(target_fps, 1e-3))))
    eff_fps = src_fps / stride

    out: list[np.ndarray] = []
    for i, frame in enumerate(iio.imiter(str(path))):
        if i % stride != 0:
            continue
        out.append(np.asarray(frame))
        if max_frames is not None and len(out) >= max_frames:
            break
    if not out:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8), 0.0
    return np.stack(out, axis=0), eff_fps


def iter_frames(path: str | Path, target_fps: float = 15.0) -> Iterable[np.ndarray]:
    """Generator variant of :func:`read_frames`."""
    frames, _ = read_frames(path, target_fps=target_fps)
    yield from frames
