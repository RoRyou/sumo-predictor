"""Tests for caption-OCR / scene-cut / per-bout-feature pipeline.

These tests do not require real video or heavy model downloads; the OCR
and pose extractors are monkey-patched.  The one "fixture frame" test
synthesises an image with PIL so we can assert OCR returns something
plausible IF easyocr is available (skipped otherwise).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.features.caption_ocr import (
    CaptionDetector,
    CaptionSample,
    Segment,
    SegmentBuilder,
    parse_score,
    parse_winner,
    segments_to_json,
)
from src.features.extract_bout_features import (
    aggregate_segment_row,
    both_tracks_share,
)
from src.features.kinematics import FEATURE_NAMES


# ---------------------------------------------------------------------- #
# Pure parser tests (no easyocr dependency)
# ---------------------------------------------------------------------- #
def test_parse_winner_matches_canonical_caption():
    assert parse_winner("Winner FUJINOKAWA 1-0") == "FUJINOKAWA"
    assert parse_winner("winner   hoshoryu  3-2") == "HOSHORYU"


def test_parse_winner_returns_none_for_unrelated_text():
    assert parse_winner("Tsukidashi frontal thrust out") is None
    assert parse_winner("") is None


def test_parse_score_handles_dash_and_slash():
    assert parse_score("Winner X 12-7") == "12-7"
    assert parse_score("Day 5  3/2") == "3-2"
    assert parse_score("no digits here") is None


# ---------------------------------------------------------------------- #
# SegmentBuilder groups consecutive frames with similar captions
# ---------------------------------------------------------------------- #
def test_segment_builder_groups_similar_consecutive_captions():
    samples = [
        CaptionSample(t=0.0, text=""),
        CaptionSample(t=0.5, text=""),
        CaptionSample(t=1.0, text="Winner FUJINOKAWA 1-0"),
        CaptionSample(t=1.5, text="Winner FUJINOKAWA 1-0"),
        CaptionSample(t=2.0, text="Winner FUJINOKAWA  1-0 "),
        CaptionSample(t=2.5, text=""),
        CaptionSample(t=3.0, text="Winner HOSHORYU 2-1"),
        CaptionSample(t=3.5, text="Winner HOSHORYU 2-1"),
    ]
    builder = SegmentBuilder(sim_thresh=0.85)
    segments = builder.build(samples, video_end=4.0)
    # 3 groups: empty intro, FUJINOKAWA cluster, HOSHORYU cluster
    assert len(segments) == 3
    assert segments[0].winner_name is None
    assert segments[1].winner_name == "FUJINOKAWA"
    assert segments[1].score == "1-0"
    assert segments[2].winner_name == "HOSHORYU"
    # Segments are non-overlapping and cover [0, video_end]
    assert segments[0].t_start == pytest.approx(0.0)
    assert segments[-1].t_end == pytest.approx(4.0)
    for a, b in zip(segments[:-1], segments[1:]):
        assert a.t_end == pytest.approx(b.t_start, abs=1e-6)


def test_segment_builder_drops_short_segments_when_requested():
    samples = [
        CaptionSample(t=0.0, text="A A A A"),
        CaptionSample(t=0.2, text="A A A A"),
        CaptionSample(t=0.3, text="totally different caption text"),
        CaptionSample(t=5.0, text="totally different caption text"),
    ]
    segs = SegmentBuilder(min_segment_dur=1.0).build(samples, video_end=6.0)
    # First A-cluster spans [0, ~0.25s] -> dropped; second cluster kept.
    assert all(s.duration() >= 1.0 for s in segs)


def test_segment_builder_handles_empty_input():
    assert SegmentBuilder().build([], video_end=10.0) == []


# ---------------------------------------------------------------------- #
# OCR returns plausible captions on a synthetic fixture frame
# ---------------------------------------------------------------------- #
def _make_caption_frame(text: str) -> np.ndarray:
    """Draw ``text`` in white on a black 200x800 strip (returns RGB uint8)."""
    pil = pytest.importorskip("PIL.Image")
    from PIL import ImageDraw, ImageFont

    img = pil.new("RGB", (800, 200), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 60)
    except OSError:
        font = ImageFont.load_default()
    draw.text((30, 60), text, fill=(255, 255, 255), font=font)
    return np.array(img)


@pytest.mark.slow
def test_ocr_reads_synthetic_caption_frame():
    """OCR returns the rikishi name on a clean synthetic caption.

    Marked ``slow`` because easyocr lazily downloads ~100 MB of weights.
    Skipped if easyocr can't load.
    """
    easyocr = pytest.importorskip("easyocr")
    try:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"easyocr unavailable: {exc}")

    frame = _make_caption_frame("Winner FUJINOKAWA 1-0")
    results = reader.readtext(frame, detail=1, paragraph=False)
    joined = " ".join(r[1] for r in results)
    # Allow some OCR slop -- but the rikishi name should appear
    assert "FUJI" in joined.upper(), joined


# ---------------------------------------------------------------------- #
# extract_bout_features schema (pose monkey-patched)
# ---------------------------------------------------------------------- #
def test_both_tracks_share_metric():
    # All-zeros => share is 0
    kp = np.zeros((10, 2, 17, 3), dtype=np.float32)
    assert both_tracks_share(kp) == 0.0
    # Half frames have both visible
    kp[:5, :, :, 2] = 0.9
    assert both_tracks_share(kp) == pytest.approx(0.5)


def test_aggregate_segment_row_schema():
    feats = np.random.default_rng(0).standard_normal((30, len(FEATURE_NAMES))).astype(
        np.float32
    )
    kp = np.zeros((30, 2, 17, 3), dtype=np.float32)
    kp[:, :, :, 2] = 0.5
    seg = {
        "t_start": 1.0,
        "t_end": 5.0,
        "winner_name": "FOO",
        "score": "1-0",
        "dominant_caption": "Winner FOO 1-0",
    }
    row = aggregate_segment_row(
        video_id="vid", segment_idx=3, segment=seg, features=feats, kp_seq=kp
    )
    # Required keys
    assert row["video_id"] == "vid"
    assert row["segment_idx"] == 3
    assert row["t_start"] == 1.0
    assert row["t_end"] == 5.0
    assert row["winner_name"] == "FOO"
    assert row["score"] == "1-0"
    assert row["n_frames"] == 30
    assert row["both_tracks_share"] == pytest.approx(1.0)
    # Mean & std columns present for every base feature
    for name in FEATURE_NAMES:
        assert f"{name}_mean" in row
        assert f"{name}_std" in row
    # Total schema cardinality: meta(9) + 2*FEATURE_NAMES
    assert len(row) == 9 + 2 * len(FEATURE_NAMES)


def test_run_writes_parquet_with_mocked_pose(tmp_path, monkeypatch):
    """End-to-end ``run()`` schema check with all heavy steps stubbed."""
    pd = pytest.importorskip("pandas")

    from src.features import extract_bout_features as ebf

    # Fake "detected" segments -- no real video needed
    fake_segments = [
        {
            "t_start": 0.0,
            "t_end": 4.0,
            "winner_name": "A",
            "score": "1-0",
            "dominant_caption": "Winner A 1-0",
            "source": "merged",
        },
        {
            "t_start": 4.0,
            "t_end": 8.0,
            "winner_name": None,
            "score": None,
            "dominant_caption": "",
            "source": "visual",
        },
    ]
    fake_meta = {"ocr_segments": [], "visual_segments": [], "duration": 8.0}

    monkeypatch.setattr(
        ebf, "detect_segments", lambda *a, **k: (fake_segments, fake_meta)
    )
    # Skip actual frame decoding -- return dummy frames
    monkeypatch.setattr(
        ebf,
        "_read_segment_frames",
        lambda video_path, t0, t1, target_fps=15.0: (
            np.zeros((20, 64, 64, 3), dtype=np.uint8),
            15.0,
        ),
    )

    def _fake_pose(frames, fps, extractor=None):
        T = frames.shape[0]
        kp = np.zeros((T, 2, 17, 3), dtype=np.float32)
        kp[:, :, :, 2] = 0.7  # high conf everywhere
        feats = np.ones((T, len(FEATURE_NAMES)), dtype=np.float32) * 0.5
        return kp, feats

    monkeypatch.setattr(ebf, "_run_pose_pipeline", _fake_pose)

    video = tmp_path / "fake.mp4"
    video.write_bytes(b"")  # path exists, never opened
    segs_out = tmp_path / "segs.json"
    feat_out = tmp_path / "features.parquet"

    ebf.run(video, segs_out, feat_out, use_ocr=False, video_id="fake")

    assert segs_out.exists() and feat_out.exists()
    df = pd.read_parquet(feat_out)
    assert len(df) == 2
    assert set(["video_id", "segment_idx", "t_start", "t_end", "winner_name", "score",
                "n_frames", "both_tracks_share"]).issubset(df.columns)
    # mean columns present
    assert "A_com_x_n_mean" in df.columns
    assert "A_com_x_n_std" in df.columns
    # winner name preserved
    assert df.iloc[0]["winner_name"] == "A"
    assert df.iloc[1]["winner_name"] is None or df.iloc[1]["winner_name"] != df.iloc[1]["winner_name"]


def test_segments_to_json_roundtrip(tmp_path):
    segs = [
        Segment(t_start=0.0, t_end=2.0, dominant_caption="Winner X 1-0",
                winner_name="X", score="1-0", n_samples=4),
        Segment(t_start=2.0, t_end=5.0, dominant_caption="", n_samples=2),
    ]
    p = segments_to_json(segs, tmp_path / "s.json")
    loaded = json.loads(p.read_text())
    assert len(loaded) == 2
    assert loaded[0]["winner_name"] == "X"
    assert loaded[1]["dominant_caption"] == ""


# ---------------------------------------------------------------------- #
# CaptionDetector construction (no scan() to avoid model download)
# ---------------------------------------------------------------------- #
def test_caption_detector_construction(tmp_path):
    p = tmp_path / "dummy.mp4"
    p.write_bytes(b"")
    det = CaptionDetector(p, sample_fps=2.0, band=(0.6, 1.0))
    assert det.sample_fps == 2.0
    assert det.band == (0.6, 1.0)
    assert det.languages == ["en"]
