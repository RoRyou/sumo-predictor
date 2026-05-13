"""Two-rikishi tracking on top of YOLOv8-pose detections.

We use ``supervision.ByteTrack`` to obtain stable track IDs across the
clip, then *lock* exactly two IDs as person A (east, bout-left in the
dohyo camera view) and person B (west, bout-right) using the initial
left/right positions on the frame they first appear.

For the smoke-test scenario the camera is fixed and there are only two
people on the dohyo, so this heuristic works reliably.  For a full
pipeline we'd add the perspective-corrected dohyo coordinate prior
(readme §5.5).

Usage
-----
>>> tracker = TwoRikishiTracker()
>>> kp_seq = pose_extractor.extract(frames, tracker=tracker)  # (T, 2, 17, 3)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.features.pose import FramePeople

logger = logging.getLogger(__name__)


class TwoRikishiTracker:
    """Online tracker that locks IDs to A/B by left/right convention.

    Parameters
    ----------
    track_activation_threshold
        Forward to ``supervision.ByteTrack``.
    minimum_matching_threshold
        Forward to ``supervision.ByteTrack``.
    frame_rate
        Effective fps of the input stream (used by ByteTrack to age
        tentative tracks).
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        minimum_matching_threshold: float = 0.8,
        frame_rate: float = 15.0,
    ) -> None:
        try:
            import supervision as sv  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "supervision not installed; pip install supervision"
            ) from exc

        self._sv = sv
        # supervision ByteTrack API has changed across versions; try
        # the modern kwargs first, then fall back.
        try:
            self._bt = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=int(frame_rate),
            )
        except TypeError:
            self._bt = sv.ByteTrack()  # type: ignore[call-arg]

        # Locked IDs (None until we see at least two stable tracks).
        self.a_id: int | None = None
        self.b_id: int | None = None

    # ------------------------------------------------------------------
    def assign(
        self,
        frame_idx: int,
        frame_shape: tuple[int, int, int],
        people: "FramePeople",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (kpA, kpB) -- each (17, 3) -- for this frame.

        Missing person => zero-keypoints with conf=0.
        """
        if len(people.boxes) == 0:
            return (
                np.zeros((17, 3), dtype=np.float32),
                np.zeros((17, 3), dtype=np.float32),
            )

        sv = self._sv
        det = sv.Detections(
            xyxy=people.boxes,
            confidence=people.scores,
            class_id=np.zeros(len(people.boxes), dtype=int),
        )
        tracked = self._bt.update_with_detections(det)
        if tracked.tracker_id is None or len(tracked) == 0:
            # tracker hasn't confirmed anything yet -- fall back to
            # left/right by box midpoint
            return self._fallback_lr(people)

        ids = np.asarray(tracked.tracker_id)
        # match tracked boxes back to ``people`` rows by IoU so we get
        # the right keypoint indices
        kp_idx = _match_by_iou(tracked.xyxy, people.boxes)
        # keep only the top-2 by confidence among tracked
        if len(ids) > 2:
            keep = np.argsort(-tracked.confidence)[:2]
            ids = ids[keep]
            tracked_xyxy = tracked.xyxy[keep]
            kp_idx = kp_idx[keep]
        else:
            tracked_xyxy = tracked.xyxy

        # Lock A/B by left/right on first frame with >=2 tracks
        if self.a_id is None or self.b_id is None:
            if len(ids) >= 2:
                midx = (tracked_xyxy[:, 0] + tracked_xyxy[:, 2]) / 2
                left = int(np.argmin(midx))
                right = int(np.argmax(midx))
                self.a_id = int(ids[left])
                self.b_id = int(ids[right])
                logger.info(
                    "Locked rikishi IDs at frame %d: A=%d (left/east), B=%d (right/west)",
                    frame_idx,
                    self.a_id,
                    self.b_id,
                )

        # Pull kp for locked IDs
        a_kp = np.zeros((17, 3), dtype=np.float32)
        b_kp = np.zeros((17, 3), dtype=np.float32)
        for k, tid in enumerate(ids):
            kp_row = kp_idx[k]
            if kp_row < 0:
                continue
            if tid == self.a_id:
                a_kp = people.keypoints[kp_row]
            elif tid == self.b_id:
                b_kp = people.keypoints[kp_row]
        return a_kp, b_kp

    # ------------------------------------------------------------------
    def _fallback_lr(
        self, people: "FramePeople"
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(people.boxes) == 0:
            return (
                np.zeros((17, 3), dtype=np.float32),
                np.zeros((17, 3), dtype=np.float32),
            )
        if len(people.boxes) == 1:
            return people.keypoints[0], np.zeros((17, 3), dtype=np.float32)
        midx = (people.boxes[:, 0] + people.boxes[:, 2]) / 2
        order = np.argsort(midx)
        return people.keypoints[order[0]], people.keypoints[order[-1]]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _match_by_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """For each box in ``boxes_a``, return the index of the best-IoU box
    in ``boxes_b`` (or -1 if none with IoU > 0)."""
    out = np.full(len(boxes_a), -1, dtype=int)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return out
    for i, a in enumerate(boxes_a):
        x1 = np.maximum(a[0], boxes_b[:, 0])
        y1 = np.maximum(a[1], boxes_b[:, 1])
        x2 = np.minimum(a[2], boxes_b[:, 2])
        y2 = np.minimum(a[3], boxes_b[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a + area_b - inter + 1e-9
        iou = inter / union
        best = int(np.argmax(iou))
        if iou[best] > 0:
            out[i] = best
    return out
