"""YOLOv8-pose wrapper that returns ``(T, 2, 17, 3)`` keypoint tensors.

The two ``person`` slots correspond to the two rikishi (A=east/left,
B=west/right) AFTER :mod:`src.features.tracking` has locked ID order.
If detection only finds one (or zero) people in a frame, the missing
slots are zero-filled and their confidence is set to 0.

Frame index 0 maps to ``frames[0]``; rows are ordered chronologically.

Keypoint layout follows the COCO-17 convention used by YOLOv8-pose --
see readme §5.2 table.

Example
-------
>>> extractor = PoseExtractor(model="yolov8n-pose.pt")
>>> kp = extractor.extract(frames)      # (T, 2, 17, 3)
>>> kp.shape
(150, 2, 17, 3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# COCO-17 keypoint indices (matches ultralytics YOLOv8-pose output)
KP_NOSE = 0
KP_LEFT_EYE, KP_RIGHT_EYE = 1, 2
KP_LEFT_EAR, KP_RIGHT_EAR = 3, 4
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_ELBOW, KP_RIGHT_ELBOW = 7, 8
KP_LEFT_WRIST, KP_RIGHT_WRIST = 9, 10
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12
KP_LEFT_KNEE, KP_RIGHT_KNEE = 13, 14
KP_LEFT_ANKLE, KP_RIGHT_ANKLE = 15, 16


@dataclass
class FramePeople:
    """All persons detected in a single frame (raw model output).

    Attributes
    ----------
    boxes : (N, 4) float32 -- xyxy
    scores : (N,) float32 -- box confidence
    keypoints : (N, 17, 3) float32 -- xyc
    """

    boxes: np.ndarray
    scores: np.ndarray
    keypoints: np.ndarray


class PoseExtractor:
    """Ultralytics YOLOv8-pose wrapper.

    Parameters
    ----------
    model
        Path/name of the YOLOv8-pose checkpoint.  Default ``yolov8n-pose.pt``
        is small enough for laptop smoke tests.
    device
        ``"mps"`` on Apple silicon, ``"cuda"`` if available, else ``"cpu"``.
        ``None`` (default) auto-picks.
    conf
        Detection confidence threshold passed to ultralytics.
    iou
        NMS IoU threshold.
    max_det
        Hard cap on detections per frame -- 5 is more than enough for a
        dohyo camera.
    """

    def __init__(
        self,
        model: str = "yolov8n-pose.pt",
        device: str | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
        max_det: int = 5,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics not installed; pip install ultralytics"
            ) from exc

        if device is None:
            device = self._auto_device()

        self.device = device
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self._yolo = YOLO(model)
        logger.info("Loaded %s on device=%s", model, device)

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch  # type: ignore
        except ImportError:
            return "cpu"
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ------------------------------------------------------------------
    # Per-frame raw output (used by tracking module too)
    # ------------------------------------------------------------------
    def detect_frame(self, frame: np.ndarray) -> FramePeople:
        """Run pose on a single RGB frame, return all persons found.

        Returned arrays are sorted by descending box confidence.
        """
        res = self._yolo.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )[0]

        boxes_t = res.boxes
        kpts_t = res.keypoints
        if boxes_t is None or len(boxes_t) == 0:
            return FramePeople(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                keypoints=np.zeros((0, 17, 3), dtype=np.float32),
            )

        boxes = boxes_t.xyxy.cpu().numpy().astype(np.float32)
        scores = boxes_t.conf.cpu().numpy().astype(np.float32)
        if kpts_t is not None and kpts_t.data is not None:
            kpts = kpts_t.data.cpu().numpy().astype(np.float32)
            # ultralytics may emit (N, 17, 2) when conf channel missing
            if kpts.shape[-1] == 2:
                conf_pad = np.ones((*kpts.shape[:-1], 1), dtype=np.float32)
                kpts = np.concatenate([kpts, conf_pad], axis=-1)
        else:
            kpts = np.zeros((boxes.shape[0], 17, 3), dtype=np.float32)

        # sort descending by score
        order = np.argsort(-scores)
        return FramePeople(
            boxes=boxes[order],
            scores=scores[order],
            keypoints=kpts[order],
        )

    # ------------------------------------------------------------------
    # Sequence-level extraction
    # ------------------------------------------------------------------
    def extract(
        self,
        frames: np.ndarray,
        tracker: Any | None = None,
    ) -> np.ndarray:
        """Run pose on a sequence; return ``(T, 2, 17, 3)``.

        When ``tracker`` is provided (an instance of
        :class:`src.features.tracking.TwoRikishiTracker`), the two output
        slots are aligned by tracked ID.  Otherwise we fall back to
        "two highest-confidence detections, left-then-right by x-mid".
        """
        T = len(frames)
        out = np.zeros((T, 2, 17, 3), dtype=np.float32)

        for t, frame in enumerate(frames):
            people = self.detect_frame(frame)
            if tracker is not None:
                a_kp, b_kp = tracker.assign(t, frame.shape, people)
            else:
                a_kp, b_kp = self._fallback_two(people)
            out[t, 0] = a_kp
            out[t, 1] = b_kp
        return out

    @staticmethod
    def _fallback_two(people: FramePeople) -> tuple[np.ndarray, np.ndarray]:
        """Pick top-2 by confidence, order by x-midpoint (left=A, right=B)."""
        if len(people.boxes) == 0:
            return (
                np.zeros((17, 3), dtype=np.float32),
                np.zeros((17, 3), dtype=np.float32),
            )
        if len(people.boxes) == 1:
            return people.keypoints[0], np.zeros((17, 3), dtype=np.float32)

        top2_idx = np.array([0, 1])
        top2 = people.boxes[top2_idx]
        midx = (top2[:, 0] + top2[:, 2]) / 2
        left = top2_idx[int(np.argmin(midx))]
        right = top2_idx[int(np.argmax(midx))]
        return people.keypoints[left], people.keypoints[right]


# ----------------------------------------------------------------------
# Visualisation helper (used by smoke test)
# ----------------------------------------------------------------------
COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


def draw_pose_overlay(
    frame: np.ndarray,
    kp_two: np.ndarray,
    out_path: str | Path,
    conf_thresh: float = 0.2,
) -> Path:
    """Save ``frame`` with a 2-skeleton overlay drawn on top.

    Parameters
    ----------
    frame
        (H, W, 3) RGB uint8 image.
    kp_two
        (2, 17, 3) keypoints xyc.  Person A drawn in red, B in blue.
    out_path
        PNG output path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(frame)
    colors = ("red", "blue")

    for p, kps in enumerate(kp_two):
        c = colors[p]
        for a, b in COCO_SKELETON:
            if kps[a, 2] < conf_thresh or kps[b, 2] < conf_thresh:
                continue
            ax.plot([kps[a, 0], kps[b, 0]], [kps[a, 1], kps[b, 1]], c=c, lw=2)
        vis = kps[kps[:, 2] >= conf_thresh]
        if len(vis):
            ax.scatter(vis[:, 0], vis[:, 1], c=c, s=20, alpha=0.8)

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path
