"""Derived kinematic features from raw 17-keypoint pose sequences.

Input
-----
``kp_seq`` : ``np.ndarray`` shape ``(T, 2, 17, 3)``, ``xyc`` per joint
where person 0 = A (east/left) and 1 = B (west/right).

Output
------
A ``(T, F)`` ``np.ndarray`` of per-frame derived features (F≈40) plus a
parallel list of feature names.  See ``FEATURE_NAMES`` for the exact
ordering.

Categories (readme §5.3):

* Single-person static (per A and B): CoM xy, CoM height (normalised by
  torso length), forward-lean angle, mean knee angle, stride width,
  torso length, mean keypoint confidence.
* Single-person dynamic: CoM velocity (vx, vy, speed), acceleration
  magnitude, CoM stability (rolling std over window=N), lean rate,
  mean wrist speed.
* Pair interaction: pair-CoM distance, relative lean (θA−θB), push-
  direction projection, contact-distance estimate (nearest wrist→body),
  advantage side (sign(CoM_A_x − CoM_B_x)).

All distance-like outputs are normalised by the (per-frame, per-person
or pair-averaged) torso length so the result is scale-invariant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.features.pose import (
    KP_LEFT_ANKLE,
    KP_LEFT_HIP,
    KP_LEFT_KNEE,
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_RIGHT_ANKLE,
    KP_RIGHT_HIP,
    KP_RIGHT_KNEE,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Feature names (ordered) -- used as parquet columns
# ----------------------------------------------------------------------
def _person_feature_names(prefix: str) -> list[str]:
    return [
        f"{prefix}_com_x_n",
        f"{prefix}_com_y_n",
        f"{prefix}_com_height_n",
        f"{prefix}_lean_angle",
        f"{prefix}_knee_angle_mean",
        f"{prefix}_stride_width_n",
        f"{prefix}_torso_len",
        f"{prefix}_kp_conf_mean",
        f"{prefix}_com_vx",
        f"{prefix}_com_vy",
        f"{prefix}_com_speed",
        f"{prefix}_com_accel_mag",
        f"{prefix}_com_stability",
        f"{prefix}_lean_rate",
        f"{prefix}_wrist_speed",
    ]


PAIR_FEATURE_NAMES = [
    "pair_com_dist_n",
    "pair_lean_diff",
    "push_dir_proj",
    "contact_dist_n",
    "advantage_side",
    "pair_torso_mean",
]


FEATURE_NAMES: list[str] = (
    _person_feature_names("A")
    + _person_feature_names("B")
    + PAIR_FEATURE_NAMES
)
# 15 per-person * 2 + 6 pair = 36 features.  Pad to 40 with reserved
# slots so the model spec (40) stays satisfied.
RESERVED_FEATURE_NAMES = [f"reserved_{i}" for i in range(4)]
FEATURE_NAMES += RESERVED_FEATURE_NAMES

FEATURE_DIM = len(FEATURE_NAMES)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class KinematicsConfig:
    """Tunable knobs for feature extraction."""

    fps: float = 15.0
    stability_window: int = 9         # ≈0.6s at 15fps
    conf_threshold: float = 0.2       # below this, treat joint as missing
    smoothing_kernel: int = 3         # 1=disable, odd values only
    eps: float = 1e-6


# ----------------------------------------------------------------------
# Low-level geometry helpers
# ----------------------------------------------------------------------
def _joint_xy(kp: np.ndarray, idx: int, conf_thresh: float) -> np.ndarray:
    """Return (T,2) xy with NaN where confidence < threshold."""
    xy = kp[:, idx, :2].astype(np.float32).copy()
    c = kp[:, idx, 2]
    xy[c < conf_thresh] = np.nan
    return xy


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _safe_norm(v: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    return np.sqrt(np.sum(v * v, axis=axis) + eps)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at vertex ``b`` formed by segments b->a and b->c (radians)."""
    v1 = a - b
    v2 = c - b
    n1 = _safe_norm(v1)
    n2 = _safe_norm(v2)
    cos = np.sum(v1 * v2, axis=-1) / (n1 * n2)
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


def _diff_rate(x: np.ndarray, fps: float) -> np.ndarray:
    """Central-difference time derivative along axis 0; pads endpoints."""
    out = np.zeros_like(x)
    out[1:-1] = (x[2:] - x[:-2]) / 2.0 * fps
    out[0] = (x[1] - x[0]) * fps
    out[-1] = (x[-1] - x[-2]) * fps
    return out


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """1-D rolling std along axis 0 with reflect padding (handles NaN)."""
    if window <= 1:
        return np.zeros_like(x)
    half = window // 2
    T = x.shape[0]
    pad = np.pad(x, ((half, half),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    out = np.zeros_like(x)
    for i in range(T):
        chunk = pad[i : i + window]
        out[i] = np.nanstd(chunk, axis=0)
    return out


def _fillna_then_smooth(x: np.ndarray, kernel: int) -> np.ndarray:
    """Linear-interpolate NaNs along time axis, then box-smooth."""
    x = x.copy()
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False
    T, D = x.shape
    for d in range(D):
        col = x[:, d]
        mask = np.isnan(col)
        if mask.all():
            col[:] = 0.0
        elif mask.any():
            idx = np.arange(T)
            col[mask] = np.interp(idx[mask], idx[~mask], col[~mask])
        x[:, d] = col
    if kernel > 1:
        k = np.ones(kernel, dtype=np.float32) / kernel
        for d in range(D):
            x[:, d] = np.convolve(x[:, d], k, mode="same")
    return x[:, 0] if squeeze else x


# ----------------------------------------------------------------------
# Per-person feature block
# ----------------------------------------------------------------------
def _person_features(
    kp: np.ndarray, cfg: KinematicsConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame features for one rikishi.

    Parameters
    ----------
    kp : (T, 17, 3)

    Returns
    -------
    feats : (T, 15)
    com_norm : (T, 2)
        Torso-length-normalised CoM coordinates (used downstream by
        pair features).
    torso : (T,)
        Torso length per frame (px), already smoothed.
    """
    T = kp.shape[0]
    conf = cfg.conf_threshold

    l_sh = _joint_xy(kp, KP_LEFT_SHOULDER, conf)
    r_sh = _joint_xy(kp, KP_RIGHT_SHOULDER, conf)
    l_hip = _joint_xy(kp, KP_LEFT_HIP, conf)
    r_hip = _joint_xy(kp, KP_RIGHT_HIP, conf)
    l_kn = _joint_xy(kp, KP_LEFT_KNEE, conf)
    r_kn = _joint_xy(kp, KP_RIGHT_KNEE, conf)
    l_an = _joint_xy(kp, KP_LEFT_ANKLE, conf)
    r_an = _joint_xy(kp, KP_RIGHT_ANKLE, conf)
    l_wr = _joint_xy(kp, KP_LEFT_WRIST, conf)
    r_wr = _joint_xy(kp, KP_RIGHT_WRIST, conf)

    sh_mid = _midpoint(l_sh, r_sh)
    hip_mid = _midpoint(l_hip, r_hip)
    wr_mid = _midpoint(l_wr, r_wr)

    # ------- torso length & CoM (image px) --------
    torso_raw = _safe_norm(sh_mid - hip_mid, eps=cfg.eps)
    torso = _fillna_then_smooth(torso_raw, cfg.smoothing_kernel)
    torso = np.maximum(torso, 1.0)  # avoid divide-by-zero

    com = hip_mid  # (T, 2) -- midpoint of the two hips
    com = _fillna_then_smooth(com, cfg.smoothing_kernel)
    com_x_n = com[:, 0] / torso
    com_y_n = com[:, 1] / torso

    # ------- height normalised (smaller hip_y → higher in image → less stable)
    com_height_n = com[:, 1] / torso

    # ------- forward-lean angle (radians, signed: positive = forward) --------
    dxy = sh_mid - hip_mid  # vector hip→shoulder
    # arctan2(dx, -dy) -> straight upright is 0, leaning forward is positive
    lean_angle_raw = np.arctan2(dxy[:, 0], -dxy[:, 1])
    lean_angle = _fillna_then_smooth(lean_angle_raw, cfg.smoothing_kernel)

    # ------- knee angle (mean of L & R) --------
    kn_l = _angle(l_hip, l_kn, l_an)
    kn_r = _angle(r_hip, r_kn, r_an)
    knee = np.nanmean(np.stack([kn_l, kn_r], axis=0), axis=0)
    knee = _fillna_then_smooth(knee, cfg.smoothing_kernel)

    # ------- stride width (|x_left_ankle - x_right_ankle| / torso) --------
    stride_raw = np.abs(l_an[:, 0] - r_an[:, 0])
    stride = _fillna_then_smooth(stride_raw, cfg.smoothing_kernel) / torso

    # ------- mean keypoint confidence --------
    kp_conf_mean = kp[:, :, 2].mean(axis=1)

    # ------- dynamic features (operate on smoothed CoM) --------
    com_v = _diff_rate(com, cfg.fps) / torso[:, None]  # normalised speed
    com_vx, com_vy = com_v[:, 0], com_v[:, 1]
    com_speed = _safe_norm(com_v, eps=cfg.eps)
    com_a = _diff_rate(com_v, cfg.fps)
    com_accel_mag = _safe_norm(com_a, eps=cfg.eps)
    com_stability = _rolling_std(com, cfg.stability_window)[:, 0] / torso

    lean_rate = _diff_rate(lean_angle[:, None], cfg.fps)[:, 0]

    # ------- wrist speed (mean of L & R, normalised) --------
    wr_mid_fill = _fillna_then_smooth(wr_mid, cfg.smoothing_kernel)
    wr_v = _diff_rate(wr_mid_fill, cfg.fps) / torso[:, None]
    wrist_speed = _safe_norm(wr_v, eps=cfg.eps)

    feats = np.stack(
        [
            com_x_n,
            com_y_n,
            com_height_n,
            lean_angle,
            knee,
            stride,
            torso,
            kp_conf_mean,
            com_vx,
            com_vy,
            com_speed,
            com_accel_mag,
            com_stability,
            lean_rate,
            wrist_speed,
        ],
        axis=1,
    )
    com_norm = com / torso[:, None]  # (T, 2)
    return feats.astype(np.float32), com_norm.astype(np.float32), torso.astype(np.float32)


# ----------------------------------------------------------------------
# Pair-interaction block
# ----------------------------------------------------------------------
def _pair_features(
    kp_a: np.ndarray,
    kp_b: np.ndarray,
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    torso_a: np.ndarray,
    torso_b: np.ndarray,
    cfg: KinematicsConfig,
) -> np.ndarray:
    """Pair-level interaction features (T, 6)."""
    conf = cfg.conf_threshold
    pair_torso = (torso_a + torso_b) / 2.0
    pair_torso = np.maximum(pair_torso, 1.0)

    com_a = (_joint_xy(kp_a, KP_LEFT_HIP, conf) + _joint_xy(kp_a, KP_RIGHT_HIP, conf)) / 2
    com_b = (_joint_xy(kp_b, KP_LEFT_HIP, conf) + _joint_xy(kp_b, KP_RIGHT_HIP, conf)) / 2
    com_a = _fillna_then_smooth(com_a, cfg.smoothing_kernel)
    com_b = _fillna_then_smooth(com_b, cfg.smoothing_kernel)

    pair_dist = _safe_norm(com_a - com_b, eps=cfg.eps) / pair_torso

    # lean_diff = feat col 3 (lean_angle) of A - of B
    lean_diff = feats_a[:, 3] - feats_b[:, 3]

    # push direction: (wrist_A - shoulder_A) · (CoM_B - CoM_A), normalised by both norms
    sh_a = (
        _joint_xy(kp_a, KP_LEFT_SHOULDER, conf)
        + _joint_xy(kp_a, KP_RIGHT_SHOULDER, conf)
    ) / 2
    wr_a = (
        _joint_xy(kp_a, KP_LEFT_WRIST, conf)
        + _joint_xy(kp_a, KP_RIGHT_WRIST, conf)
    ) / 2
    sh_a = _fillna_then_smooth(sh_a, cfg.smoothing_kernel)
    wr_a = _fillna_then_smooth(wr_a, cfg.smoothing_kernel)
    arm = wr_a - sh_a
    target = com_b - com_a
    push_proj = np.sum(arm * target, axis=-1) / (
        _safe_norm(arm, eps=cfg.eps) * _safe_norm(target, eps=cfg.eps)
    )

    # contact distance: min over (wrist_A, all_body_B_keypoints)
    body_idx = [
        KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP,
        KP_LEFT_KNEE, KP_RIGHT_KNEE, KP_LEFT_ANKLE, KP_RIGHT_ANKLE,
    ]
    contact = np.full(kp_a.shape[0], np.nan, dtype=np.float32)
    for t in range(kp_a.shape[0]):
        wrists_a = []
        for w_idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
            if kp_a[t, w_idx, 2] >= conf:
                wrists_a.append(kp_a[t, w_idx, :2])
        body_b = []
        for b_idx in body_idx:
            if kp_b[t, b_idx, 2] >= conf:
                body_b.append(kp_b[t, b_idx, :2])
        if not wrists_a or not body_b:
            continue
        wa = np.asarray(wrists_a)            # (Wa, 2)
        bb = np.asarray(body_b)              # (Bb, 2)
        d = np.linalg.norm(
            wa[:, None, :] - bb[None, :, :], axis=-1
        )
        contact[t] = d.min()
    contact = _fillna_then_smooth(contact, cfg.smoothing_kernel) / pair_torso

    advantage = np.sign(com_a[:, 0] - com_b[:, 0]).astype(np.float32)

    out = np.stack(
        [
            pair_dist,
            lean_diff,
            push_proj,
            contact,
            advantage,
            pair_torso,
        ],
        axis=1,
    )
    return out.astype(np.float32)


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------
def compute_features(
    kp_seq: np.ndarray,
    cfg: KinematicsConfig | None = None,
) -> np.ndarray:
    """Compute the full (T, F) feature matrix from a (T, 2, 17, 3) pose tensor.

    NaNs are replaced with 0 before returning.
    """
    cfg = cfg or KinematicsConfig()
    if kp_seq.ndim != 4 or kp_seq.shape[1] != 2 or kp_seq.shape[2] != 17:
        raise ValueError(
            f"kp_seq must be (T,2,17,3); got {kp_seq.shape}"
        )

    feats_a, _, torso_a = _person_features(kp_seq[:, 0], cfg)
    feats_b, _, torso_b = _person_features(kp_seq[:, 1], cfg)
    pair = _pair_features(
        kp_seq[:, 0], kp_seq[:, 1],
        feats_a, feats_b, torso_a, torso_b,
        cfg,
    )

    T = kp_seq.shape[0]
    reserved = np.zeros((T, len(RESERVED_FEATURE_NAMES)), dtype=np.float32)

    feats = np.concatenate([feats_a, feats_b, pair, reserved], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    assert feats.shape[1] == FEATURE_DIM, (feats.shape, FEATURE_DIM)
    return feats.astype(np.float32)


def features_to_dataframe(feats: np.ndarray):
    """Convenience -- wrap feature matrix in a pandas.DataFrame."""
    import pandas as pd

    if feats.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected {FEATURE_DIM} columns, got {feats.shape[1]}")
    return pd.DataFrame(feats, columns=FEATURE_NAMES)
