"""Pure-Python tests for the Phase-2 (Route B) pose pipeline.

These tests do NOT touch a real video or download model weights; they
exercise the kinematics + model code paths only.  Heavy integration is
left to a smoke driver script.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.features.kinematics import (
    FEATURE_DIM,
    FEATURE_NAMES,
    KinematicsConfig,
    compute_features,
)
from src.models.temporal import AttentionPooling, PoseTower, PoseTowerClassifier


# ---------------------------------------------------------------------- #
# Test fixtures
# ---------------------------------------------------------------------- #
def _fake_kp_seq(T: int = 30, seed: int = 0) -> np.ndarray:
    """Return a plausible (T,2,17,3) keypoint sequence with high conf."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(100, 500, size=(T, 2, 17, 2)).astype(np.float32)
    conf = rng.uniform(0.5, 1.0, size=(T, 2, 17, 1)).astype(np.float32)
    return np.concatenate([xy, conf], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------- #
# Kinematics
# ---------------------------------------------------------------------- #
def test_feature_names_length_matches_dim():
    assert len(FEATURE_NAMES) == FEATURE_DIM == 40


def test_compute_features_shape_and_no_nans():
    kp = _fake_kp_seq(T=50)
    feats = compute_features(kp)
    assert feats.shape == (50, FEATURE_DIM)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()
    assert feats.dtype == np.float32


def test_compute_features_handles_low_confidence():
    """Joints with confidence below threshold should be interpolated, not crash."""
    kp = _fake_kp_seq(T=30)
    # Knock confidence on half the frames down for both persons
    kp[5:15, :, :, 2] = 0.05
    cfg = KinematicsConfig(conf_threshold=0.2)
    feats = compute_features(kp, cfg=cfg)
    assert feats.shape == (30, FEATURE_DIM)
    assert not np.isnan(feats).any()


def test_compute_features_rejects_wrong_shape():
    with pytest.raises(ValueError):
        compute_features(np.zeros((10, 1, 17, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        compute_features(np.zeros((10, 2, 13, 3), dtype=np.float32))


# ---------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------- #
def test_attention_pooling_mask_shape():
    pool = AttentionPooling(dim=16)
    h = torch.randn(3, 10, 16)
    mask = torch.ones(3, 10, dtype=torch.bool)
    mask[:, 7:] = False
    out = pool(h, mask=mask)
    assert out.shape == (3, 16)


def test_posetower_output_shape():
    tower = PoseTower(feat_dim=FEATURE_DIM, hidden=64, embed_dim=128)
    x = torch.randn(2, 50, FEATURE_DIM)
    out = tower(x)
    assert out.shape == (2, 128)


def test_classifier_forward_backward():
    """Loss should decrease after a few gradient steps on synthetic data."""
    torch.manual_seed(0)
    model = PoseTowerClassifier(feat_dim=FEATURE_DIM, hidden=32, embed_dim=64)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    x = torch.randn(8, 30, FEATURE_DIM)
    y = (torch.rand(8) > 0.5).float()
    crit = torch.nn.BCEWithLogitsLoss()

    initial = float(crit(model(x), y).item())
    for _ in range(20):
        loss = crit(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = float(crit(model(x), y).item())
    assert final < initial, f"loss did not decrease: {initial:.4f} -> {final:.4f}"


# ---------------------------------------------------------------------- #
# End-to-end (kinematics -> model)
# ---------------------------------------------------------------------- #
def test_kp_to_model_pipeline():
    kp = _fake_kp_seq(T=64)
    feats = compute_features(kp)
    model = PoseTower(feat_dim=FEATURE_DIM, hidden=32, embed_dim=128)
    x = torch.from_numpy(feats).unsqueeze(0)
    out = model(x)
    assert out.shape == (1, 128)
