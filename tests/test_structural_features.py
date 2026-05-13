"""Unit tests for new structural feature helpers (T7 and T15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.structural import (
    BASHO_DAYS,
    KimariteMatchupTable,
    add_stage_features,
    symmetric_augment,
)


# --------------------------------------------------------------------- #
# T15 — stage features
# --------------------------------------------------------------------- #
def test_add_stage_features_columns_present():
    df = pd.DataFrame({"day": [1, 5, 6, 10, 11, 15]})
    out = add_stage_features(df)
    for c in ("stage_early", "stage_mid", "stage_late", "day_norm", "senshuraku"):
        assert c in out.columns


def test_add_stage_features_buckets_correct():
    df = pd.DataFrame({"day": [1, 5, 6, 10, 11, 15]})
    out = add_stage_features(df)
    assert out["stage_early"].tolist() == [1, 1, 0, 0, 0, 0]
    assert out["stage_mid"].tolist() == [0, 0, 1, 1, 0, 0]
    assert out["stage_late"].tolist() == [0, 0, 0, 0, 1, 1]
    assert out["senshuraku"].tolist() == [0, 0, 0, 0, 0, 1]
    assert out["day_norm"].iloc[-1] == pytest.approx(15 / BASHO_DAYS)


def test_add_stage_features_uses_day_of_basho_if_present():
    df = pd.DataFrame({"day_of_basho": [1, 15], "day": [99, 99]})
    out = add_stage_features(df)
    # prefers day_of_basho — so senshuraku at row 1 must be 1
    assert out["senshuraku"].tolist() == [0, 1]


# --------------------------------------------------------------------- #
# T7 — KimariteMatchupTable
# --------------------------------------------------------------------- #
def test_kimarite_matchup_table_fit_transform_basic():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "pushing_ratio_A": rng.uniform(0, 1, size=n),
        "belt_ratio_A": rng.uniform(0, 1, size=n),
        "pushing_ratio_B": rng.uniform(0, 1, size=n),
        "belt_ratio_B": rng.uniform(0, 1, size=n),
        "y": rng.integers(0, 2, size=n),
    })
    km = KimariteMatchupTable(smoothing=5.0).fit(df)
    assert km.global_mean_ == pytest.approx(df["y"].mean(), abs=1e-9)
    assert len(km.lookup_) >= 1
    out = km.transform(df)
    assert "kimarite_matchup_wr" in out.columns
    # winrate values must be in [0, 1]
    assert (out["kimarite_matchup_wr"] >= 0).all()
    assert (out["kimarite_matchup_wr"] <= 1).all()


def test_kimarite_matchup_table_unseen_pair_falls_back_to_global_mean():
    train = pd.DataFrame({
        "pushing_ratio_A": [0.9, 0.9],
        "belt_ratio_A": [0.0, 0.0],
        "pushing_ratio_B": [0.9, 0.9],
        "belt_ratio_B": [0.0, 0.0],
        "y": [1, 0],
    })
    km = KimariteMatchupTable(smoothing=100.0).fit(train)
    # unseen test pair (belt vs pushing)
    test = pd.DataFrame({
        "pushing_ratio_A": [0.0],
        "belt_ratio_A": [0.9],
        "pushing_ratio_B": [0.9],
        "belt_ratio_B": [0.0],
    })
    out = km.transform(test)
    # should equal global mean (0.5 of [1, 0])
    assert out["kimarite_matchup_wr"].iloc[0] == pytest.approx(0.5)


def test_kimarite_matchup_dominant_classification():
    # explicitly pushing dominant
    assert KimariteMatchupTable._dominant(0.8, 0.1) == "pushing"
    # belt dominant
    assert KimariteMatchupTable._dominant(0.1, 0.7) == "belt"
    # other dominant
    assert KimariteMatchupTable._dominant(0.1, 0.1) == "other"
    # NaN handling
    assert KimariteMatchupTable._dominant(np.nan, np.nan) == "other"


# --------------------------------------------------------------------- #
# Symmetric augmentation — verifies T6 fix
# --------------------------------------------------------------------- #
def test_symmetric_augment_swaps_te_columns():
    df = pd.DataFrame({
        "y": [1, 0],
        "te__heya_A": [0.55, 0.40],
        "te__heya_B": [0.45, 0.60],
        "te__shusshin_A": [0.51, 0.52],
        "te__shusshin_B": [0.49, 0.48],
        "rank_diff": [10.0, -20.0],
    })
    out = symmetric_augment(df)
    assert len(out) == 4
    # mirror rows are the second half
    mirror = out.iloc[2:].reset_index(drop=True)
    assert mirror["te__heya_A"].tolist() == [0.45, 0.60]
    assert mirror["te__heya_B"].tolist() == [0.55, 0.40]
    assert mirror["te__shusshin_A"].tolist() == [0.49, 0.48]
    assert mirror["y"].tolist() == [0, 1]
    # rank_diff sign flipped
    assert mirror["rank_diff"].tolist() == [-10.0, 20.0]
