"""Unit tests for src.features.chronos_encoder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.chronos_encoder import (
    ChronosEncoder,
    FeatureEnricher,
    RikishiSequenceBuilder,
)


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
def _toy_bouts() -> pd.DataFrame:
    """Hand-built bouts table with 3 rikishi over 3 basho.

    Rikishi 1 — appears every basho (in 1 bout/day for a couple days).
    Rikishi 2 — appears in 202301 and 202303.
    Rikishi 3 — appears only in 202303 (new rikishi for 202303 onwards).
    """
    rows = [
        # basho 202301, day 1
        ("202301", 1, 1, 1, 2, 1),  # 1 beats 2
        ("202301", 1, 2, 2, 1, 2),  # 2 beats 1
        ("202301", 2, 1, 1, 2, 1),  # 1 beats 2
        # basho 202302, day 1 — only rikishi 1 vs a new opponent 4
        ("202302", 1, 1, 1, 4, 4),  # 4 beats 1
        ("202302", 1, 2, 4, 1, 1),  # 1 beats 4
        # basho 202303
        ("202303", 1, 1, 1, 3, 1),
        ("202303", 1, 2, 2, 3, 2),
    ]
    return pd.DataFrame(
        rows,
        columns=["bashoId", "day", "matchNo", "eastId", "westId", "winnerId"],
    )


@pytest.fixture
def toy_bouts() -> pd.DataFrame:
    return _toy_bouts()


@pytest.fixture
def toy_features() -> pd.DataFrame:
    # A handful of rows that re-use the toy rikishi ids, ordered like the
    # real features.parquet (one row per bout) but only need bashoId, day,
    # matchNo, eastId, westId, y.  We replicate to >=8 rows so PCA(n=4) has
    # enough samples to fit.
    rows = [
        ("202302", 1, 1, 1, 4, 0),
        ("202302", 1, 2, 4, 1, 1),
        ("202303", 1, 1, 1, 3, 1),
        ("202303", 1, 2, 2, 3, 1),
        ("202303", 1, 3, 3, 1, 0),
        ("202303", 1, 4, 3, 2, 0),
        ("202303", 2, 1, 1, 2, 1),
        ("202303", 2, 2, 2, 1, 0),
    ]
    return pd.DataFrame(
        rows,
        columns=["bashoId", "day", "matchNo", "eastId", "westId", "y"],
    )


# ---------------------------------------------------------------------- #
# 1. Sequence builder — no leakage
# ---------------------------------------------------------------------- #
def test_sequence_builder_strictly_prior(toy_bouts: pd.DataFrame) -> None:
    builder = RikishiSequenceBuilder(toy_bouts, max_history=10)

    # Rikishi 1 in basho 202301: NO history (first basho they appear in)
    assert builder.history("202301", 1).size == 0

    # Rikishi 1 in basho 202302: should see only 202301 bouts.
    # In 202301 rikishi 1 went W, L, W (matches 1-day-1 win, 1-day-2 loss, 2-day-1 win).
    h = builder.history("202302", 1)
    assert list(h) == [1.0, -1.0, 1.0]

    # Rikishi 1 in basho 202303: 202301 (3 bouts) + 202302 (L, W)
    h = builder.history("202303", 1)
    assert list(h) == [1.0, -1.0, 1.0, -1.0, 1.0]

    # Rikishi 3 in basho 202303: never appeared before — empty
    assert builder.history("202303", 3).size == 0

    # Unknown rikishi id — empty
    assert builder.history("202303", 9999).size == 0


def test_sequence_builder_truncates(toy_bouts: pd.DataFrame) -> None:
    builder = RikishiSequenceBuilder(toy_bouts, max_history=2)
    h = builder.history("202303", 1)
    # 5 historical bouts, capped to 2 -> last two
    assert len(h) == 2
    assert list(h) == [-1.0, 1.0]


# ---------------------------------------------------------------------- #
# 2. ChronosEncoder.embed shape — slow, needs HF download
# ---------------------------------------------------------------------- #
@pytest.mark.slow
def test_chronos_encoder_shape() -> None:
    enc = ChronosEncoder(model_name="amazon/chronos-bolt-small", device="cpu")
    seqs = [
        np.array([1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float32),
        np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        np.zeros(0, dtype=np.float32),  # empty -> neutral fill
    ]
    emb = enc.embed(seqs, batch_size=4)
    assert emb.shape == (3, enc.embed_dim)
    assert emb.dtype == np.float32
    # Embeddings for differing sequences must differ
    assert not np.allclose(emb[0], emb[1])


# ---------------------------------------------------------------------- #
# 3. End-to-end enricher: shape preservation + 2 * pca_dim new cols
# ---------------------------------------------------------------------- #
class _StubEncoder:
    """Deterministic stand-in for ChronosEncoder that doesn't need HF."""

    embed_dim = 32

    def embed(self, sequences, batch_size: int = 64) -> np.ndarray:
        rng = np.random.default_rng(0)
        out = np.zeros((len(sequences), self.embed_dim), dtype=np.float32)
        for i, s in enumerate(sequences):
            # encode length + last value + a deterministic noise vector
            arr = np.asarray(s, dtype=np.float32)
            seed = int(arr.size * 17 + (arr[-1] if arr.size else 0) * 31)
            r = np.random.default_rng(seed).standard_normal(self.embed_dim)
            out[i] = r.astype(np.float32)
        # avoid unused-rng warning
        _ = rng
        return out


def test_enricher_preserves_rows_and_adds_columns(
    toy_bouts: pd.DataFrame, toy_features: pd.DataFrame, tmp_path
) -> None:
    builder = RikishiSequenceBuilder(toy_bouts, max_history=20)
    encoder = _StubEncoder()
    enricher = FeatureEnricher(
        encoder=encoder,  # type: ignore[arg-type]
        builder=builder,
        pca_dim=4,  # small so it fits with only 3 rows
        cache_dir=tmp_path / "cache",
    )
    enriched, stats = enricher.enrich(toy_features)

    # Row count preserved
    assert len(enriched) == len(toy_features)
    # 2 * pca_dim new columns
    new_cols = [c for c in enriched.columns if c.startswith("chronos_")]
    assert len(new_cols) == 2 * 4
    assert set(new_cols) == {f"chronos_A_{i}" for i in range(4)} | {
        f"chronos_B_{i}" for i in range(4)
    }
    # All original columns still present
    for c in toy_features.columns:
        assert c in enriched.columns
    # Stats sanity
    assert stats.n_rows == len(toy_features)
    assert stats.n_new_cols == 8
    assert stats.max_len >= 1
    # PCA artefacts persisted
    assert (tmp_path / "cache" / "chronos_pca_A.joblib").exists()
    assert (tmp_path / "cache" / "chronos_pca_B.joblib").exists()
