"""Chronos-Bolt per-rikishi sequence embeddings.

For every bout row in ``features.parquet`` this module produces a dense
embedding of each rikishi's recent bout history using Amazon's
``chronos-bolt-small`` time-series foundation model.

Pipeline
--------
1. :class:`RikishiSequenceBuilder` walks the raw ``bouts.parquet`` and, for
   each (bashoId, rikishiId) pair, returns the last ``max_history`` bout
   outcomes **strictly before** that basho.  Default sequence type is the
   win/loss series encoded as +/-1.
2. :class:`ChronosEncoder` wraps :class:`chronos.ChronosBoltPipeline` and
   exposes ``embed(sequences) -> np.ndarray (N, D)``.  We mean-pool the
   encoder hidden states across patches to collapse the variable
   ``(batch, num_patches+1, d_model)`` output into a fixed-length vector.
3. :class:`FeatureEnricher` runs the encoder for every bout row twice (once
   for east, once for west), fits a PCA to ``--pca-dim`` per side, and
   writes ``features_chronos.parquet`` (original columns + 2 * pca-dim new
   ``chronos_A_*`` / ``chronos_B_*`` columns).

CLI
---
::

    python -m src.features.chronos_encoder enrich \
        --features data/processed/features.parquet \
        --bouts    data/raw/bouts.parquet \
        --out      data/processed/features_chronos.parquet \
        --pca-dim  16 \
        --max-history 50
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# 1. Sequence builder
# ---------------------------------------------------------------------- #
class RikishiSequenceBuilder:
    """Build per-rikishi temporal sequences strictly *prior* to a basho.

    Parameters
    ----------
    bouts_df :
        Raw ``bouts.parquet`` dataframe with at least the columns
        ``bashoId, day, matchNo, eastId, westId, winnerId``.
    max_history :
        Maximum number of trailing bouts to keep per rikishi.  Shorter
        histories are returned as-is; new rikishi with no history get an
        empty array (callers should substitute a neutral sequence).
    """

    def __init__(self, bouts_df: pd.DataFrame, max_history: int = 50) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self.max_history = int(max_history)
        # Normalise types so str comparisons work
        df = bouts_df.copy()
        df["bashoId"] = df["bashoId"].astype(str)
        df = df.sort_values(["bashoId", "day", "matchNo"]).reset_index(drop=True)

        # Long-format: one row per (rikishi appearance).  Outcome is +1 if the
        # rikishi won that bout, -1 otherwise.
        east = df[["bashoId", "day", "matchNo", "eastId", "winnerId"]].rename(
            columns={"eastId": "rikishiId"}
        )
        east["outcome"] = np.where(east["rikishiId"] == east["winnerId"], 1.0, -1.0)
        west = df[["bashoId", "day", "matchNo", "westId", "winnerId"]].rename(
            columns={"westId": "rikishiId"}
        )
        west["outcome"] = np.where(west["rikishiId"] == west["winnerId"], 1.0, -1.0)
        long = pd.concat([east, west], ignore_index=True)
        long = long.sort_values(["rikishiId", "bashoId", "day", "matchNo"]).reset_index(drop=True)
        self._long = long[["rikishiId", "bashoId", "outcome"]]

        # Pre-index per-rikishi outcome arrays + matching bashoId arrays so
        # the per-row lookup is O(log n) instead of a fresh groupby.
        self._by_rikishi: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for rid, grp in self._long.groupby("rikishiId", sort=False):
            self._by_rikishi[int(rid)] = (
                grp["bashoId"].to_numpy(),
                grp["outcome"].to_numpy(dtype=np.float32),
            )

    # ------------------------------------------------------------------ #
    def history(self, basho_id: str, rikishi_id: int) -> np.ndarray:
        """Return last ``max_history`` outcomes strictly before ``basho_id``."""
        basho_id = str(basho_id)
        info = self._by_rikishi.get(int(rikishi_id))
        if info is None:
            return np.zeros(0, dtype=np.float32)
        bashos, outcomes = info
        # bashos is sorted; np.searchsorted gives the cut-off (strictly <)
        cut = int(np.searchsorted(bashos, basho_id, side="left"))
        if cut <= 0:
            return np.zeros(0, dtype=np.float32)
        start = max(0, cut - self.max_history)
        return outcomes[start:cut]

    def __len__(self) -> int:  # pragma: no cover - debug only
        return len(self._by_rikishi)


# ---------------------------------------------------------------------- #
# 2. Chronos encoder
# ---------------------------------------------------------------------- #
class ChronosEncoder:
    """Wrap :class:`chronos.ChronosBoltPipeline` and expose ``embed()``.

    The Chronos-Bolt ``embed`` API returns ``(batch, num_patches+1, d_model)``
    hidden states.  We mean-pool across patches to produce a single
    ``(batch, d_model)`` vector per series, which downstream PCA reduces to
    a manageable dimensionality.

    Parameters
    ----------
    model_name :
        HuggingFace model id.  Default ``amazon/chronos-bolt-small`` (~48M
        params, CPU-friendly).
    device :
        Torch device map.  Default ``cpu``.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-small",
        device: str = "cpu",
    ) -> None:
        import torch
        from chronos import ChronosBoltPipeline

        self.model_name = model_name
        self.device = device
        logger.info("Loading Chronos pipeline %s on %s", model_name, device)
        self.pipeline = ChronosBoltPipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32,
        )
        # Pull d_model from the first encoded sample
        sample_emb, _ = self.pipeline.embed(torch.tensor([0.0, 1.0, -1.0, 0.5]))
        self.embed_dim = int(sample_emb.shape[-1])
        logger.info("Chronos d_model=%d", self.embed_dim)

    # ------------------------------------------------------------------ #
    def _to_tensor_list(self, sequences: Sequence[np.ndarray]):
        import torch

        out = []
        for seq in sequences:
            arr = np.asarray(seq, dtype=np.float32)
            if arr.size == 0:
                # Neutral fill: a single zero entry; Chronos handles short series
                arr = np.zeros(1, dtype=np.float32)
            out.append(torch.from_numpy(arr))
        return out

    def embed(self, sequences: Sequence[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """Embed an iterable of 1-D series. Returns ``(N, d_model)`` float32."""
        import torch
        from tqdm import tqdm

        n = len(sequences)
        out = np.zeros((n, self.embed_dim), dtype=np.float32)
        for start in tqdm(range(0, n, batch_size), desc="chronos.embed", leave=False):
            chunk = sequences[start : start + batch_size]
            tensor_list = self._to_tensor_list(chunk)
            with torch.no_grad():
                emb, _ = self.pipeline.embed(tensor_list)
            # emb: (B, num_patches+1, d_model) -> mean-pool over patches
            pooled = emb.float().mean(dim=1).cpu().numpy()
            out[start : start + len(chunk)] = pooled
        return out


# ---------------------------------------------------------------------- #
# 3. Feature enricher
# ---------------------------------------------------------------------- #
@dataclass
class EnrichmentStats:
    n_rows: int
    n_new_cols: int
    median_len: float
    max_len: int
    empty_sequences: int
    runtime_s: float
    out_path: str


class FeatureEnricher:
    """Top-level orchestrator that joins Chronos embeddings to the feature matrix.

    For every row in ``features.parquet`` we extract the eastId and westId
    history (last ``max_history`` outcomes strictly before that basho), pass
    them through the Chronos encoder, fit a PCA reducer separately for the
    east-history embeddings and the west-history embeddings, and write the
    enriched parquet with ``chronos_A_0..chronos_A_{D-1}`` and
    ``chronos_B_0..chronos_B_{D-1}`` columns appended.
    """

    def __init__(
        self,
        encoder: ChronosEncoder,
        builder: RikishiSequenceBuilder,
        pca_dim: int = 16,
        cache_dir: Path | None = None,
        random_state: int = 42,
    ) -> None:
        self.encoder = encoder
        self.builder = builder
        self.pca_dim = int(pca_dim)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    def _build_sequences(self, df: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
        seqs_a: list[np.ndarray] = []
        seqs_b: list[np.ndarray] = []
        lens: list[int] = []
        empty = 0
        for basho, eid, wid in zip(df["bashoId"].astype(str), df["eastId"], df["westId"]):
            ha = self.builder.history(basho, int(eid))
            hb = self.builder.history(basho, int(wid))
            seqs_a.append(ha)
            seqs_b.append(hb)
            lens.append(len(ha))
            lens.append(len(hb))
            if len(ha) == 0:
                empty += 1
            if len(hb) == 0:
                empty += 1
        lens_arr = np.asarray(lens)
        stats = {
            "median_len": float(np.median(lens_arr)),
            "mean_len": float(lens_arr.mean()),
            "max_len": int(lens_arr.max()),
            "empty_sequences": int(empty),
        }
        return seqs_a, seqs_b, stats

    # ------------------------------------------------------------------ #
    def enrich(self, features: pd.DataFrame, batch_size: int = 64) -> tuple[pd.DataFrame, EnrichmentStats]:
        import joblib
        from sklearn.decomposition import PCA

        t0 = time.time()
        logger.info("Building sequences for %d rows", len(features))
        seqs_a, seqs_b, seq_stats = self._build_sequences(features)
        logger.info("Sequence stats: %s", seq_stats)

        logger.info("Encoding east-history sequences (%d)", len(seqs_a))
        emb_a = self.encoder.embed(seqs_a, batch_size=batch_size)
        logger.info("Encoding west-history sequences (%d)", len(seqs_b))
        emb_b = self.encoder.embed(seqs_b, batch_size=batch_size)

        logger.info(
            "Fitting PCA(n=%d) on east(%s) and west(%s) embeddings",
            self.pca_dim, emb_a.shape, emb_b.shape,
        )
        pca_a = PCA(n_components=self.pca_dim, random_state=self.random_state)
        pca_b = PCA(n_components=self.pca_dim, random_state=self.random_state)
        red_a = pca_a.fit_transform(emb_a).astype(np.float32)
        red_b = pca_b.fit_transform(emb_b).astype(np.float32)

        if self.cache_dir is not None:
            joblib.dump(pca_a, self.cache_dir / "chronos_pca_A.joblib")
            joblib.dump(pca_b, self.cache_dir / "chronos_pca_B.joblib")
            logger.info("Cached PCA models in %s", self.cache_dir)

        cols_a = [f"chronos_A_{i}" for i in range(self.pca_dim)]
        cols_b = [f"chronos_B_{i}" for i in range(self.pca_dim)]
        enriched = features.copy().reset_index(drop=True)
        for i, c in enumerate(cols_a):
            enriched[c] = red_a[:, i]
        for i, c in enumerate(cols_b):
            enriched[c] = red_b[:, i]

        runtime = time.time() - t0
        stats = EnrichmentStats(
            n_rows=len(enriched),
            n_new_cols=2 * self.pca_dim,
            median_len=seq_stats["median_len"],
            max_len=seq_stats["max_len"],
            empty_sequences=seq_stats["empty_sequences"],
            runtime_s=runtime,
            out_path="",
        )
        return enriched, stats


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 1:
        level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_enrich(args: argparse.Namespace) -> int:
    features_path = Path(args.features)
    bouts_path = Path(args.bouts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features %s and bouts %s", features_path, bouts_path)
    feats = pd.read_parquet(features_path)
    bouts = pd.read_parquet(bouts_path)

    builder = RikishiSequenceBuilder(bouts, max_history=args.max_history)
    encoder = ChronosEncoder(model_name=args.model, device=args.device)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_path.parent / "chronos_cache"
    enricher = FeatureEnricher(encoder=encoder, builder=builder, pca_dim=args.pca_dim, cache_dir=cache_dir)

    enriched, stats = enricher.enrich(feats, batch_size=args.batch_size)
    enriched.to_parquet(out_path, index=False)
    stats.out_path = str(out_path)

    logger.info("Wrote %s (rows=%d, +%d cols, runtime=%.1fs)",
                out_path, stats.n_rows, stats.n_new_cols, stats.runtime_s)
    print(
        f"n_rows={stats.n_rows} new_cols={stats.n_new_cols} "
        f"median_seq_len={stats.median_len} max_seq_len={stats.max_len} "
        f"empty_sequences={stats.empty_sequences} runtime_s={stats.runtime_s:.1f} "
        f"out={stats.out_path}"
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chronos-Bolt sequence enrichment")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("enrich", help="Append Chronos embedding columns to features.parquet")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--bouts", default="data/raw/bouts.parquet")
    r.add_argument("--out", default="data/processed/features_chronos.parquet")
    r.add_argument("--pca-dim", type=int, default=16)
    r.add_argument("--max-history", type=int, default=50)
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--model", default="amazon/chronos-bolt-small")
    r.add_argument("--device", default="cpu")
    r.add_argument("--cache-dir", default=None,
                   help="dir to dump fitted PCA models (default: <out_parent>/chronos_cache)")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_enrich)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
