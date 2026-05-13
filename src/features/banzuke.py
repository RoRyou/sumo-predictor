"""Prior-basho banzuke features (kachikoshi / makekoshi state).

Adds per-bout features derived from each rikishi's *previous makuuchi
basho* final record:

    prev_wins_A, prev_losses_A, prev_absences_A
    prev_winrate_A      = wins / max(wins+losses, 1)
    prev_kachikoshi_A   = 1 if wins >= 8 else 0
    prev_makekoshi_A    = 1 if losses >= 8 else 0
    prev_basho_gap_A    = number of basho since rikishi's last makuuchi
                          appearance (0 if previous basho; 99 if never)
    + same for B, plus _diff variants

The 'previous makuuchi basho' is well-defined because we only model
makuuchi bouts; rikishi who drop to juryo simply get gap >= 1 for the
returning basho.  No leakage: the join uses *strictly prior* bashoId.

CLI::

    python -m src.features.banzuke enrich \\
        --features data/processed/features.parquet \\
        --banzuke data/raw/banzuke.parquet \\
        --out data/processed/features_v2.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_prev_basho_lookup(banzuke: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with one row per (rikishiId, bashoId).

    For each row: the rikishi's *previous makuuchi appearance* — values
    ``prev_wins, prev_losses, prev_absences, prev_basho_gap`` and the
    derived ``prev_winrate, prev_kachikoshi, prev_makekoshi``.

    ``prev_basho_gap`` counts basho slots (Jan/Mar/May/Jul/Sep/Nov per year)
    between the current basho and the prior makuuchi appearance.  Missing
    -> 99 with a 0/0 record (treated as "no info" by downstream models).
    """
    bz = banzuke[banzuke["division"] == "Makuuchi"].copy()
    bz["bashoId"] = bz["bashoId"].astype(str)
    bz = bz.sort_values(["rikishiId", "bashoId"]).reset_index(drop=True)

    bz["_basho_idx"] = _basho_to_idx(bz["bashoId"])

    # Within each rikishi group, shift to get "previous makuuchi appearance"
    g = bz.groupby("rikishiId", group_keys=False)
    bz["prev_wins"] = g["wins"].shift(1)
    bz["prev_losses"] = g["losses"].shift(1)
    bz["prev_absences"] = g["absences"].shift(1)
    bz["_prev_basho_idx"] = g["_basho_idx"].shift(1)
    bz["prev_basho_gap"] = (bz["_basho_idx"] - bz["_prev_basho_idx"]).fillna(99).astype(int)
    bz["prev_basho_gap"] = bz["prev_basho_gap"].clip(upper=99)

    bz["prev_wins"] = bz["prev_wins"].fillna(0)
    bz["prev_losses"] = bz["prev_losses"].fillna(0)
    bz["prev_absences"] = bz["prev_absences"].fillna(0)
    denom = (bz["prev_wins"] + bz["prev_losses"]).clip(lower=1)
    bz["prev_winrate"] = bz["prev_wins"] / denom
    bz["prev_kachikoshi"] = (bz["prev_wins"] >= 8).astype(int)
    bz["prev_makekoshi"] = (bz["prev_losses"] >= 8).astype(int)

    return bz[
        [
            "bashoId",
            "rikishiId",
            "prev_wins",
            "prev_losses",
            "prev_absences",
            "prev_basho_gap",
            "prev_winrate",
            "prev_kachikoshi",
            "prev_makekoshi",
        ]
    ]


def _basho_to_idx(s: pd.Series) -> pd.Series:
    """Map YYYYMM string -> integer index (Jan=0, Mar=1, ... Nov=5)."""
    yr = s.str[:4].astype(int)
    mo = s.str[4:].astype(int)
    return yr * 6 + ((mo - 1) // 2)


def enrich(features: pd.DataFrame, banzuke: pd.DataFrame) -> pd.DataFrame:
    """Add prev_* features to a per-bout feature frame.

    ``features`` must have ``bashoId, eastId, westId`` columns.
    """
    features = features.copy()
    features["bashoId"] = features["bashoId"].astype(str)
    lookup = build_prev_basho_lookup(banzuke)
    lookup["rikishiId"] = lookup["rikishiId"].astype(int)

    feat_cols = [
        "prev_wins",
        "prev_losses",
        "prev_absences",
        "prev_basho_gap",
        "prev_winrate",
        "prev_kachikoshi",
        "prev_makekoshi",
    ]

    for side, side_id in (("A", "eastId"), ("B", "westId")):
        merged = features.merge(
            lookup.rename(columns={c: f"{c}_{side}" for c in feat_cols}),
            left_on=["bashoId", side_id],
            right_on=["bashoId", "rikishiId"],
            how="left",
        )
        # drop the duplicate rikishiId column
        if "rikishiId" in merged.columns:
            merged = merged.drop(columns=["rikishiId"])
        features = merged

    # Diff features
    features["prev_winrate_diff"] = features["prev_winrate_A"].fillna(0) - features["prev_winrate_B"].fillna(0)
    features["prev_wins_diff"] = features["prev_wins_A"].fillna(0) - features["prev_wins_B"].fillna(0)
    features["prev_kachi_diff"] = features["prev_kachikoshi_A"].fillna(0) - features["prev_kachikoshi_B"].fillna(0)
    # Sentinel-fill remaining NaNs so XGB can use them
    for c in feat_cols:
        for side in ("A", "B"):
            col = f"{c}_{side}"
            if col in features.columns:
                features[col] = features[col].fillna(-1.0 if c == "prev_winrate" else 0.0)
    return features


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _setup_logging(v: int) -> None:
    logging.basicConfig(
        level=logging.INFO if v else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_enrich(args: argparse.Namespace) -> int:
    features = pd.read_parquet(args.features)
    banzuke = pd.read_parquet(args.banzuke)
    logger.info("Loaded features=%d, banzuke=%d", len(features), len(banzuke))
    out_df = enrich(features, banzuke)
    new_cols = [c for c in out_df.columns if c not in features.columns]
    logger.info("Added %d columns: %s", len(new_cols), new_cols)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Saved {args.out}  rows={len(out_df)}  cols={out_df.shape[1]}  added={len(new_cols)}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Add prior-basho banzuke features")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("enrich")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--banzuke", default="data/raw/banzuke.parquet")
    r.add_argument("--out", default="data/processed/features_v2.parquet")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_enrich)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
