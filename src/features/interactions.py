"""Pairwise/interaction features for Route A.

Trees can learn interactions but only via deep splits.  At ~17–28k rows our
plateau analysis (manual stack 60.36% ≈ AutoGluon 60.30%) suggests the
existing first-order signal is exhausted.  Explicitly synthesising the
*most physically meaningful* interactions short-circuits the depth-vs-data
trade-off and gives the GBDTs surface-aligned splits.

Strategy
--------
Three buckets of interactions, ordered by prior plausibility:

1. **Skill × magnitude** — does the *gap* matter more when both sides are
   confident?
2. **Pressure × form** — late-basho days × current streak / record.
3. **Confidence-weighted h2h** — only trust h2h when count > prior.

The module is leakage-safe: every interaction is a deterministic function
of features already present at bout-eval time (built by
:mod:`src.features.structural` with strict no-leakage).

CLI::

    python -m src.features.interactions enrich \\
        --features data/processed/features.parquet \\
        --out data/processed/features_interact.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with engineered interaction columns appended.

    Safe to call repeatedly: existing interaction columns are overwritten.
    """
    out = df.copy()

    # ------------------------------------------------------------------ #
    # 1. Skill × magnitude
    # ------------------------------------------------------------------ #
    # Rank gap weighted by recent winrate gap: high-rank-diff with skill-
    # divergent rikishi should be near-deterministic; same gap with the
    # weaker side trending up is much closer to 50/50.
    if {"rank_diff", "winrate_diff_90"} <= set(out.columns):
        out["ix_rank_x_winrate90"] = out["rank_diff"] * out["winrate_diff_90"]
    if {"rank_diff", "winrate_diff_30"} <= set(out.columns):
        out["ix_rank_x_winrate30"] = out["rank_diff"] * out["winrate_diff_30"]
    if {"rank_diff", "h2h_winrate"} <= set(out.columns):
        # h2h_winrate is in [0,1]; centre at 0.5 to make sign meaningful
        out["ix_rank_x_h2h"] = out["rank_diff"] * (out["h2h_winrate"] - 0.5)

    # Career-skill diff: prefer over short-window when both sides have history.
    if {"career_winrate_A", "career_winrate_B"} <= set(out.columns):
        out["ix_career_winrate_diff"] = (
            out["career_winrate_A"].fillna(0.5) - out["career_winrate_B"].fillna(0.5)
        )

    # ------------------------------------------------------------------ #
    # 2. Pressure × form (late-basho / kachikoshi pressure)
    # ------------------------------------------------------------------ #
    if {"day_of_basho", "streak_A", "streak_B"} <= set(out.columns):
        d = out["day_of_basho"].astype(float)
        day_norm = (d / 15.0).clip(0.0, 1.0)
        out["ix_day_x_streak_diff"] = day_norm * (out["streak_A"] - out["streak_B"])

    # Win count diff this basho, weighted by how late in the basho it is.
    if {"day_of_basho", "record_w_A", "record_w_B"} <= set(out.columns):
        d = out["day_of_basho"].astype(float)
        day_norm = (d / 15.0).clip(0.0, 1.0)
        out["ix_record_diff_x_day"] = day_norm * (
            out["record_w_A"] - out["record_w_B"]
        )

    # "Kachikoshi pressure": within 1 win of clearing kachikoshi (8 wins).
    # Encoded as: A needs (8 - record_w_A) wins out of (15 - day_of_basho)
    # remaining days; same for B.  Higher pressure → fewer remaining days.
    if {"record_w_A", "record_l_A", "record_w_B", "record_l_B", "day_of_basho"} <= set(
        out.columns
    ):
        remaining = (15 - out["day_of_basho"].astype(float)).clip(lower=0)
        out["ix_kachi_pressure_A"] = (8 - out["record_w_A"]).clip(lower=0) / (remaining + 1)
        out["ix_kachi_pressure_B"] = (8 - out["record_w_B"]).clip(lower=0) / (remaining + 1)
        out["ix_kachi_pressure_diff"] = (
            out["ix_kachi_pressure_A"] - out["ix_kachi_pressure_B"]
        )

    # ------------------------------------------------------------------ #
    # 3. Confidence-weighted h2h
    # ------------------------------------------------------------------ #
    if {"h2h_winrate", "h2h_count"} <= set(out.columns):
        # Trust h2h winrate proportional to confidence (count).  Centre on 0.5.
        # log1p softens the long tail of huge h2h_count values.
        out["ix_h2h_weighted"] = (out["h2h_winrate"] - 0.5) * np.log1p(out["h2h_count"])

    # ------------------------------------------------------------------ #
    # 4. Physical mismatch
    # ------------------------------------------------------------------ #
    if {"height_diff", "weight_diff"} <= set(out.columns):
        # When one side is both taller AND heavier, the physical mismatch
        # compounds; if heavier-but-shorter (low CoM), it's actually an
        # advantage in sumo.  Encode both same-sign product and signed-pair.
        out["ix_height_x_weight"] = out["height_diff"] * out["weight_diff"]
    if {"bmi_diff", "weight_diff"} <= set(out.columns):
        # BMI captures build, weight captures absolute mass.  Both matter.
        out["ix_bmi_x_weight"] = out["bmi_diff"] * out["weight_diff"]

    # ------------------------------------------------------------------ #
    # 5. Age decline × winrate (older rikishi with poor recent form decline fast)
    # ------------------------------------------------------------------ #
    if {"age_diff", "winrate_diff_30"} <= set(out.columns):
        out["ix_age_x_winrate30"] = out["age_diff"] * out["winrate_diff_30"]

    # ------------------------------------------------------------------ #
    # 6. Style matchup × style preference strength
    # ------------------------------------------------------------------ #
    if {"pushing_ratio_A", "belt_ratio_B"} <= set(out.columns):
        # A is a pusher fighting a belt-style: classic style clash.
        out["ix_push_vs_belt"] = (
            out["pushing_ratio_A"] * out["belt_ratio_B"]
            - out["pushing_ratio_B"] * out["belt_ratio_A"]
        )

    logger.info(
        "added %d interaction columns",
        sum(1 for c in out.columns if c.startswith("ix_")),
    )
    return out


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
    df = pd.read_parquet(args.features)
    logger.info("Loaded features=%d (cols=%d)", len(df), df.shape[1])
    out_df = add_interactions(df)
    new_cols = sorted(c for c in out_df.columns if c.startswith("ix_"))
    logger.info("New columns (%d): %s", len(new_cols), new_cols)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(
        f"Saved {args.out}  rows={len(out_df)}  cols={out_df.shape[1]}  "
        f"added={len(new_cols)}"
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Add hand-crafted interaction features")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("enrich")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--out", default="data/processed/features_interact.parquet")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_enrich)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
