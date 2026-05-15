"""Bayesian-shrunk head-to-head features.

The existing ``h2h_winrate`` is the empirical frequency of A beating B
prior to the current bout — undefined when h2h_count=0 (filled 0.5),
and extremely noisy when h2h_count∈{1,2}.  Replace with a Beta posterior
mean using a prior derived from each pair's bout-level expectation
(midpoint of rank_diff-based prior + Elo-based prior).

For each bout we expose:

* ``h2h_bayes``         — Beta posterior mean, ``(α + w) / (α + β + n)``.
                          α, β set to 2 (low-info prior centred at 0.5).
* ``h2h_strength``      — log(1 + n) effective sample size weight.
* ``h2h_residual``      — observed h2h_winrate − rank-prior expected p.
                          Captures pair-specific "matchup advantage" beyond
                          rank.

All computed strictly from bouts BEFORE the current one.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def compute_bayes_h2h(
    bouts: pd.DataFrame,
    alpha_prior: float = 2.0,
    beta_prior: float = 2.0,
) -> pd.DataFrame:
    """Walk chronologically; emit per-bout snapshot of Beta-shrunk h2h."""
    bouts = bouts.copy()
    bouts["bashoId"] = bouts["bashoId"].astype(str)
    bouts.sort_values(["bashoId", "day", "matchNo"], inplace=True)
    bouts.reset_index(drop=True, inplace=True)

    # pair (sorted_lo, sorted_hi) -> wins_lo, wins_hi
    counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])

    rows: list[dict] = []
    for r in bouts.itertuples(index=False):
        try:
            east_id = int(r.eastId)
            west_id = int(r.westId)
            winner_id = int(r.winnerId)
        except (TypeError, ValueError):
            continue
        if east_id == 0 or west_id == 0 or east_id == west_id:
            continue
        if winner_id not in (east_id, west_id):
            continue

        key = _pair_key(east_id, west_id)
        c = counts[key]
        n = c[0] + c[1]
        if east_id == key[0]:
            w_east = c[0]
        else:
            w_east = c[1]
        bayes = (alpha_prior + w_east) / (alpha_prior + beta_prior + n)
        strength = math.log(1.0 + n)
        # observed minus 0.5 prior, scaled by sqrt(n) (Cohen-h style direction)
        emp = (w_east / n) if n > 0 else 0.5
        residual = emp - 0.5
        residual_signed = residual * math.sqrt(n) if n > 0 else 0.0

        rows.append({
            "bashoId": r.bashoId,
            "day": int(r.day),
            "matchNo": int(r.matchNo),
            "eastId": east_id,
            "westId": west_id,
            "h2h_bayes": bayes,
            "h2h_strength": strength,
            "h2h_residual_signed": residual_signed,
            "h2h_emp": emp,
        })

        # update counts
        if winner_id == east_id:
            if east_id == key[0]:
                c[0] += 1
            else:
                c[1] += 1
        else:
            if west_id == key[0]:
                c[0] += 1
            else:
                c[1] += 1

    return pd.DataFrame.from_records(rows)


def enrich(features: pd.DataFrame, bayes: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["bashoId"] = features["bashoId"].astype(str)
    bayes = bayes.copy()
    bayes["bashoId"] = bayes["bashoId"].astype(str)
    key = ["bashoId", "day", "matchNo", "eastId", "westId"]
    return features.merge(bayes, on=key, how="left", suffixes=("", "_bh"))


def _setup_logging(v: int) -> None:
    logging.basicConfig(
        level=logging.INFO if v else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_enrich(args: argparse.Namespace) -> int:
    features = pd.read_parquet(args.features)
    bouts = pd.read_parquet(args.bouts)
    logger.info("Loaded features=%d, bouts=%d", len(features), len(bouts))
    bayes = compute_bayes_h2h(bouts, alpha_prior=args.alpha, beta_prior=args.beta)
    logger.info("Bayes h2h shape: %s", bayes.shape)
    out_df = enrich(features, bayes)
    added = [c for c in out_df.columns if c not in features.columns]
    logger.info("Added %d columns: %s", len(added), added)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Saved {args.out}  rows={len(out_df)}  cols={out_df.shape[1]}  added={len(added)}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bayesian h2h shrinkage features")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("enrich")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--bouts", default="data/raw/bouts.parquet")
    r.add_argument("--out", default="data/processed/features_bayes_h2h.parquet")
    r.add_argument("--alpha", type=float, default=2.0)
    r.add_argument("--beta", type=float, default=2.0)
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_enrich)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
