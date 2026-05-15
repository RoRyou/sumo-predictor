"""Elo + TrueSkill skill ratings and upset-rate features.

These are oversights in the previous feature engineering: we used ``rank_diff``
(positional) and ``winrate_diff_*`` (windowed averages) but never an
*adaptive skill rating* that updates after every bout.  Elo is the classic
single feature in sports prediction; TrueSkill (Bayesian) adds an explicit
uncertainty term.  Upset-rate captures how often each rikishi beats
higher-ranked opponents — orthogonal to overall winrate.

For each per-bout row we add:

  elo_A, elo_B, elo_diff
  ts_mu_A, ts_mu_B, ts_mu_diff
  ts_sigma_A, ts_sigma_B
  upset_rate_A, upset_rate_B          (frequency of wins vs higher-ranked)
  upset_rate_diff
  bouts_seen_A, bouts_seen_B          (how many prior bouts in our window)

All values are computed using ONLY bouts strictly before the current one —
no leakage.

CLI::

    python -m src.features.skill_ratings enrich \\
        --features data/processed/features.parquet \\
        --bouts data/raw/bouts.parquet \\
        --banzuke data/raw/banzuke.parquet \\
        --out data/processed/features_skill.parquet
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


# ---------------------------------------------------------------------- #
# Elo
# ---------------------------------------------------------------------- #
class EloTracker:
    """Standard chess-style Elo with K-factor tuned for sumo (15 bouts per
    basho × 6 basho/year — relatively fast turnover, K=24 is reasonable)."""

    def __init__(self, k: float = 24.0, initial: float = 1500.0) -> None:
        self.k = k
        self.initial = initial
        self.ratings: dict[int, float] = defaultdict(lambda: initial)
        self.history: dict[int, list[float]] = defaultdict(list)

    def get(self, rikishi_id: int) -> float:
        return self.ratings.get(rikishi_id, self.initial)

    def expected(self, a: int, b: int) -> float:
        ra, rb = self.get(a), self.get(b)
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, winner: int, loser: int) -> None:
        rw, rl = self.get(winner), self.get(loser)
        ew = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
        delta = self.k * (1.0 - ew)
        self.ratings[winner] = rw + delta
        self.ratings[loser] = rl - delta
        self.history[winner].append(self.ratings[winner])
        self.history[loser].append(self.ratings[loser])


# ---------------------------------------------------------------------- #
# TrueSkill
# ---------------------------------------------------------------------- #
class TrueSkillTracker:
    """Bayesian skill with Microsoft's TrueSkill (1-v-1)."""

    def __init__(self) -> None:
        import trueskill

        self.env = trueskill.TrueSkill(draw_probability=0.0)
        self.ratings: dict[int, "trueskill.Rating"] = {}

    def _get(self, rikishi_id: int):
        import trueskill

        if rikishi_id not in self.ratings:
            self.ratings[rikishi_id] = self.env.create_rating()
        return self.ratings[rikishi_id]

    def mu(self, rikishi_id: int) -> float:
        return float(self._get(rikishi_id).mu)

    def sigma(self, rikishi_id: int) -> float:
        return float(self._get(rikishi_id).sigma)

    def update(self, winner: int, loser: int) -> None:
        rw, rl = self._get(winner), self._get(loser)
        new_w, new_l = self.env.rate_1vs1(rw, rl)
        self.ratings[winner] = new_w
        self.ratings[loser] = new_l


# ---------------------------------------------------------------------- #
# Upset rate (count + rate of beating higher-ranked opponents)
# ---------------------------------------------------------------------- #
class UpsetTracker:
    """Track per-rikishi:
        * total bouts seen
        * bouts against a higher-ranked opponent (lower rankValue)
        * wins against a higher-ranked opponent
    """

    def __init__(self) -> None:
        self.bouts: dict[int, int] = defaultdict(int)
        self.bouts_vs_higher: dict[int, int] = defaultdict(int)
        self.wins_vs_higher: dict[int, int] = defaultdict(int)

    def get_rate(self, rikishi_id: int) -> float:
        denom = self.bouts_vs_higher.get(rikishi_id, 0)
        if denom == 0:
            return 0.5  # uninformative prior
        return self.wins_vs_higher[rikishi_id] / denom

    def update(self, winner: int, loser: int, winner_rank: float, loser_rank: float) -> None:
        self.bouts[winner] += 1
        self.bouts[loser] += 1
        # lower rankValue = higher position
        if not math.isnan(winner_rank) and not math.isnan(loser_rank):
            if winner_rank > loser_rank:  # winner was lower-ranked → upset
                self.bouts_vs_higher[winner] += 1
                self.wins_vs_higher[winner] += 1
                self.bouts_vs_higher[loser] += 1  # lost AS higher-ranked (no win to count)
            elif winner_rank < loser_rank:  # winner was higher-ranked → expected
                self.bouts_vs_higher[loser] += 1


# ---------------------------------------------------------------------- #
# Build per-bout feature DataFrame
# ---------------------------------------------------------------------- #
def compute_skill_features(
    bouts: pd.DataFrame,
    banzuke: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Walk bouts chronologically, emit one row per bout with skill features
    snapshot-as-of-just-before that bout."""
    bouts = bouts.copy()
    bouts["bashoId"] = bouts["bashoId"].astype(str)
    bouts.sort_values(["bashoId", "day", "matchNo"], inplace=True)
    bouts.reset_index(drop=True, inplace=True)

    # Build rank lookup (bashoId, rikishiId) -> rankValue
    rank_lookup: dict[tuple[str, int], float] = {}
    if banzuke is not None and not banzuke.empty:
        bz = banzuke.copy()
        bz["bashoId"] = bz["bashoId"].astype(str)
        for _, row in bz.iterrows():
            try:
                rid = int(row["rikishiId"])
                rv = float(row["rankValue"]) if pd.notna(row.get("rankValue")) else np.nan
            except (TypeError, ValueError):
                continue
            rank_lookup[(str(row["bashoId"]), rid)] = rv

    elo = EloTracker(k=24.0)
    ts = TrueSkillTracker()
    upset = UpsetTracker()

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

        # Snapshot BEFORE update
        bid = r.bashoId
        east_rank = rank_lookup.get((bid, east_id), np.nan)
        west_rank = rank_lookup.get((bid, west_id), np.nan)

        elo_a, elo_b = elo.get(east_id), elo.get(west_id)
        ts_mu_a, ts_mu_b = ts.mu(east_id), ts.mu(west_id)
        ts_sg_a, ts_sg_b = ts.sigma(east_id), ts.sigma(west_id)
        ups_a, ups_b = upset.get_rate(east_id), upset.get_rate(west_id)
        bs_a, bs_b = upset.bouts.get(east_id, 0), upset.bouts.get(west_id, 0)

        rows.append({
            "bashoId": bid,
            "day": int(r.day),
            "matchNo": int(r.matchNo),
            "eastId": east_id,
            "westId": west_id,
            "elo_A": elo_a,
            "elo_B": elo_b,
            "elo_diff": elo_a - elo_b,
            "elo_expected_A": elo.expected(east_id, west_id),
            "ts_mu_A": ts_mu_a,
            "ts_mu_B": ts_mu_b,
            "ts_mu_diff": ts_mu_a - ts_mu_b,
            "ts_sigma_A": ts_sg_a,
            "ts_sigma_B": ts_sg_b,
            "ts_skill_A": ts_mu_a - 3 * ts_sg_a,  # conservative skill estimate
            "ts_skill_B": ts_mu_b - 3 * ts_sg_b,
            "ts_skill_diff": (ts_mu_a - 3 * ts_sg_a) - (ts_mu_b - 3 * ts_sg_b),
            "upset_rate_A": ups_a,
            "upset_rate_B": ups_b,
            "upset_rate_diff": ups_a - ups_b,
            "bouts_seen_A": bs_a,
            "bouts_seen_B": bs_b,
        })

        # Now UPDATE trackers
        winner, loser = (east_id, west_id) if winner_id == east_id else (west_id, east_id)
        winner_rank = east_rank if winner == east_id else west_rank
        loser_rank = east_rank if loser == east_id else west_rank
        elo.update(winner, loser)
        ts.update(winner, loser)
        upset.update(winner, loser, winner_rank, loser_rank)

    return pd.DataFrame.from_records(rows)


def enrich(features: pd.DataFrame, skill: pd.DataFrame) -> pd.DataFrame:
    """Merge skill features into the per-bout feature frame on
    (bashoId, day, matchNo, eastId, westId)."""
    features = features.copy()
    features["bashoId"] = features["bashoId"].astype(str)
    skill = skill.copy()
    skill["bashoId"] = skill["bashoId"].astype(str)
    key = ["bashoId", "day", "matchNo", "eastId", "westId"]
    merged = features.merge(skill, on=key, how="left", suffixes=("", "_skill"))
    return merged


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
    bouts = pd.read_parquet(args.bouts)
    banzuke = pd.read_parquet(args.banzuke) if args.banzuke else None
    logger.info("Loaded features=%d, bouts=%d", len(features), len(bouts))
    skill = compute_skill_features(bouts, banzuke)
    logger.info("Skill features shape: %s", skill.shape)
    out_df = enrich(features, skill)
    added = [c for c in out_df.columns if c not in features.columns]
    logger.info("Added %d columns: %s", len(added), added)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Saved {args.out}  rows={len(out_df)}  cols={out_df.shape[1]}  added={len(added)}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Elo + TrueSkill + upset-rate features")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("enrich")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--bouts", default="data/raw/bouts.parquet")
    r.add_argument("--banzuke", default="data/raw/banzuke.parquet")
    r.add_argument("--out", default="data/processed/features_skill.parquet")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_enrich)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
