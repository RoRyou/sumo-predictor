"""Structural feature engineering for Route A (tabular tower).

Builds one row per bout with features computed only from bouts that happened
*before* the bout in question — strict no-leakage construction.

Pipeline:
    raw bouts (data/raw/bouts.parquet)
    + rikishi profiles (data/raw/rikishis.parquet)
    + banzuke (data/raw/banzuke.parquet)
      |
      v
    1. Build a chronological event stream (bashoId, day, matchNo)
    2. For every rikishi maintain rolling counters:
         - recent_N winrate (N in {10, 30, 90} bouts)
         - kimarite preference (pushing / belt / others)
         - in-basho streak / running record
         - days since last bout
    3. For every (rikishiA, rikishiB) pair maintain h2h counters
         - apply Bayesian shrinkage to h2h winrate
    4. Apply time-decay sample weight (exp(-lambda*delta_basho))
    5. Emit symmetric bout rows (rank_diff = A - B) using the canonical
       'east is A' convention.  Symmetric augmentation is left to the
       training pipeline (so we don't double the on-disk file size).

CLI::

    python -m src.features.structural build \\
        --raw-dir data/raw --out data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Kimarite categories.  Source: sumo-api kimarite + Japan Sumo Assoc.
# These cover ~90% of bouts; the rest go into 'other'.
PUSHING_KIMARITE = {
    "oshidashi", "oshitaoshi", "tsukidashi", "tsukitaoshi", "tsukiotoshi",
    "hatakikomi", "hikiotoshi", "tsukihiza", "sokubiotoshi",
}
BELT_KIMARITE = {
    "yorikiri", "yoritaoshi", "uwatenage", "shitatenage", "uwatedashinage",
    "shitatedashinage", "kotenage", "sukuinage", "tsukiotoshi", "kakenage",
    "kubinage", "uwatehineri", "shitatehineri", "tsuridashi", "tsuriotoshi",
    "watashikomi", "hansoku",
}
TECHNIQUE_KIMARITE = {
    "okuridashi", "okuritaoshi", "okurihikiotoshi", "uchimuso", "sotomuso",
    "kawazugake", "kekaeshi", "ketaguri", "kirikaeshi", "ashitori",
    "susoharai", "susotori", "chongake", "mitokorozeme",
}

DEFAULT_WINRATE_WINDOWS = (10, 30, 90)

# Time-decay lambda per basho (5-6 basho/year * lambda 0.05 ~= 30% decay/year)
DEFAULT_TIME_DECAY_LAMBDA = 0.05

# Bayesian shrinkage prior strength (in "virtual bouts")
DEFAULT_H2H_SHRINKAGE_ALPHA = 5


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def basho_to_int(basho_id: str | int) -> int:
    """``'202301'`` -> ``202301`` for cheap arithmetic."""
    return int(basho_id)


def basho_delta(later: str | int, earlier: str | int) -> int:
    """Return the number of basho between two ``YYYYMM`` ids (6/year)."""
    later, earlier = basho_to_int(later), basho_to_int(earlier)
    yl, ml = divmod(later, 100)
    ye, me = divmod(earlier, 100)
    # months are 1,3,5,7,9,11 -> indices 0..5
    idx_l = yl * 6 + ((ml - 1) // 2)
    idx_e = ye * 6 + ((me - 1) // 2)
    return idx_l - idx_e


def parse_rank_value(rank: str | None) -> float:
    """Map a banzuke rank string to a numeric value (lower = higher).

    The API already provides ``rankValue`` on banzuke rows; we use this
    helper only when banzuke joining fails.  Yokozuna ~ 101, Ozeki ~ 201,
    Sekiwake ~ 301, Komusubi ~ 401, Maegashira 1-17 ~ 501-517+, etc.
    """
    if rank is None:
        return np.nan
    r = rank.strip().lower()
    side_off = 1 if "west" in r else 0  # tie-breaker: west slightly lower
    if r.startswith("yokozuna"):
        return 100 + side_off
    if r.startswith("ozeki"):
        return 200 + side_off
    if r.startswith("sekiwake"):
        return 300 + side_off
    if r.startswith("komusubi"):
        return 400 + side_off
    if r.startswith("maegashira"):
        # Maegashira 1 East/West -> 501/502, etc.
        try:
            n = int(r.split()[1])
        except (IndexError, ValueError):
            n = 18
        return 500 + 2 * n + side_off
    if r.startswith("juryo"):
        try:
            n = int(r.split()[1])
        except (IndexError, ValueError):
            n = 15
        return 700 + 2 * n + side_off
    return 900.0


def classify_kimarite(k: str | None) -> str:
    """Bucket a kimarite into pushing / belt / technique / other."""
    if k is None or not isinstance(k, str) or not k:
        return "other"
    k = k.lower()
    if k in PUSHING_KIMARITE:
        return "pushing"
    if k in BELT_KIMARITE:
        return "belt"
    if k in TECHNIQUE_KIMARITE:
        return "technique"
    return "other"


def compute_age_years(birth_date: str | None, basho_id: str) -> float:
    """Approximate age in years at the start of a basho."""
    if not birth_date:
        return np.nan
    try:
        by = int(birth_date[:4])
        bm = int(birth_date[5:7])
    except (ValueError, TypeError):
        return np.nan
    y = int(basho_id[:4])
    m = int(basho_id[4:6])
    return (y - by) + (m - bm) / 12.0


# ---------------------------------------------------------------------- #
# Per-rikishi rolling state
# ---------------------------------------------------------------------- #
@dataclass
class RikishiState:
    """Mutable state for a single rikishi as we walk the bout stream."""

    recent_results: dict[int, deque[int]] = field(  # window -> 1/0
        default_factory=lambda: {n: deque(maxlen=n) for n in DEFAULT_WINRATE_WINDOWS}
    )
    kimarite_counts: dict[str, int] = field(
        default_factory=lambda: {"pushing": 0, "belt": 0, "technique": 0, "other": 0}
    )
    kimarite_wins: dict[str, int] = field(
        default_factory=lambda: {"pushing": 0, "belt": 0, "technique": 0, "other": 0}
    )
    last_basho: str | None = None
    last_day: int = 0
    last_date_idx: int = -1  # global integer index for "days since last bout"

    current_basho: str | None = None
    streak: int = 0  # positive = win streak, negative = loss streak
    record_w: int = 0
    record_l: int = 0
    bouts_this_basho: int = 0

    total_bouts: int = 0
    total_wins: int = 0

    def reset_basho(self, basho_id: str) -> None:
        self.current_basho = basho_id
        self.streak = 0
        self.record_w = 0
        self.record_l = 0
        self.bouts_this_basho = 0

    def update(self, basho_id: str, day: int, won: bool, kimarite_cat: str) -> None:
        # update rolling windows
        outcome = 1 if won else 0
        for q in self.recent_results.values():
            q.append(outcome)
        # update kimarite preference (only count when *we* won; pushing tendency)
        if won:
            self.kimarite_wins[kimarite_cat] = self.kimarite_wins.get(kimarite_cat, 0) + 1
        self.kimarite_counts[kimarite_cat] = self.kimarite_counts.get(kimarite_cat, 0) + 1
        # streak
        if won:
            self.streak = self.streak + 1 if self.streak >= 0 else 1
            self.record_w += 1
        else:
            self.streak = self.streak - 1 if self.streak <= 0 else -1
            self.record_l += 1
        self.bouts_this_basho += 1
        self.last_basho = basho_id
        self.last_day = day
        self.total_bouts += 1
        if won:
            self.total_wins += 1


# ---------------------------------------------------------------------- #
# Pair-level h2h
# ---------------------------------------------------------------------- #
@dataclass
class H2HRecord:
    a_wins: int = 0
    b_wins: int = 0

    @property
    def count(self) -> int:
        return self.a_wins + self.b_wins


# ---------------------------------------------------------------------- #
# Main feature builder
# ---------------------------------------------------------------------- #
class StructuralFeatureBuilder:
    """Walk bouts in chronological order, emit feature rows.

    Parameters
    ----------
    winrate_windows:
        Sliding windows for ``winrate_recent_N`` features.
    h2h_alpha:
        Bayesian shrinkage strength for ``h2h_winrate`` (in virtual bouts).
    time_decay_lambda:
        Lambda for ``exp(-lambda * (max_basho - basho))`` sample weight.
    """

    def __init__(
        self,
        winrate_windows: tuple[int, ...] = DEFAULT_WINRATE_WINDOWS,
        h2h_alpha: float = DEFAULT_H2H_SHRINKAGE_ALPHA,
        time_decay_lambda: float = DEFAULT_TIME_DECAY_LAMBDA,
    ) -> None:
        self.windows = winrate_windows
        self.h2h_alpha = h2h_alpha
        self.lambda_ = time_decay_lambda

    def build(
        self,
        bouts: pd.DataFrame,
        rikishis: pd.DataFrame,
        banzuke: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return the feature matrix.  ``bouts`` will be sorted in place."""
        if bouts.empty:
            return pd.DataFrame()

        bouts = bouts.copy()
        # sort by (bashoId asc, day asc, matchNo asc) for chronological walk
        bouts.sort_values(["bashoId", "day", "matchNo"], inplace=True)
        bouts.reset_index(drop=True, inplace=True)

        # global day index used for "days since last bout"
        bouts["date_idx"] = bouts.groupby("bashoId").ngroup() * 100 + bouts["day"]

        rik_idx = rikishis.set_index("id") if "id" in rikishis.columns else None

        banzuke_map: dict[tuple[str, int], dict[str, Any]] = {}
        if banzuke is not None and not banzuke.empty:
            for _, row in banzuke.iterrows():
                banzuke_map[(str(row["bashoId"]), int(row["rikishiId"]))] = {
                    "rank": row.get("rank"),
                    "rankValue": row.get("rankValue"),
                }

        states: dict[int, RikishiState] = defaultdict(RikishiState)
        h2h: dict[tuple[int, int], H2HRecord] = defaultdict(H2HRecord)

        rows: list[dict[str, Any]] = []
        max_basho_int = basho_to_int(bouts["bashoId"].iloc[-1])

        for r in bouts.itertuples(index=False):
            east_id = int(r.eastId) if not pd.isna(r.eastId) else 0
            west_id = int(r.westId) if not pd.isna(r.westId) else 0
            if east_id == 0 or west_id == 0 or east_id == west_id:
                continue
            winner_id = int(r.winnerId) if not pd.isna(r.winnerId) else 0
            if winner_id not in (east_id, west_id):
                continue  # fusen (forfeit) or missing — drop

            # ensure basho-level resets
            for rid in (east_id, west_id):
                st = states[rid]
                if st.current_basho != r.bashoId:
                    st.reset_basho(r.bashoId)

            sA, sB = states[east_id], states[west_id]
            row = self._build_row(
                r, east_id, west_id, sA, sB, h2h, rik_idx, banzuke_map, max_basho_int
            )
            rows.append(row)

            # update states AFTER feature emission (no leakage)
            kimarite_cat = classify_kimarite(getattr(r, "kimarite", None))
            east_won = winner_id == east_id
            sA.update(r.bashoId, int(r.day), east_won, kimarite_cat)
            sB.update(r.bashoId, int(r.day), not east_won, kimarite_cat)
            sA.last_date_idx = r.date_idx
            sB.last_date_idx = r.date_idx

            key = (min(east_id, west_id), max(east_id, west_id))
            rec = h2h[key]
            if (east_id, west_id) == key:
                if east_won:
                    rec.a_wins += 1
                else:
                    rec.b_wins += 1
            else:
                if east_won:
                    rec.b_wins += 1
                else:
                    rec.a_wins += 1

        return pd.DataFrame.from_records(rows)

    # ------------------------------------------------------------------ #
    def _build_row(
        self,
        bout,
        east_id: int,
        west_id: int,
        sA: RikishiState,
        sB: RikishiState,
        h2h: dict[tuple[int, int], H2HRecord],
        rik_idx: pd.DataFrame | None,
        banzuke_map: dict[tuple[str, int], dict[str, Any]],
        max_basho_int: int,
    ) -> dict[str, Any]:
        basho_id = str(bout.bashoId)
        # --- profile lookups -----------------------------------------
        prof_a = rik_idx.loc[east_id].to_dict() if rik_idx is not None and east_id in rik_idx.index else {}
        prof_b = rik_idx.loc[west_id].to_dict() if rik_idx is not None and west_id in rik_idx.index else {}
        h_a, w_a = prof_a.get("height", np.nan), prof_a.get("weight", np.nan)
        h_b, w_b = prof_b.get("height", np.nan), prof_b.get("weight", np.nan)
        bmi_a = (w_a / ((h_a / 100) ** 2)) if (h_a and w_a) else np.nan
        bmi_b = (w_b / ((h_b / 100) ** 2)) if (h_b and w_b) else np.nan
        age_a = compute_age_years(prof_a.get("birthDate"), basho_id)
        age_b = compute_age_years(prof_b.get("birthDate"), basho_id)

        # --- rank from banzuke (fallback: from bout's eastRank/westRank) -
        bz_a = banzuke_map.get((basho_id, east_id), {})
        bz_b = banzuke_map.get((basho_id, west_id), {})
        rank_a = bz_a.get("rankValue") or parse_rank_value(bout.eastRank)
        rank_b = bz_b.get("rankValue") or parse_rank_value(bout.westRank)

        # --- winrate windows -----------------------------------------
        wr_a = {n: _mean_or_nan(sA.recent_results[n]) for n in self.windows}
        wr_b = {n: _mean_or_nan(sB.recent_results[n]) for n in self.windows}

        # --- kimarite preference (style ratio) ------------------------
        push_a, belt_a = _style_ratio(sA)
        push_b, belt_b = _style_ratio(sB)

        # --- h2h with Bayesian shrinkage ------------------------------
        key = (min(east_id, west_id), max(east_id, west_id))
        rec = h2h.get(key, H2HRecord())
        if (east_id, west_id) == key:
            a_wins, b_wins = rec.a_wins, rec.b_wins
        else:
            a_wins, b_wins = rec.b_wins, rec.a_wins
        # shrink toward prior 0.5 (no info)
        h2h_count = a_wins + b_wins
        h2h_wr = (a_wins + 0.5 * self.h2h_alpha) / (h2h_count + self.h2h_alpha)

        # --- streak / record -----------------------------------------
        days_since_a = (bout.date_idx - sA.last_date_idx) if sA.last_date_idx >= 0 else 999
        days_since_b = (bout.date_idx - sB.last_date_idx) if sB.last_date_idx >= 0 else 999

        # --- time-decay weight ---------------------------------------
        delta = max_basho_int and basho_delta(max_basho_int, basho_to_int(basho_id))
        weight = math.exp(-self.lambda_ * max(delta, 0))

        return {
            # identifiers
            "bashoId": basho_id,
            "day": int(bout.day),
            "matchNo": int(bout.matchNo),
            "eastId": east_id,
            "westId": west_id,
            "y": 1 if bout.winnerId == east_id else 0,  # east wins
            "kimarite": getattr(bout, "kimarite", None),
            "sample_weight": weight,
            # categorical (raw — for target encoding downstream)
            "heya_A": prof_a.get("heya"),
            "heya_B": prof_b.get("heya"),
            "shusshin_A": prof_a.get("shusshin"),
            "shusshin_B": prof_b.get("shusshin"),
            # diff features
            "rank_diff": _safe_sub(rank_a, rank_b),
            "height_diff": _safe_sub(h_a, h_b),
            "weight_diff": _safe_sub(w_a, w_b),
            "bmi_diff": _safe_sub(bmi_a, bmi_b),
            "age_diff": _safe_sub(age_a, age_b),
            # winrate diffs
            **{f"winrate_diff_{n}": _safe_sub(wr_a[n], wr_b[n]) for n in self.windows},
            **{f"winrate_A_{n}": wr_a[n] for n in self.windows},
            **{f"winrate_B_{n}": wr_b[n] for n in self.windows},
            # style
            "pushing_ratio_A": push_a,
            "pushing_ratio_B": push_b,
            "belt_ratio_A": belt_a,
            "belt_ratio_B": belt_b,
            "style_compat": (push_a - push_b) - (belt_a - belt_b),  # if A pushes vs B belts
            # h2h
            "h2h_count": h2h_count,
            "h2h_winrate": h2h_wr,
            # in-basho state
            "streak_A": sA.streak,
            "streak_B": sB.streak,
            "record_w_A": sA.record_w,
            "record_l_A": sA.record_l,
            "record_w_B": sB.record_w,
            "record_l_B": sB.record_l,
            "day_of_basho": int(bout.day),
            "bouts_this_basho_A": sA.bouts_this_basho,
            "bouts_this_basho_B": sB.bouts_this_basho,
            # fatigue
            "days_since_last_A": days_since_a,
            "days_since_last_B": days_since_b,
            # career
            "career_winrate_A": (sA.total_wins / sA.total_bouts) if sA.total_bouts else np.nan,
            "career_winrate_B": (sB.total_wins / sB.total_bouts) if sB.total_bouts else np.nan,
            "career_bouts_A": sA.total_bouts,
            "career_bouts_B": sB.total_bouts,
        }


# ---------------------------------------------------------------------- #
def _mean_or_nan(q: deque[int]) -> float:
    return (sum(q) / len(q)) if len(q) else np.nan


def _safe_sub(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a) - float(b)


def _style_ratio(st: RikishiState) -> tuple[float, float]:
    total = sum(st.kimarite_counts.values()) or 1
    push = st.kimarite_counts.get("pushing", 0) / total
    belt = st.kimarite_counts.get("belt", 0) / total
    return push, belt


# ---------------------------------------------------------------------- #
# Symmetric augmentation (apply during training, not at build time)
# ---------------------------------------------------------------------- #
def symmetric_augment(df: pd.DataFrame) -> pd.DataFrame:
    """Return df concatenated with a mirrored copy (swap A<->B, flip label)."""
    swap_pairs = [
        ("eastId", "westId"),
        ("heya_A", "heya_B"),
        ("shusshin_A", "shusshin_B"),
        ("pushing_ratio_A", "pushing_ratio_B"),
        ("belt_ratio_A", "belt_ratio_B"),
        ("streak_A", "streak_B"),
        ("record_w_A", "record_w_B"),
        ("record_l_A", "record_l_B"),
        ("bouts_this_basho_A", "bouts_this_basho_B"),
        ("days_since_last_A", "days_since_last_B"),
        ("career_winrate_A", "career_winrate_B"),
        ("career_bouts_A", "career_bouts_B"),
    ]
    for w in DEFAULT_WINRATE_WINDOWS:
        swap_pairs.append((f"winrate_A_{w}", f"winrate_B_{w}"))

    flip_sign = [
        "rank_diff", "height_diff", "weight_diff", "bmi_diff", "age_diff",
        "style_compat",
        *(f"winrate_diff_{w}" for w in DEFAULT_WINRATE_WINDOWS),
    ]
    mirror = df.copy()
    for a, b in swap_pairs:
        if a in mirror.columns and b in mirror.columns:
            mirror[a], mirror[b] = df[b].values, df[a].values
    for c in flip_sign:
        if c in mirror.columns:
            mirror[c] = -mirror[c]
    if "h2h_winrate" in mirror.columns:
        mirror["h2h_winrate"] = 1.0 - mirror["h2h_winrate"]
    mirror["y"] = 1 - df["y"].values
    return pd.concat([df, mirror], ignore_index=True)


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


def cmd_build(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    bouts_p = raw_dir / "bouts.parquet"
    rikishis_p = raw_dir / "rikishis.parquet"
    banzuke_p = raw_dir / "banzuke.parquet"
    if not bouts_p.exists() or not rikishis_p.exists():
        print(f"Missing required input under {raw_dir}", file=sys.stderr)
        return 2

    bouts = pd.read_parquet(bouts_p)
    rikishis = pd.read_parquet(rikishis_p)
    banzuke = pd.read_parquet(banzuke_p) if banzuke_p.exists() else None

    print(
        f"Loaded bouts={len(bouts):,}  rikishis={len(rikishis):,}"
        f"  banzuke={'-' if banzuke is None else f'{len(banzuke):,}'}",
        file=sys.stderr,
    )

    builder = StructuralFeatureBuilder(
        winrate_windows=tuple(args.winrate_windows),
        h2h_alpha=args.h2h_alpha,
        time_decay_lambda=args.time_decay_lambda,
    )
    feats = builder.build(bouts, rikishis, banzuke)
    print(f"Built features: shape={feats.shape}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"Saved -> {out}  (rows={len(feats):,}, cols={feats.shape[1]})")
    print(
        f"  y mean (east-win rate): {feats['y'].mean():.4f}",
        file=sys.stderr,
    )
    if "h2h_count" in feats.columns:
        print(
            f"  h2h_count>0 share: {(feats['h2h_count'] > 0).mean():.3f}",
            file=sys.stderr,
        )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build structural features for Route A")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Build feature matrix from raw parquets")
    b.add_argument("--raw-dir", default="data/raw")
    b.add_argument("--out", default="data/processed/features.parquet")
    b.add_argument(
        "--winrate-windows", type=int, nargs="+", default=list(DEFAULT_WINRATE_WINDOWS)
    )
    b.add_argument("--h2h-alpha", type=float, default=DEFAULT_H2H_SHRINKAGE_ALPHA)
    b.add_argument(
        "--time-decay-lambda", type=float, default=DEFAULT_TIME_DECAY_LAMBDA
    )
    b.add_argument("-v", "--verbose", action="count", default=0)
    b.set_defaults(func=cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
