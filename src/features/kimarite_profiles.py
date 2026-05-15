"""Kimarite offensive & vulnerability profiles + cross-product match feature.

For each rikishi we build two probability vectors over kimarite *categories*
(coarse buckets to avoid sparsity):

* ``offensive[r][cat]``   = P(rikishi r wins by technique-category cat | wins)
* ``vulnerability[r][cat]`` = P(rikishi r loses to technique-category cat | loses)

Both are computed chronologically with exponential time-decay (recent bouts
weighted more) over the rikishi's prior bouts strictly before the current bout.

The bout-level feature we expose is a **scalar advantage**:

    style_adv_A = sum_c offensive_A[c] * vulnerability_B[c]
                - sum_c offensive_B[c] * vulnerability_A[c]

This is a *single scalar* signed feature: positive ⇒ A's strengths match
B's weaknesses better than vice-versa.  We also expose the two raw scores
``style_match_A`` and ``style_match_B`` so the GBDT can interact with other
features.

Five coarse categories (covers ~95% of decisions):

* push       : oshidashi, tsukidashi, oshitaoshi, tsukitaoshi
* force-out  : yorikiri, yoritaoshi
* slap-down  : hatakikomi, hikiotoshi, tsukiotoshi
* throw      : *nage family (uwatenage, shitatenage, kotenage, sukuinage, ...)
* trick/edge : okuridashi, katasukashi, kimedashi, tottari, ...
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

CATEGORIES = ["push", "force_out", "slap_down", "throw", "trick"]

KIMARITE_MAP = {
    # push
    "oshidashi": "push", "tsukidashi": "push", "oshitaoshi": "push",
    "tsukitaoshi": "push",
    # force out
    "yorikiri": "force_out", "yoritaoshi": "force_out",
    "abisetaoshi": "force_out",
    # slap-down / pull
    "hatakikomi": "slap_down", "hikiotoshi": "slap_down",
    "tsukiotoshi": "slap_down",
    # throw
    "uwatenage": "throw", "shitatenage": "throw", "kotenage": "throw",
    "sukuinage": "throw", "uwatedashinage": "throw", "shitatedashinage": "throw",
    "kubinage": "throw", "tsukaminage": "throw", "yagura nage": "throw",
    "yaguranage": "throw", "tsuridashi": "throw", "kakenage": "throw",
    # trick / edge / leg
    "okuridashi": "trick", "katasukashi": "trick", "kimedashi": "trick",
    "tottari": "trick", "sotogake": "trick", "uchigake": "trick",
    "kekaeshi": "trick", "ashitori": "trick", "watashikomi": "trick",
    "kawazugake": "trick", "chongake": "trick", "nichonage": "trick",
}


def kimarite_to_category(k: str) -> str | None:
    if k is None or pd.isna(k):
        return None
    return KIMARITE_MAP.get(k.lower())


def compute_profile_features(
    bouts: pd.DataFrame,
    half_life_basho: float = 6.0,
) -> pd.DataFrame:
    """Walk chronologically; for each bout emit per-rikishi profile scores
    snapshot-as-of-just-before that bout.

    half_life_basho: half-life in *basho* units (6 basho ≈ 1 year).
    """
    bouts = bouts.copy()
    bouts["bashoId"] = bouts["bashoId"].astype(str)
    bouts.sort_values(["bashoId", "day", "matchNo"], inplace=True)
    bouts.reset_index(drop=True, inplace=True)

    # decay constant per basho in number-of-basho units
    decay = math.log(2.0) / half_life_basho

    # store counts per (rikishi, cat) with exponentially-weighted decay
    win_counts: dict[int, dict[str, float]] = defaultdict(lambda: {c: 0.0 for c in CATEGORIES})
    loss_counts: dict[int, dict[str, float]] = defaultdict(lambda: {c: 0.0 for c in CATEGORIES})
    win_total: dict[int, float] = defaultdict(float)
    loss_total: dict[int, float] = defaultdict(float)

    # convert bashoId to a numeric ordinal (YYYYMM → epoch-ish)
    unique_basho = sorted(bouts["bashoId"].unique())
    basho_ordinal = {b: i for i, b in enumerate(unique_basho)}
    bouts["b_ord"] = bouts["bashoId"].map(basho_ordinal)
    current_basho_ord = -1

    def decay_state(target_ord: int) -> None:
        """Decay all counts to target_ord."""
        nonlocal current_basho_ord
        if target_ord == current_basho_ord or current_basho_ord < 0:
            current_basho_ord = target_ord
            return
        dt = target_ord - current_basho_ord
        if dt <= 0:
            return
        factor = math.exp(-decay * dt)
        for d in (win_counts, loss_counts):
            for rid in list(d.keys()):
                for c in CATEGORIES:
                    d[rid][c] *= factor
        for t in (win_total, loss_total):
            for rid in list(t.keys()):
                t[rid] *= factor
        current_basho_ord = target_ord

    UNIFORM = np.array([1.0 / len(CATEGORIES)] * len(CATEGORIES))
    PRIOR_STRENGTH = 3.0  # Dirichlet-style smoothing

    def offensive(rid: int) -> np.ndarray:
        n = win_total.get(rid, 0.0)
        if n == 0:
            return UNIFORM.copy()
        vec = np.array([win_counts[rid][c] for c in CATEGORIES])
        return (vec + PRIOR_STRENGTH * UNIFORM) / (n + PRIOR_STRENGTH)

    def vulnerability(rid: int) -> np.ndarray:
        n = loss_total.get(rid, 0.0)
        if n == 0:
            return UNIFORM.copy()
        vec = np.array([loss_counts[rid][c] for c in CATEGORIES])
        return (vec + PRIOR_STRENGTH * UNIFORM) / (n + PRIOR_STRENGTH)

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

        b_ord = r.b_ord
        decay_state(b_ord)

        off_a = offensive(east_id)
        off_b = offensive(west_id)
        vul_a = vulnerability(east_id)
        vul_b = vulnerability(west_id)

        style_match_a = float(np.dot(off_a, vul_b))
        style_match_b = float(np.dot(off_b, vul_a))

        rows.append({
            "bashoId": r.bashoId,
            "day": int(r.day),
            "matchNo": int(r.matchNo),
            "eastId": east_id,
            "westId": west_id,
            "style_match_A": style_match_a,
            "style_match_B": style_match_b,
            "style_adv": style_match_a - style_match_b,
            **{f"off_A_{c}": off_a[i] for i, c in enumerate(CATEGORIES)},
            **{f"off_B_{c}": off_b[i] for i, c in enumerate(CATEGORIES)},
            **{f"vul_A_{c}": vul_a[i] for i, c in enumerate(CATEGORIES)},
            **{f"vul_B_{c}": vul_b[i] for i, c in enumerate(CATEGORIES)},
        })

        cat = kimarite_to_category(r.kimarite)
        if cat is None:
            continue
        if winner_id == east_id:
            win_counts[east_id][cat] += 1.0
            loss_counts[west_id][cat] += 1.0
            win_total[east_id] += 1.0
            loss_total[west_id] += 1.0
        else:
            win_counts[west_id][cat] += 1.0
            loss_counts[east_id][cat] += 1.0
            win_total[west_id] += 1.0
            loss_total[east_id] += 1.0

    return pd.DataFrame.from_records(rows)


def enrich(features: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["bashoId"] = features["bashoId"].astype(str)
    profile = profile.copy()
    profile["bashoId"] = profile["bashoId"].astype(str)
    key = ["bashoId", "day", "matchNo", "eastId", "westId"]
    return features.merge(profile, on=key, how="left", suffixes=("", "_kp"))


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
    profile = compute_profile_features(bouts, half_life_basho=args.half_life)
    logger.info("Profile shape: %s", profile.shape)
    out_df = enrich(features, profile)
    added = [c for c in out_df.columns if c not in features.columns]
    logger.info("Added %d columns", len(added))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Saved {args.out}  rows={len(out_df)}  cols={out_df.shape[1]}  added={len(added)}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kimarite offensive/vulnerability profile features")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("enrich")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--bouts", default="data/raw/bouts.parquet")
    r.add_argument("--out", default="data/processed/features_kimarite.parquet")
    r.add_argument("--half-life", type=float, default=6.0,
                   help="Half-life of profile decay in basho units (6 basho = 1 year)")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_enrich)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
