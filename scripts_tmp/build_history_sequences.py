"""Build per-rikishi bout-history sequences for sequence modeling.

For each (bashoId, day, matchNo, rikishiId), find this rikishi's LAST K bouts
(all bouts before the current one, chronologically). Each step is:
  (opponent_id, outcome 0/1, opponent_rankValue, days_since, is_kachikoshi_pressure)

Output: per-rikishi history dictionary mapping (bashoId, day, matchNo) → ndarray (K, F)

We pre-compute once and save as a single artifact, reusable by all sequence models.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def build():
    print("Loading raw bouts + banzuke ...")
    bouts = pd.read_parquet('data/raw/bouts.parquet')
    banzuke = pd.read_parquet('data/raw/banzuke.parquet')

    # Convert bashoId to string for sorting
    bouts['bashoId'] = bouts['bashoId'].astype(str)
    banzuke['bashoId'] = banzuke['bashoId'].astype(str)
    bouts = bouts[bouts['division'] == 'Makuuchi'].copy()  # match features.parquet scope

    # Rank lookup: (bashoId, rikishiId) -> rankValue
    bz_lookup = banzuke.set_index(['bashoId', 'rikishiId'])['rankValue'].to_dict()

    # Sort bouts chronologically
    bouts = bouts.sort_values(['bashoId', 'day', 'matchNo']).reset_index(drop=True)
    print(f"  total bouts: {len(bouts)}")

    # Build per-rikishi history
    # For each rikishi, accumulate (bashoId, day, matchNo, opponent_id, won, opp_rank)
    from collections import defaultdict
    history = defaultdict(list)  # rikishi_id -> list of (bashoId, day, matchNo, opp, won, opp_rank)
    K = 20  # max history length

    # Result: per-bout, per-side history (K, 4) arrays
    east_hist = {}  # (bashoId, day, matchNo) -> (K, 4) array
    west_hist = {}

    F = 4  # features per step: [outcome, opp_rank_normalized, days_since_prev, side_pair_indicator]

    for i, row in enumerate(bouts.itertuples(index=False)):
        if i % 5000 == 0:
            print(f"  ... {i} / {len(bouts)}")
        bid = row.bashoId
        day = int(row.day)
        mno = int(row.matchNo)
        east = int(row.eastId)
        west = int(row.westId)
        winner = int(row.winnerId)
        e_rank = bz_lookup.get((bid, east), 500)  # default mid-rank
        w_rank = bz_lookup.get((bid, west), 500)

        # Build history for THIS bout (using bouts BEFORE)
        e_history = history[east][-K:] if len(history[east]) > 0 else []
        w_history = history[west][-K:] if len(history[west]) > 0 else []

        def to_arr(hist):
            arr = np.zeros((K, F), dtype=np.float32)
            for j, (prev_bid, prev_day, _, _, won, opp_rank) in enumerate(hist[-K:]):
                arr[j, 0] = won
                arr[j, 1] = (opp_rank - 300) / 100.0  # normalize
                # days since prev bout (rough estimate using basho gaps)
                arr[j, 2] = 0.0  # placeholder
                arr[j, 3] = j / K  # position
            return arr

        east_hist[(bid, day, mno)] = to_arr(e_history)
        west_hist[(bid, day, mno)] = to_arr(w_history)

        # Append THIS bout to history (after using past for prediction)
        history[east].append((bid, day, mno, west, 1 if winner == east else 0, w_rank))
        history[west].append((bid, day, mno, east, 1 if winner == west else 0, e_rank))

    # Save as numpy arrays indexed by key
    print(f"\nBuilt {len(east_hist)} bout histories. Sample:")
    key0 = list(east_hist.keys())[1000]
    print(f"  Key {key0}: east_hist[0:3] = {east_hist[key0][:3]}")

    # Save as parquet with key columns + flat history
    rows = []
    for key, east_h in east_hist.items():
        bid, day, mno = key
        west_h = west_hist[key]
        row_dict = {'bashoId': bid, 'day': day, 'matchNo': mno}
        # Flatten east history
        for j in range(K):
            for f in range(F):
                row_dict[f'east_h{j}_f{f}'] = east_h[j, f]
        for j in range(K):
            for f in range(F):
                row_dict[f'west_h{j}_f{f}'] = west_h[j, f]
        rows.append(row_dict)
    seqdf = pd.DataFrame(rows)
    out = 'data/processed/rikishi_history_seq.parquet'
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    seqdf.to_parquet(out, index=False)
    print(f"\nSaved {out}: {seqdf.shape}")


if __name__ == "__main__":
    build()
