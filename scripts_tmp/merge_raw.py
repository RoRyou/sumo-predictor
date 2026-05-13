"""One-off merge of 2008-2014 + 2015-2024 raw parquets back into data/raw/.

After merging, scans bouts for any eastId/westId not in rikishis and fetches them
individually using SumoApi.fetch_rikishi.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/liang.lu/project/sumo_pred")
RAW_DIR = ROOT / "data" / "raw"
NEW_DIR = ROOT / "data" / "raw_2008_2014"
BACKUP_DIR = RAW_DIR / ".backup_2015_2024"

assert BACKUP_DIR.exists(), "backup must exist before merge"


def load_pair(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_parquet(BACKUP_DIR / f"{name}.parquet")
    new_path = NEW_DIR / f"{name}.parquet"
    new = pd.read_parquet(new_path) if new_path.exists() else pd.DataFrame()
    return old, new


# bouts -------------------------------------------------------------------
old_b, new_b = load_pair("bouts")
print(f"[bouts] old={len(old_b)} new={len(new_b)}", flush=True)
combined = pd.concat([new_b, old_b], ignore_index=True)
# dedupe on the natural key
dedup_cols = ["bashoId", "day", "matchNo", "eastId", "westId"]
# matchNo can be NaN in fallback; include kimarite/winnerId? Use full key fallback
before = len(combined)
combined = combined.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
print(f"[bouts] merged={len(combined)} (dropped {before-len(combined)} dups)")
combined.to_parquet(RAW_DIR / "bouts.parquet", index=False)
merged_bouts = combined

# bashos ------------------------------------------------------------------
old_x, new_x = load_pair("bashos")
print(f"[bashos] old={len(old_x)} new={len(new_x)}")
combined = pd.concat([new_x, old_x], ignore_index=True)
before = len(combined)
combined = combined.drop_duplicates(subset=["bashoId"], keep="last").reset_index(drop=True)
print(f"[bashos] merged={len(combined)} (dropped {before-len(combined)} dups)")
combined.to_parquet(RAW_DIR / "bashos.parquet", index=False)

# banzuke -----------------------------------------------------------------
old_z, new_z = load_pair("banzuke")
print(f"[banzuke] old={len(old_z)} new={len(new_z)}")
combined = pd.concat([new_z, old_z], ignore_index=True)
before = len(combined)
combined = combined.drop_duplicates(subset=["bashoId", "rikishiId"], keep="last").reset_index(drop=True)
print(f"[banzuke] merged={len(combined)} (dropped {before-len(combined)} dups)")
combined.to_parquet(RAW_DIR / "banzuke.parquet", index=False)

# rikishis ----------------------------------------------------------------
old_r, new_r = load_pair("rikishis")
print(f"[rikishis] old={len(old_r)} new={len(new_r)}")
combined = pd.concat([old_r, new_r], ignore_index=True)
# updatedAt to keep latest
if "updatedAt" in combined.columns:
    combined = combined.sort_values("updatedAt").drop_duplicates(subset=["id"], keep="last")
else:
    combined = combined.drop_duplicates(subset=["id"], keep="last")
combined = combined.reset_index(drop=True)
print(f"[rikishis] merged={len(combined)}")

# Find missing IDs in bouts -----------------------------------------------
bout_ids = set(merged_bouts["eastId"].dropna().astype(int).tolist()
               + merged_bouts["westId"].dropna().astype(int).tolist())
have_ids = set(combined["id"].dropna().astype(int).tolist())
missing = sorted(bout_ids - have_ids)
print(f"[rikishis] bout IDs not in rikishis = {len(missing)}")

if missing:
    sys.path.insert(0, str(ROOT))
    from src.data.sumo_api import SumoApi

    api = SumoApi(rate_per_sec=5.0)
    new_rows = []
    for i, rid in enumerate(missing, 1):
        try:
            rec = api.fetch_rikishi(rid)
        except Exception as exc:
            print(f"  fetch {rid} failed: {exc}")
            continue
        if rec is None:
            continue
        new_rows.append(rec)
        if i % 20 == 0:
            print(f"  fetched {i}/{len(missing)}", flush=True)
    print(f"[rikishis] retrieved {len(new_rows)} individual records")
    if new_rows:
        add_df = pd.DataFrame.from_records(new_rows)
        # align columns
        for c in combined.columns:
            if c not in add_df.columns:
                add_df[c] = None
        add_df = add_df[combined.columns]
        combined = pd.concat([combined, add_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    print(f"[rikishis] final={len(combined)}")

combined.to_parquet(RAW_DIR / "rikishis.parquet", index=False)
print("DONE")
