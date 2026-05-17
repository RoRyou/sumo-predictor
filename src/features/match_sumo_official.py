"""Parse Japan Sumo Association (官方频道) bout-clip titles and align them
to features.parquet bouts via shikona JP matching.

Title format:
    大相撲　[力士A]ー[力士B]＜令和X年Y月場所・Z日目＞SUMO

Optionally with `（部屋名）` suffix on each rikishi name.

Steps:
1. Read line-delimited JSON dump from yt-dlp --flat-playlist --dump-json.
2. Parse each title; reject if not bout-clip format.
3. Convert 令和 era + Japanese month/day kanji → basho_id (YYYYMM) and integer day.
4. Build shikona-JP → rikishi_id lookup from rikishis.parquet (uses shikonaJp).
5. For each parsed bout, try to find a matching feature row with same
   (basho_id, day) and either (east_shikona_jp matches A AND west matches B)
   OR (A↔B swap).
6. Emit a parquet table: (video_id, duration, basho_id, day, matchNo,
   eastId, westId, score) for matches.

CLI::

    python -m src.features.match_sumo_official build \\
        --listing /tmp/sumo_official_all.jsonl \\
        --rikishis data/raw/rikishis.parquet \\
        --features data/processed/features.parquet \\
        --out data/processed/sumo_official_alignment.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# 令和元年 = 2019, 令和2年 = 2020, ..., 令和6年 = 2024, 令和7年 = 2025
REIWA_BASE = 2018  # so reiwa_year + REIWA_BASE = western year
HEISEI_BASE = 1988  # heisei_year + HEISEI_BASE = western year (e.g., 平成27年 = 2015)

KANJI_NUM = {
    "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
}
MONTH_KANJI = {
    "一": 1, "二": 3, "三": 5, "四": 7, "五": 9, "六": 11,
    # The above maps 1月→Jan(01), 2月→Mar(03), ... but the official channel uses
    # actual month names: 一月=Jan→01, 三月=Mar→03, 五月=May→05, 七月=Jul→07,
    # 九月=Sep→09, 十一月=Nov→11. So the simple `一二三...` mapping returns 1..6
    # but actual basho month is in {1, 3, 5, 7, 9, 11}.
}
# Correct: the kanji *digit* maps directly to month: 一月=1, 三月=3, ...
KANJI_DIGIT = KANJI_NUM


def normalize_digits(s: str) -> str:
    """Convert fullwidth ０-９ to ASCII 0-9."""
    out = []
    for ch in s:
        if "０" <= ch <= "９":
            out.append(chr(ord(ch) - ord("０") + ord("0")))
        else:
            out.append(ch)
    return "".join(out)


def kanji_to_int(s: str) -> int | None:
    """Convert Japanese kanji digits/ordinals to integer (supports up to 15)."""
    s = normalize_digits(s.strip())
    if s in KANJI_DIGIT:
        return KANJI_DIGIT[s]
    if s.isdigit():
        return int(s)
    # Compound: 十一 = 10+1 etc; handled by KANJI_DIGIT
    return None


# Title pattern
RE_TITLE = re.compile(
    r"大相撲[\s　]*(?P<a>.+?)[ーー\-](?P<b>.+?)[＜<]"
    r"(?:令和|平成)(?P<era_year>[一二三四五六七八九十0-9０-９]+)年"
    r"(?P<month>[一二三四五六七八九十0-9０-９]+)月場所[・\s]*"
    r"(?P<day>[一二三四五六七八九十0-9０-９]+|初|千秋楽)日?目?"
)


def parse_title(title: str) -> dict | None:
    """Parse a bout-clip title into structured fields."""
    if "大相撲" not in title:
        return None
    if "幕下上位" in title or "幕下五番" in title or "幕下上位五番" in title:
        return None  # not a single-bout clip
    m = RE_TITLE.search(title)
    if not m:
        return None
    era_year_str = m.group("era_year")
    month_str = m.group("month")
    day_str = m.group("day")

    era_year = kanji_to_int(era_year_str)
    if era_year is None:
        return None
    if "平成" in title:
        year = HEISEI_BASE + era_year
    else:
        year = REIWA_BASE + era_year

    month = kanji_to_int(month_str)
    if month is None:
        return None
    basho_id = f"{year}{month:02d}"

    if day_str == "初":
        day = 1
    elif day_str == "千秋楽":
        day = 15
    else:
        day = kanji_to_int(day_str)
        if day is None:
            return None

    a_raw = m.group("a").strip()
    b_raw = m.group("b").strip()
    # Strip (部屋名) suffix
    a = re.sub(r"[（(].*?[）)]", "", a_raw).strip()
    b = re.sub(r"[（(].*?[）)]", "", b_raw).strip()
    return {
        "basho_id": basho_id,
        "day": day,
        "rikishi_a": a,
        "rikishi_b": b,
        "rikishi_a_raw": a_raw,
        "rikishi_b_raw": b_raw,
    }


def parse_listing(path: Path) -> pd.DataFrame:
    rows = []
    skipped = 0
    not_bout = 0
    for line in path.open("r"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        title = d.get("title", "") or ""
        vid = d.get("id", "") or ""
        dur = d.get("duration", 0) or 0
        if "大相撲" not in title:
            not_bout += 1
            continue
        parsed = parse_title(title)
        if parsed is None:
            skipped += 1
            continue
        rows.append({
            "video_id": vid,
            "duration": dur,
            "title": title,
            **parsed,
        })
    logger.info("Parsed: %d, Skipped (大相撲 but no match): %d, Non-bout: %d",
                len(rows), skipped, not_bout)
    return pd.DataFrame.from_records(rows)


def align_to_features(
    videos: pd.DataFrame,
    rikishis: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """For each video, find a matching feature row (basho_id, day, east, west).

    Match by shikonaJp (with rapidfuzz fallback to handle minor variants).
    """
    from rapidfuzz import process, fuzz

    r = rikishis.copy()
    # shikonaJp formats: '遠藤(えんどう)' / '宝富士　大輔' / '朝乃山広暉'.
    # Channel titles use only the ring name (e.g. '宝富士'), which is
    # the part before any whitespace OR (furigana).
    def clean_shikona(s):
        if pd.isna(s):
            return ""
        s = str(s)
        # Cut at furigana paren or any whitespace
        s = re.split(r"[（(\s　]", s, maxsplit=1)[0]
        return s.strip()
    r["shikonaJp_clean"] = r["shikonaJp"].apply(clean_shikona)
    # Build map: cleaned -> id. Skip empties.
    name_to_id = {}
    for n, i in zip(r["shikonaJp_clean"], r["id"]):
        if n and n not in name_to_id:
            name_to_id[n] = int(i)
    all_names = list(name_to_id.keys())
    logger.info("Built shikona index: %d names from %d rikishi", len(name_to_id), len(r))

    def resolve(name: str) -> int | None:
        if not name:
            return None
        if name in name_to_id:
            return name_to_id[name]
        # Fuzzy fallback (high threshold to avoid false matches)
        m = process.extractOne(name, all_names, scorer=fuzz.ratio, score_cutoff=92)
        if m:
            return name_to_id[m[0]]
        return None

    feats = features.copy()
    feats["bashoId"] = feats["bashoId"].astype(str)
    # Index by (basho, day) for quick lookup
    by_bd: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for r in feats.itertuples(index=False):
        by_bd[(r.bashoId, int(r.day))].append((int(r.eastId), int(r.westId), int(r.matchNo)))

    out_rows = []
    n_no_basho = 0
    n_no_match = 0
    for v in videos.itertuples(index=False):
        key = (v.basho_id, v.day)
        candidates = by_bd.get(key, [])
        if not candidates:
            n_no_basho += 1
            continue
        a_id = resolve(v.rikishi_a)
        b_id = resolve(v.rikishi_b)
        if a_id is None or b_id is None:
            n_no_match += 1
            continue
        # Look for (a→east, b→west) or (b→east, a→west)
        found = None
        for east, west, mn in candidates:
            if (east == a_id and west == b_id) or (east == b_id and west == a_id):
                found = (east, west, mn)
                break
        if found is None:
            n_no_match += 1
            continue
        east, west, mn = found
        out_rows.append({
            "video_id": v.video_id,
            "duration": v.duration,
            "title": v.title,
            "basho_id": v.basho_id,
            "day": v.day,
            "matchNo": mn,
            "eastId": east,
            "westId": west,
            "rikishi_a": v.rikishi_a,
            "rikishi_b": v.rikishi_b,
        })
    logger.info("Aligned: %d videos. Unmapped basho: %d, no-name-match: %d",
                len(out_rows), n_no_basho, n_no_match)
    return pd.DataFrame.from_records(out_rows)


def _setup_logging(v: int) -> None:
    logging.basicConfig(
        level=logging.INFO if v else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_build(args: argparse.Namespace) -> int:
    videos = parse_listing(Path(args.listing))
    print(f"Parsed videos: {len(videos)}")
    if len(videos) == 0:
        return 1

    # Distribution by basho
    by_basho = videos.groupby("basho_id").size().sort_index()
    print("Videos per basho (parsed only):")
    print(by_basho.to_string())

    rikishis = pd.read_parquet(args.rikishis)
    features = pd.read_parquet(args.features)
    aligned = align_to_features(videos, rikishis, features)
    print()
    print(f"Aligned bouts: {len(aligned)}")
    if len(aligned):
        print("Aligned per basho:")
        print(aligned.groupby("basho_id").size().sort_index().to_string())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(args.out, index=False)
    print(f"Saved {args.out}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parse & align Japan Sumo Association bout clips")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("build")
    r.add_argument("--listing", required=True, help="JSON-lines from yt-dlp --flat-playlist --dump-json")
    r.add_argument("--rikishis", default="data/raw/rikishis.parquet")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--out", default="data/processed/sumo_official_alignment.parquet")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_build)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
