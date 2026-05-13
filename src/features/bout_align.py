"""Align OCR-derived video segments to sumo-api bouts.

Given a list of merged segments (each with a possible ``winner_name`` from
caption-OCR) and a target ``bashoId``, look up the rikishi id whose
shikona best matches the OCR string, then find the unique bout in that
basho they won.

Lookup strategy
---------------
For each basho we build a candidate name -> rikishi_id mapping using the
*bout-level* shikona columns (``eastShikona``/``westShikona``).  These
strings are the short single-token forms used by the sumo-api (e.g.
``"Terunofuji"``), which is the form viewers actually see on the
broadcast caption.  Falling back to ``rikishis.parquet`` would give the
multi-word full name (``"Terunofuji Haruo"``) which never matches the
caption text.

For matching:

1. Exact (case-fold) lookup against the per-basho candidate set.
2. Fuzzy fallback via :func:`rapidfuzz.process.extractOne` with
   :func:`rapidfuzz.fuzz.WRatio` and a configurable cutoff.

The ``score`` field of a segment (e.g. ``"3-1"``) is optionally used as
a *running-record* disambiguator: when a rikishi has multiple wins in
the same basho, segments are paired with bouts in increasing
``(day, matchNo)`` order, matching the highlight-reel convention of
chronological ordering.
"""

from __future__ import annotations

import argparse
import json
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Name normalisation
# ----------------------------------------------------------------------
def _normalize(name: str) -> str:
    """Upper-case, strip diacritics, collapse whitespace."""
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.upper().split())


# ----------------------------------------------------------------------
# Aligner
# ----------------------------------------------------------------------
@dataclass
class BoutMatch:
    bashoId: str
    day: int
    matchNo: int
    eastId: int
    westId: int
    winnerId: int
    kimarite: str | None
    y_east: int  # 1 if east won, 0 if west won

    def to_dict(self) -> dict[str, Any]:
        return {
            "bashoId": self.bashoId,
            "day": int(self.day),
            "matchNo": int(self.matchNo),
            "eastId": int(self.eastId),
            "westId": int(self.westId),
            "winnerId": int(self.winnerId),
            "kimarite": self.kimarite,
            "y_east": int(self.y_east),
        }


class BoutAligner:
    """Map OCR'd winner names to sumo-api bouts within a basho.

    Parameters
    ----------
    rikishis_df
        Full rikishi roster (``id, shikonaEn, ...``).  Used as a
        secondary lookup table when a name doesn't show up in the
        per-basho bout list.
    bouts_df
        All-bouts table (``bashoId, day, matchNo, eastId, eastShikona,
        westId, westShikona, winnerId, kimarite``).
    """

    def __init__(self, rikishis_df, bouts_df) -> None:
        self.rikishis = rikishis_df
        self.bouts = bouts_df
        self._basho_lookup: dict[str, dict[str, int]] = {}
        self._basho_candidates: dict[str, list[str]] = {}
        self._bouts_by_basho: dict[str, "pd.DataFrame"] = {}
        self._build_lookups()

    def _build_lookups(self) -> None:
        # primary: per-basho {normalised_short_shikona -> rikishi_id}
        for basho_id, grp in self.bouts.groupby("bashoId"):
            mapping: dict[str, int] = {}
            for _, row in grp[["eastId", "eastShikona"]].dropna().iterrows():
                key = _normalize(row["eastShikona"])
                if key:
                    mapping.setdefault(key, int(row["eastId"]))
            for _, row in grp[["westId", "westShikona"]].dropna().iterrows():
                key = _normalize(row["westShikona"])
                if key:
                    mapping.setdefault(key, int(row["westId"]))
            self._basho_lookup[str(basho_id)] = mapping
            self._basho_candidates[str(basho_id)] = list(mapping.keys())
            # Cache the full bouts dataframe per basho (for matchup-name lookup
            # which needs both eastId/westId, not just winnerId).
            self._bouts_by_basho[str(basho_id)] = grp.reset_index(drop=True)

    # ------------------------------------------------------------------
    def match_winner(
        self,
        winner_name: str,
        basho_id: str,
        cutoff: float = 0.85,
    ) -> int | None:
        """Return rikishi_id whose shikona best matches ``winner_name``.

        Falls back to a roster-wide search (via rikishis.shikonaEn first
        token) when the basho is unknown or has no candidates.
        """
        if not winner_name:
            return None
        norm = _normalize(winner_name)
        lookup = self._basho_lookup.get(str(basho_id), {})
        candidates = self._basho_candidates.get(str(basho_id), [])

        # 1) exact case-fold match within basho
        if norm in lookup:
            return int(lookup[norm])

        # 2) fuzzy WRatio fallback within basho
        if candidates:
            from rapidfuzz import fuzz, process

            hit = process.extractOne(
                norm, candidates, scorer=fuzz.WRatio, score_cutoff=cutoff * 100
            )
            if hit is not None:
                best, score, _ = hit
                logger.debug(
                    "fuzzy match %s -> %s (WRatio=%.1f)", winner_name, best, score
                )
                return int(lookup[best])

        # 3) roster-wide fallback using rikishis.shikonaEn first token
        if "shikonaEn" in self.rikishis.columns:
            shik = self.rikishis["shikonaEn"].dropna()
            first_token = shik.str.split().str[0].apply(_normalize)
            ids = self.rikishis.loc[shik.index, "id"].astype(int).tolist()
            roster = list(first_token.values)
            if norm in roster:
                return int(ids[roster.index(norm)])
            if roster:
                from rapidfuzz import fuzz, process

                hit = process.extractOne(
                    norm, roster, scorer=fuzz.WRatio, score_cutoff=cutoff * 100
                )
                if hit is not None:
                    best, score, idx = hit
                    return int(ids[idx])

        return None

    # ------------------------------------------------------------------
    def candidate_bouts(self, winner_id: int, basho_id: str):
        """Bouts in ``basho_id`` won by ``winner_id`` (chronologically)."""
        m = (self.bouts["bashoId"] == basho_id) & (self.bouts["winnerId"] == winner_id)
        return self.bouts.loc[m].sort_values(["day", "matchNo"])

    # ------------------------------------------------------------------
    def match_segment_to_bout(
        self,
        segment: dict,
        basho_id: str,
        *,
        cutoff: float = 0.85,
        prior_uses: dict[int, int] | None = None,
    ) -> BoutMatch | None:
        """Resolve one segment to a unique bout.

        Resolution order:
        1. ``winner_name`` (explicit "Winner X" caption) — most reliable.
        2. ``matchup_names`` (two-name "A vs B" graphic) — look up the bout
           where *both* names appear regardless of who won; the winner is
           then read from the bout table.
        3. ``lone_name`` (post-bout single-name banner) — same as #1.

        ``prior_uses`` is a mutable ``{rikishi_id -> count}`` map used to
        disambiguate when the same rikishi won multiple bouts in the
        basho.  The N-th time we see them in the segment list, we pick
        the N-th of their wins ordered by ``(day, matchNo)``.
        """
        # Path 1: explicit winner caption.
        winner_name = segment.get("winner_name") or segment.get("lone_name")
        if winner_name:
            rid = self.match_winner(winner_name, basho_id, cutoff=cutoff)
            if rid is not None:
                cb = self.candidate_bouts(rid, basho_id)
                if not cb.empty:
                    return self._pick_from_candidates(cb, rid, prior_uses)

        # Path 2: matchup graphic — two names.
        matchup = segment.get("matchup_names")
        if matchup and len(matchup) == 2:
            id_a = self.match_winner(matchup[0], basho_id, cutoff=cutoff)
            id_b = self.match_winner(matchup[1], basho_id, cutoff=cutoff)
            if id_a is not None and id_b is not None and id_a != id_b:
                # Find bout where both participate
                bouts_in_basho = self._bouts_by_basho.get(basho_id)
                if bouts_in_basho is not None and not bouts_in_basho.empty:
                    pair_mask = (
                        ((bouts_in_basho["eastId"] == id_a) & (bouts_in_basho["westId"] == id_b))
                        | ((bouts_in_basho["eastId"] == id_b) & (bouts_in_basho["westId"] == id_a))
                    )
                    cb = bouts_in_basho[pair_mask].sort_values(["day", "matchNo"]).reset_index(drop=True)
                    if not cb.empty:
                        idx = 0
                        if prior_uses is not None:
                            key = -1 * (min(id_a, id_b) * 10000 + max(id_a, id_b))
                            idx = prior_uses.get(key, 0)
                            prior_uses[key] = idx + 1
                        idx = min(idx, len(cb) - 1)
                        row = cb.iloc[idx]
                        return BoutMatch(
                            bashoId=str(row["bashoId"]),
                            day=int(row["day"]),
                            matchNo=int(row["matchNo"]),
                            eastId=int(row["eastId"]),
                            westId=int(row["westId"]),
                            winnerId=int(row["winnerId"]),
                            kimarite=(None if row["kimarite"] is None else str(row["kimarite"])),
                            y_east=int(int(row["winnerId"]) == int(row["eastId"])),
                        )
        return None

    def _pick_from_candidates(self, cb, rid, prior_uses):
        idx = 0
        if prior_uses is not None:
            idx = prior_uses.get(rid, 0)
            prior_uses[rid] = idx + 1
        idx = min(idx, len(cb) - 1)
        row = cb.iloc[idx]
        return BoutMatch(
            bashoId=str(row["bashoId"]),
            day=int(row["day"]),
            matchNo=int(row["matchNo"]),
            eastId=int(row["eastId"]),
            westId=int(row["westId"]),
            winnerId=int(row["winnerId"]),
            kimarite=(None if row["kimarite"] is None else str(row["kimarite"])),
            y_east=int(int(row["winnerId"]) == int(row["eastId"])),
        )

    # ------------------------------------------------------------------
    def align_segments(
        self,
        segments: list[dict],
        basho_id: str,
        *,
        cutoff: float = 0.85,
    ) -> list[dict]:
        """Annotate each segment in-place with an ``alignment`` dict (or None)."""
        prior: dict[int, int] = {}
        out: list[dict] = []
        for seg in segments:
            match = self.match_segment_to_bout(
                seg, basho_id, cutoff=cutoff, prior_uses=prior
            )
            new = dict(seg)
            new["alignment"] = match.to_dict() if match is not None else None
            out.append(new)
        return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rikishis = pd.read_parquet(args.rikishis)
    bouts = pd.read_parquet(args.bouts)
    aligner = BoutAligner(rikishis, bouts)

    payload = json.loads(Path(args.segments).read_text())
    segs = payload.get("segments", payload if isinstance(payload, list) else [])
    aligned = aligner.align_segments(segs, args.basho, cutoff=args.cutoff)

    n_resolved = sum(1 for s in aligned if s["alignment"] is not None)
    logger.info(
        "Resolved %d / %d segments to bouts in basho %s",
        n_resolved,
        len(aligned),
        args.basho,
    )

    out_payload = (
        {**payload, "segments": aligned, "alignment_basho": args.basho}
        if isinstance(payload, dict)
        else aligned
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_payload, indent=2))
    logger.info("Wrote %s", args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.features.bout_align")
    sp = p.add_subparsers(dest="cmd", required=True)
    run_p = sp.add_parser("run", help="Align segments JSON to bouts")
    run_p.add_argument("--segments", required=True, type=Path)
    run_p.add_argument("--basho", required=True, help="bashoId, e.g. 202307")
    run_p.add_argument("--rikishis", required=True, type=Path)
    run_p.add_argument("--bouts", required=True, type=Path)
    run_p.add_argument("--out", required=True, type=Path)
    run_p.add_argument("--cutoff", type=float, default=0.85)
    run_p.set_defaults(func=_cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
