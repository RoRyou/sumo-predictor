"""Client for the sumo-api.com REST API.

Provides paginated access to rikishi, basho, banzuke and torikumi (bout)
endpoints with rate-limiting and retries.  The module is intended both as a
library (``from src.data.sumo_api import SumoApi``) and a CLI entry-point
(``python -m src.data.sumo_api fetch ...``).

Endpoints relied upon (verified working):
    * GET /rikishis?limit=N&skip=K&measurements=true
    * GET /rikishi/{id}
    * GET /rikishi/{id}/stats
    * GET /rikishi/{id}/matches?bashoId=YYYYMM
    * GET /basho/{bashoId}
    * GET /basho/{bashoId}/banzuke/{division}
    * GET /basho/{bashoId}/torikumi/{division}/{day}
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

API_BASE = "https://www.sumo-api.com/api"

# Six "hon-basho" (grand tournaments) per year - Jan, Mar, May, Jul, Sep, Nov
BASHO_MONTHS = (1, 3, 5, 7, 9, 11)

DEFAULT_DIVISION = "Makuuchi"
ALL_DIVISIONS = (
    "Makuuchi",
    "Juryo",
    "Makushita",
    "Sandanme",
    "Jonidan",
    "Jonokuchi",
)

# Days in a hon-basho.  Makuuchi/Juryo are 15 days; lower divisions only 7,
# but torikumi endpoint just returns 404 on missing days which we handle.
BASHO_DAYS = 15

logger = logging.getLogger(__name__)


class SumoApiError(RuntimeError):
    """Raised when the API returns a non-recoverable error."""


class SumoApi:
    """Thin wrapper around the public sumo-api.com REST API.

    Adds:
        * retry with exponential back-off
        * client-side rate limit (default 5 req/s)
        * 404 handling -> returns ``None``
    """

    def __init__(
        self,
        base_url: str = API_BASE,
        max_retries: int = 3,
        rate_per_sec: float = 5.0,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.min_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "sumo-predictor/0.0.1 (+https://github.com/)",
                "Accept": "application/json",
            }
        )
        self._last_call_at: float = 0.0

    # ------------------------------------------------------------------ #
    # Low-level
    # ------------------------------------------------------------------ #
    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        wait = self.min_interval - (now - self._last_call_at)
        if wait > 0:
            time.sleep(wait)
        self._last_call_at = time.monotonic()

    def get(self, path: str, **params: Any) -> dict[str, Any] | None:
        """Issue a GET against the API.

        Returns ``None`` on 404 (so callers can skip missing data).
        Raises :class:`SumoApiError` when retries are exhausted.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        # Drop ``None`` params so requests doesn't serialise them
        clean_params = {k: v for k, v in params.items() if v is not None}

        backoff = 1.0
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=clean_params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_err = exc
                logger.warning(
                    "GET %s (attempt %d/%d) raised %s; backing off %.1fs",
                    url,
                    attempt,
                    self.max_retries,
                    exc.__class__.__name__,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue

            if resp.status_code == 404:
                logger.debug("GET %s -> 404 (treated as missing)", url)
                return None

            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = SumoApiError(
                    f"GET {url} -> {resp.status_code}: {resp.text[:200]}"
                )
                logger.warning(
                    "GET %s -> %d (attempt %d/%d); backing off %.1fs",
                    url,
                    resp.status_code,
                    attempt,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue

            if not resp.ok:
                raise SumoApiError(
                    f"GET {url} -> {resp.status_code}: {resp.text[:200]}"
                )

            try:
                return resp.json()
            except ValueError as exc:
                raise SumoApiError(
                    f"GET {url} returned non-JSON: {resp.text[:200]}"
                ) from exc

        raise SumoApiError(
            f"GET {url} failed after {self.max_retries} retries: {last_err}"
        )

    # ------------------------------------------------------------------ #
    # Rikishi
    # ------------------------------------------------------------------ #
    def fetch_all_rikishis(
        self, measurements: bool = False, page_size: int = 1000
    ) -> pd.DataFrame:
        """Page through ``/rikishis`` until exhausted.

        Returns a DataFrame with one row per rikishi.

        Note
        ----
        ``measurements=true`` on this endpoint has been observed to silently
        disable the ``skip`` parameter (server bug), so callers paginating
        through the full list MUST leave it ``False``.  Use
        :meth:`fetch_rikishi` per-id if you need measurement history.
        """
        records: list[dict[str, Any]] = []
        skip = 0
        total = None
        pbar: tqdm | None = None

        while True:
            payload = self.get(
                "rikishis",
                limit=page_size,
                skip=skip,
                measurements="true" if measurements else None,
            )
            if not payload:
                break

            if total is None:
                total = int(payload.get("total", 0) or 0)
                pbar = tqdm(
                    total=total,
                    desc="rikishis",
                    unit="r",
                    file=sys.stderr,
                    leave=False,
                )

            batch = payload.get("records") or []
            if not batch:
                break

            records.extend(batch)
            if pbar is not None:
                pbar.update(len(batch))

            skip += len(batch)
            if total is not None and skip >= total:
                break
            # NOTE: sumo-api silently caps `limit` (observed cap = 100) so we
            # cannot use ``len(batch) < page_size`` as an end-of-data signal.
            # Trust ``total`` instead; only bail on an empty batch above.

        if pbar is not None:
            pbar.close()

        df = pd.DataFrame.from_records(records)
        if not df.empty and "id" in df.columns:
            df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
        return df

    def fetch_rikishi(self, rikishi_id: int) -> dict[str, Any] | None:
        return self.get(f"rikishi/{rikishi_id}")

    def fetch_rikishi_stats(self, rikishi_id: int) -> dict[str, Any] | None:
        return self.get(f"rikishi/{rikishi_id}/stats")

    def fetch_rikishi_matches(
        self, rikishi_id: int, basho_id: str | None = None
    ) -> dict[str, Any] | None:
        return self.get(f"rikishi/{rikishi_id}/matches", bashoId=basho_id)

    # ------------------------------------------------------------------ #
    # Basho / Banzuke / Bouts
    # ------------------------------------------------------------------ #
    def fetch_basho(self, basho_id: str) -> dict[str, Any] | None:
        """Top-level basho metadata.  ``basho_id`` is ``YYYYMM``."""
        return self.get(f"basho/{basho_id}")

    def fetch_banzuke(
        self, basho_id: str, division: str = DEFAULT_DIVISION
    ) -> dict[str, Any] | None:
        return self.get(f"basho/{basho_id}/banzuke/{division}")

    def fetch_torikumi_day(
        self, basho_id: str, division: str, day: int
    ) -> dict[str, Any] | None:
        return self.get(f"basho/{basho_id}/torikumi/{division}/{day}")

    def fetch_bouts_for_basho(
        self,
        basho_id: str,
        division: str = DEFAULT_DIVISION,
        days: int = BASHO_DAYS,
    ) -> pd.DataFrame:
        """Return one row per bout for a single basho/division.

        Tries the torikumi endpoint first (cleaner schema).  Falls back to the
        ``banzuke`` ``record[]`` per-rikishi log de-duplicated to one row per
        bout if torikumi is unavailable.

        Returned columns::

            bashoId, division, day, matchNo,
            eastId, eastShikona, eastRank,
            westId, westShikona, westRank,
            kimarite, winnerId
        """
        bouts: list[dict[str, Any]] = []
        torikumi_ok = False

        for day in range(1, days + 1):
            payload = self.fetch_torikumi_day(basho_id, division, day)
            if not payload:
                continue
            # Some basho only return the top-level wrapper without 'torikumi'
            day_bouts = payload.get("torikumi") or []
            if not day_bouts:
                continue
            torikumi_ok = True
            for bout in day_bouts:
                bouts.append(
                    {
                        "bashoId": bout.get("bashoId") or basho_id,
                        "division": bout.get("division") or division,
                        "day": bout.get("day", day),
                        "matchNo": bout.get("matchNo"),
                        "eastId": bout.get("eastId"),
                        "eastShikona": bout.get("eastShikona"),
                        "eastRank": bout.get("eastRank"),
                        "westId": bout.get("westId"),
                        "westShikona": bout.get("westShikona"),
                        "westRank": bout.get("westRank"),
                        "kimarite": bout.get("kimarite"),
                        "winnerId": bout.get("winnerId"),
                    }
                )

        if torikumi_ok:
            return pd.DataFrame.from_records(bouts)

        # Fallback: reconstruct from banzuke record[] (one log per rikishi).
        logger.info(
            "torikumi missing for %s/%s; falling back to banzuke records",
            basho_id,
            division,
        )
        banzuke = self.fetch_banzuke(basho_id, division)
        if not banzuke:
            return pd.DataFrame.from_records(bouts)

        side_lookup: dict[int, dict[str, Any]] = {}
        for side in ("east", "west"):
            for entry in banzuke.get(side) or []:
                rid = entry.get("rikishiID") or entry.get("rikishiId")
                if rid is not None:
                    side_lookup[int(rid)] = {
                        "shikona": entry.get("shikonaEn"),
                        "rank": entry.get("rank"),
                        "side": side,
                    }

        seen: set[tuple[str, int, int]] = set()
        for side in ("east", "west"):
            for entry in banzuke.get(side) or []:
                rid_raw = entry.get("rikishiID") or entry.get("rikishiId")
                if rid_raw is None:
                    continue
                rid = int(rid_raw)
                for day_idx, day_rec in enumerate(entry.get("record") or [], start=1):
                    opp_raw = day_rec.get("opponentID") or day_rec.get("opponentId")
                    if opp_raw is None:
                        continue
                    opp = int(opp_raw)
                    pair_key = (basho_id, min(rid, opp), max(rid, opp))
                    if (pair_key[0], pair_key[1], pair_key[2]) in seen:
                        continue
                    # tag with day so multi-meet pairings (rare) keep distinct
                    full_key = (f"{basho_id}-{day_idx}", min(rid, opp), max(rid, opp))
                    if full_key in seen:
                        continue
                    seen.add(full_key)

                    self_meta = side_lookup.get(rid, {})
                    opp_meta = side_lookup.get(opp, {})
                    if self_meta.get("side") == "east":
                        east_id, west_id = rid, opp
                        east_meta, west_meta = self_meta, opp_meta
                    else:
                        east_id, west_id = opp, rid
                        east_meta, west_meta = opp_meta, self_meta

                    result = (day_rec.get("result") or "").lower()
                    if result.startswith("win"):
                        winner_id: int | None = rid
                    elif result.startswith("loss") or result.startswith("lose"):
                        winner_id = opp
                    else:
                        winner_id = None

                    bouts.append(
                        {
                            "bashoId": basho_id,
                            "division": division,
                            "day": day_idx,
                            "matchNo": None,
                            "eastId": east_id,
                            "eastShikona": east_meta.get("shikona"),
                            "eastRank": east_meta.get("rank"),
                            "westId": west_id,
                            "westShikona": west_meta.get("shikona"),
                            "westRank": west_meta.get("rank"),
                            "kimarite": day_rec.get("kimarite"),
                            "winnerId": winner_id,
                        }
                    )

        return pd.DataFrame.from_records(bouts)

    def fetch_banzuke_rows(
        self, basho_id: str, division: str = DEFAULT_DIVISION
    ) -> pd.DataFrame:
        """Return one row per rikishi-per-basho with their banzuke rank/record.

        Columns: ``bashoId, division, side, rikishiId, shikonaEn, rank,
        wins, losses, absences``.
        """
        banzuke = self.fetch_banzuke(basho_id, division)
        rows: list[dict[str, Any]] = []
        if not banzuke:
            return pd.DataFrame.from_records(rows)
        for side in ("east", "west"):
            for entry in banzuke.get(side) or []:
                wins = entry.get("wins")
                losses = entry.get("losses")
                absences = entry.get("absences")
                if wins is None or losses is None or absences is None:
                    # derive from record[]
                    record = entry.get("record") or []
                    wins_d = sum(
                        1
                        for r in record
                        if (r.get("result") or "").lower().startswith("win")
                    )
                    losses_d = sum(
                        1
                        for r in record
                        if (r.get("result") or "").lower().startswith("loss")
                        or (r.get("result") or "").lower().startswith("lose")
                    )
                    absences_d = sum(
                        1
                        for r in record
                        if (r.get("result") or "").lower().startswith("absen")
                    )
                    if wins is None:
                        wins = wins_d
                    if losses is None:
                        losses = losses_d
                    if absences is None:
                        absences = absences_d
                rows.append(
                    {
                        "bashoId": basho_id,
                        "division": division,
                        "side": side,
                        "rikishiId": entry.get("rikishiID")
                        or entry.get("rikishiId"),
                        "shikonaEn": entry.get("shikonaEn"),
                        "rank": entry.get("rank"),
                        "rankValue": entry.get("rankValue"),
                        "wins": wins,
                        "losses": losses,
                        "absences": absences,
                    }
                )
        return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def bashos_in_range(start_year: int, end_year: int) -> Iterator[str]:
    """Yield basho IDs (``YYYYMM``) for the six tournaments per year inclusive."""
    for year in range(start_year, end_year + 1):
        for month in BASHO_MONTHS:
            yield f"{year:04d}{month:02d}"


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def cmd_fetch(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = SumoApi(rate_per_sec=args.rate)

    # ------------------------------------------------------------------ #
    # Rikishis
    # ------------------------------------------------------------------ #
    logger.info("Fetching rikishis (paginated)...")
    rikishi_df = api.fetch_all_rikishis(measurements=False, page_size=args.page_size)
    rikishi_path = out_dir / "rikishis.parquet"
    rikishi_df.to_parquet(rikishi_path, index=False)
    logger.info("Wrote %d rikishis -> %s", len(rikishi_df), rikishi_path)

    # ------------------------------------------------------------------ #
    # Basho / Bouts / Banzuke loop
    # ------------------------------------------------------------------ #
    basho_ids = list(bashos_in_range(args.start, args.end))
    divisions = args.divisions or [DEFAULT_DIVISION]

    basho_rows: list[dict[str, Any]] = []
    bout_frames: list[pd.DataFrame] = []
    banzuke_frames: list[pd.DataFrame] = []

    for basho_id in tqdm(basho_ids, desc="basho", unit="b", file=sys.stderr):
        meta = api.fetch_basho(basho_id)
        if meta is None:
            logger.warning("Basho %s: 404, skipping", basho_id)
            continue
        flat_meta = {
            "bashoId": basho_id,
            "date": meta.get("date"),
            "location": meta.get("location"),
            "startDate": meta.get("startDate"),
            "endDate": meta.get("endDate"),
            "yushoMakuuchi": _yusho_for(meta, "Makuuchi"),
            "yushoJuryo": _yusho_for(meta, "Juryo"),
            "yushoMakushita": _yusho_for(meta, "Makushita"),
            "specialPrizeCount": len(meta.get("specialPrizes") or []),
        }
        basho_rows.append(flat_meta)

        for division in divisions:
            try:
                bouts_df = api.fetch_bouts_for_basho(basho_id, division=division)
            except SumoApiError as exc:
                logger.warning(
                    "Bouts fetch failed for %s/%s: %s", basho_id, division, exc
                )
                bouts_df = pd.DataFrame()
            if not bouts_df.empty:
                bout_frames.append(bouts_df)

            banzuke_df = api.fetch_banzuke_rows(basho_id, division=division)
            if not banzuke_df.empty:
                banzuke_frames.append(banzuke_df)

    bashos_df = pd.DataFrame.from_records(basho_rows)
    bouts_df = (
        pd.concat(bout_frames, ignore_index=True)
        if bout_frames
        else pd.DataFrame()
    )
    banzuke_df = (
        pd.concat(banzuke_frames, ignore_index=True)
        if banzuke_frames
        else pd.DataFrame()
    )

    bashos_path = out_dir / "bashos.parquet"
    bouts_path = out_dir / "bouts.parquet"
    banzuke_path = out_dir / "banzuke.parquet"

    bashos_df.to_parquet(bashos_path, index=False)
    if not bouts_df.empty:
        bouts_df.to_parquet(bouts_path, index=False)
    if not banzuke_df.empty:
        banzuke_df.to_parquet(banzuke_path, index=False)

    logger.info("Wrote %d basho -> %s", len(bashos_df), bashos_path)
    logger.info("Wrote %d bouts -> %s", len(bouts_df), bouts_path)
    logger.info("Wrote %d banzuke rows -> %s", len(banzuke_df), banzuke_path)

    _print_summary(rikishi_df, bashos_df, bouts_df, banzuke_df)
    return 0


def _yusho_for(meta: dict[str, Any], division: str) -> str | None:
    for entry in meta.get("yusho") or []:
        if (entry.get("type") or "").lower() == division.lower():
            return entry.get("shikonaEn")
    return None


def _print_summary(
    rikishi_df: pd.DataFrame,
    bashos_df: pd.DataFrame,
    bouts_df: pd.DataFrame,
    banzuke_df: pd.DataFrame,
) -> None:
    out = sys.stderr
    print("\n=== Fetch summary ===", file=out)
    print(f"rikishis:    {len(rikishi_df):>6,d} rows", file=out)
    print(f"bashos:      {len(bashos_df):>6,d} rows", file=out)
    print(f"bouts:       {len(bouts_df):>6,d} rows", file=out)
    print(f"banzuke:     {len(banzuke_df):>6,d} rows", file=out)

    if not bouts_df.empty:
        date_min = bouts_df["bashoId"].min()
        date_max = bouts_df["bashoId"].max()
        print(f"  date range:  {date_min} .. {date_max}", file=out)

        kimarite_filled = bouts_df["kimarite"].notna() & (
            bouts_df["kimarite"].astype(str).str.strip() != ""
        )
        pct = 100.0 * kimarite_filled.mean() if len(bouts_df) else 0.0
        print(f"  kimarite filled: {pct:.1f}%", file=out)

        winner_filled = bouts_df["winnerId"].notna()
        print(
            f"  winnerId filled: {100.0 * winner_filled.mean():.1f}%",
            file=out,
        )

        top_k = (
            bouts_df.loc[kimarite_filled, "kimarite"]
            .astype(str)
            .str.lower()
            .value_counts()
            .head(10)
        )
        print("  top 10 kimarite:", file=out)
        for name, count in top_k.items():
            print(f"    {name:<18s} {count:>6,d}", file=out)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.sumo_api",
        description="Fetch sumo-api.com data and write parquet files.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="Fetch rikishis + bouts + banzuke parquet files")
    f.add_argument("--start", type=int, default=2015, help="start year (inclusive)")
    f.add_argument("--end", type=int, default=2024, help="end year (inclusive)")
    f.add_argument(
        "--out", type=str, default="data/raw/", help="output directory"
    )
    f.add_argument(
        "--divisions",
        nargs="*",
        default=[DEFAULT_DIVISION],
        help="divisions to pull bouts for",
    )
    f.add_argument("--rate", type=float, default=5.0, help="max requests per second")
    f.add_argument("--page-size", type=int, default=1000, help="rikishi page size")
    f.add_argument("-v", "--verbose", action="count", default=1)
    f.set_defaults(func=cmd_fetch)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
