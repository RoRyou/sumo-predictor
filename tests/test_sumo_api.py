"""Live sanity tests against the public sumo-api.com endpoints.

These tests do a tiny number of real network calls and rely on basho
202301 being preserved.  Mark slow if the network is flaky::

    pytest -m "not live"

to skip.
"""

from __future__ import annotations

import pytest

from src.data.sumo_api import SumoApi, bashos_in_range


@pytest.fixture(scope="module")
def api() -> SumoApi:
    # Be polite: low rate so flaky tests don't hammer the API.
    return SumoApi(rate_per_sec=2.0)


@pytest.mark.live
def test_bashos_in_range_basic() -> None:
    out = list(bashos_in_range(2023, 2023))
    assert out == ["202301", "202303", "202305", "202307", "202309", "202311"]


@pytest.mark.live
def test_fetch_basho_meta(api: SumoApi) -> None:
    meta = api.fetch_basho("202301")
    assert meta is not None, "live API returned no data for basho 202301"
    assert meta.get("date") == "202301"
    assert (meta.get("location") or "").startswith("Tokyo")
    yusho = meta.get("yusho") or []
    assert any(y.get("type", "").lower() == "makuuchi" for y in yusho)


@pytest.mark.live
def test_fetch_bouts_for_basho_202301(api: SumoApi) -> None:
    df = api.fetch_bouts_for_basho("202301", division="Makuuchi")
    assert not df.empty, "expected at least one Makuuchi bout for basho 202301"
    # 15 days × ~20 matches/day = ~300 bouts
    assert len(df) >= 200, f"got only {len(df)} bouts (expected >= 200)"
    expected_cols = {
        "bashoId",
        "division",
        "day",
        "matchNo",
        "eastId",
        "eastShikona",
        "eastRank",
        "westId",
        "westShikona",
        "westRank",
        "kimarite",
        "winnerId",
    }
    assert expected_cols.issubset(df.columns), (
        f"missing columns: {expected_cols - set(df.columns)}"
    )
    # >=80% of bouts should have a winner and a kimarite recorded
    assert df["winnerId"].notna().mean() >= 0.8
    assert df["kimarite"].notna().mean() >= 0.8


@pytest.mark.live
def test_fetch_all_rikishis_first_page(api: SumoApi) -> None:
    # Use a small page so the test is fast.
    df = api.fetch_all_rikishis(measurements=True, page_size=50)
    assert not df.empty
    assert {"id", "shikonaEn"}.issubset(df.columns)
    # The full population is ~600 - we expect at least a few hundred.
    assert len(df) >= 100
