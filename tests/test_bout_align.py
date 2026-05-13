"""Unit tests for :mod:`src.features.bout_align`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features.bout_align import BoutAligner, _normalize

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def _load_real():
    rik = pd.read_parquet(DATA / "rikishis.parquet")
    bts = pd.read_parquet(DATA / "bouts.parquet")
    return rik, bts


# ---------------------------------------------------------------------- #
# Normalisation
# ---------------------------------------------------------------------- #
def test_normalize_uppercases_and_strips_diacritics():
    assert _normalize("Terunofuji") == "TERUNOFUJI"
    assert _normalize("  hoSHoryu  ") == "HOSHORYU"
    assert _normalize("") == ""
    assert _normalize(None) == ""


# ---------------------------------------------------------------------- #
# Real-data exact match
# ---------------------------------------------------------------------- #
def test_exact_match_terunofuji_in_202307():
    rik, bts = _load_real()
    aligner = BoutAligner(rik, bts)
    rid = aligner.match_winner("Terunofuji", "202307")
    assert rid == 45  # Terunofuji Haruo (sumo-api id)


def test_exact_match_case_insensitive():
    rik, bts = _load_real()
    aligner = BoutAligner(rik, bts)
    assert aligner.match_winner("TERUNOFUJI", "202307") == 45
    assert aligner.match_winner("terunofuji", "202307") == 45


# ---------------------------------------------------------------------- #
# Fuzzy match within cutoff
# ---------------------------------------------------------------------- #
def test_fuzzy_match_within_cutoff():
    rik, bts = _load_real()
    aligner = BoutAligner(rik, bts)
    # one-char OCR typo
    rid = aligner.match_winner("Terunofuy", "202307", cutoff=0.80)
    assert rid == 45


def test_below_cutoff_returns_none():
    rik, bts = _load_real()
    aligner = BoutAligner(rik, bts)
    assert aligner.match_winner("Foobarbaz", "202307", cutoff=0.85) is None


def test_empty_name_returns_none():
    rik, bts = _load_real()
    aligner = BoutAligner(rik, bts)
    assert aligner.match_winner("", "202307") is None


# ---------------------------------------------------------------------- #
# match_segment_to_bout
# ---------------------------------------------------------------------- #
def test_match_segment_returns_bout_dict():
    rik, bts = _load_real()
    aligner = BoutAligner(rik, bts)
    seg = {"winner_name": "Terunofuji", "score": "1-0"}
    m = aligner.match_segment_to_bout(seg, "202307")
    assert m is not None
    d = m.to_dict()
    assert d["bashoId"] == "202307"
    assert d["winnerId"] == 45
    # east or west id should equal winner
    assert d["winnerId"] in (d["eastId"], d["westId"])
    # y_east is consistent
    assert d["y_east"] == int(d["winnerId"] == d["eastId"])


def test_match_segment_unknown_winner_returns_none():
    rik, bts = _load_real()
    aligner = BoutAligner(rik, bts)
    assert (
        aligner.match_segment_to_bout({"winner_name": "Foobar"}, "202307") is None
    )
    assert aligner.match_segment_to_bout({"winner_name": None}, "202307") is None


# ---------------------------------------------------------------------- #
# Multi-bout disambiguation -- N-th appearance maps to N-th win in order
# ---------------------------------------------------------------------- #
def test_disambig_picks_chronological_win():
    rik, bts = _load_real()
    aligner = BoutAligner(rik, bts)
    # Use the per-basho winnerId with the most wins
    g = bts[bts["bashoId"] == "202307"]
    winners = g.groupby("winnerId").size().sort_values(ascending=False)
    top_rid = int(winners.index[0])
    # find their shikona
    rows = g[g["winnerId"] == top_rid]
    east = rows[rows["eastId"] == top_rid]["eastShikona"]
    west = rows[rows["westId"] == top_rid]["westShikona"]
    shik = (east.tolist() + west.tolist())[0]
    wins = aligner.candidate_bouts(top_rid, "202307")
    n = len(wins)
    assert n >= 2

    prior: dict[int, int] = {}
    seg = {"winner_name": shik}
    m1 = aligner.match_segment_to_bout(seg, "202307", prior_uses=prior)
    m2 = aligner.match_segment_to_bout(seg, "202307", prior_uses=prior)
    m3 = aligner.match_segment_to_bout(seg, "202307", prior_uses=prior)
    assert m1.day == int(wins.iloc[0]["day"])
    assert m1.matchNo == int(wins.iloc[0]["matchNo"])
    assert m2.day == int(wins.iloc[1]["day"])
    assert m2.matchNo == int(wins.iloc[1]["matchNo"])
    # Different bouts unless rikishi only has one win (we asserted n>=2)
    assert (m1.day, m1.matchNo) != (m2.day, m2.matchNo)


def test_align_segments_attaches_alignment_field():
    rik, bts = _load_real()
    aligner = BoutAligner(rik, bts)
    segs = [
        {"winner_name": "Terunofuji", "score": "1-0", "t_start": 0.0, "t_end": 4.0},
        {"winner_name": None, "t_start": 4.0, "t_end": 8.0},
        {"winner_name": "Foobar", "t_start": 8.0, "t_end": 12.0},
    ]
    out = aligner.align_segments(segs, "202307")
    assert out[0]["alignment"] is not None
    assert out[0]["alignment"]["winnerId"] == 45
    assert out[1]["alignment"] is None
    assert out[2]["alignment"] is None
