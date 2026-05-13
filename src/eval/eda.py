"""Quick EDA report for the raw sumo data.

Emits a markdown report summarising:
    - bout volume, year/basho coverage
    - east-win rate (the trivial baseline)
    - kimarite distribution
    - rikishi profile completeness (height/weight)
    - banzuke rank coverage

CLI::

    python -m src.eval.eda run --raw-dir data/raw --out reports/01_eda.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def render(raw_dir: Path) -> str:
    bouts_p = raw_dir / "bouts.parquet"
    rik_p = raw_dir / "rikishis.parquet"
    banzuke_p = raw_dir / "banzuke.parquet"
    bashos_p = raw_dir / "bashos.parquet"

    lines: list[str] = ["# Sumo Data EDA Report", ""]

    if bashos_p.exists():
        bashos = pd.read_parquet(bashos_p)
        lines += [
            "## Bashos",
            f"- rows: **{len(bashos):,}**",
            f"- range: `{bashos['bashoId'].min()}` … `{bashos['bashoId'].max()}`",
            "",
        ]

    if bouts_p.exists():
        bouts = pd.read_parquet(bouts_p)
        bouts["bashoId"] = bouts["bashoId"].astype(str)
        lines += [
            "## Bouts",
            f"- rows: **{len(bouts):,}**",
            f"- basho range: `{bouts['bashoId'].min()}` … `{bouts['bashoId'].max()}`",
            f"- unique basho: {bouts['bashoId'].nunique()}",
            f"- days covered: {bouts['day'].min()} … {bouts['day'].max()}",
            "",
        ]
        # east-win baseline
        valid = bouts.dropna(subset=["winnerId"])
        valid = valid[valid["winnerId"].isin([0]) == False]  # noqa: E712
        valid = valid[(valid["eastId"] > 0) & (valid["westId"] > 0)]
        east_win = (valid["winnerId"] == valid["eastId"]).mean()
        lines += [
            f"- valid bouts (winner present, east+west ids non-zero): **{len(valid):,}**",
            f"- east-win rate (trivial baseline): **{east_win:.4f}**",
            f"- kimarite filled rate: **{(valid['kimarite'].astype(str).str.len() > 0).mean():.4f}**",
            "",
        ]
        topk = valid["kimarite"].value_counts().head(10)
        lines += ["### Top-10 kimarite", "", "| kimarite | count | share |", "|---|---:|---:|"]
        for k, c in topk.items():
            lines.append(f"| `{k}` | {c:,} | {c/len(valid):.3f} |")
        lines.append("")

        # bouts per basho
        per_basho = valid.groupby("bashoId").size()
        lines += [
            "### Bouts per basho",
            f"- min: {per_basho.min()}  max: {per_basho.max()}  median: {per_basho.median():.0f}",
            "",
        ]

    if rik_p.exists():
        rik = pd.read_parquet(rik_p)
        lines += [
            "## Rikishis",
            f"- rows: **{len(rik):,}**",
            f"- height non-null: {rik['height'].notna().sum():,} ({rik['height'].notna().mean():.3f})",
            f"- weight non-null: {rik['weight'].notna().sum():,} ({rik['weight'].notna().mean():.3f})",
            f"- birthDate non-null: {rik['birthDate'].notna().sum():,}",
            f"- unique heya: {rik['heya'].nunique()}",
            "",
        ]
        if "height" in rik.columns and rik["height"].notna().any():
            lines += [
                "### Anthropometry",
                f"- height (cm): mean={rik['height'].mean():.1f} std={rik['height'].std():.1f}",
                f"- weight (kg): mean={rik['weight'].mean():.1f} std={rik['weight'].std():.1f}",
                "",
            ]

    if banzuke_p.exists():
        bz = pd.read_parquet(banzuke_p)
        lines += [
            "## Banzuke",
            f"- rows (rikishi×basho): **{len(bz):,}**",
            f"- rankValue non-null: {bz['rankValue'].notna().sum():,} ({bz['rankValue'].notna().mean():.3f})",
            f"- unique rikishi: {bz['rikishiId'].nunique()}",
            "",
        ]
        if "rank" in bz.columns:
            top_ranks = bz["rank"].value_counts().head(8)
            lines += ["### Most common rank labels", "", "| rank | count |", "|---|---:|"]
            for r, c in top_ranks.items():
                lines.append(f"| `{r}` | {c:,} |")
            lines.append("")

    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = render(Path(args.raw_dir))
    out.write_text(report)
    print(report)
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EDA report")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--raw-dir", default="data/raw")
    r.add_argument("--out", default="reports/01_eda.md")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
