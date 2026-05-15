"""SOTA v3 candidate: integrate more diverse bags + pose blend.

Extends ``ensemble_v2.py`` to support an arbitrary number of bag inputs.
Picks weights honestly: simplex grid search on val_acc, then for tied
configs picks the centre of the plateau (smallest L2 deviation from uniform).

If no candidate strictly improves SOTA v2 val=62.05%, we report the failure
and keep SOTA v2.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


def aligned_pairs(pose_path: Path, features_path: Path, test_start: str = "202401") -> list[tuple[int, int]]:
    pose = pd.read_parquet(pose_path)
    pose["bashoId"] = pose["bashoId"].astype(str)
    struct = pd.read_parquet(features_path)
    struct["bashoId"] = struct["bashoId"].astype(str)
    test_struct = struct[struct["bashoId"] >= test_start].reset_index(drop=True)
    test_idx = {
        (r.bashoId, int(r.day), int(r.matchNo)): i for i, r in test_struct.iterrows()
    }
    pairs: list[tuple[int, int]] = []
    for k_idx, r in pose.iterrows():
        k = (r["bashoId"], int(r["day"]), int(r["matchNo"]))
        if k in test_idx:
            pairs.append((k_idx, test_idx[k]))
    return pairs


def simplex_grid(n: int, step: float = 0.1) -> list[tuple[float, ...]]:
    """All n-tuples summing to 1.0 in increments of step."""
    k = int(round(1.0 / step))
    out: list[tuple[float, ...]] = []
    for combo in itertools.combinations_with_replacement(range(n), k):
        counts = [0] * n
        for c in combo:
            counts[c] += 1
        # All permutations of this multiset
        seen = set()
        for perm in itertools.permutations(counts):
            if perm in seen:
                continue
            seen.add(perm)
            out.append(tuple(p * step for p in perm))
    return out


def run(
    bag_paths: list[Path],
    bag_names: list[str],
    ag_path: Path,
    lucky_path: Path,
    hybrid_path: Path,
    pose_path: Path,
    features_path: Path,
    out_dir: Path,
    weight_step: float = 0.1,
) -> dict:
    bags = [np.load(p) for p in bag_paths]
    ag = np.load(ag_path)
    lucky = np.load(lucky_path)
    hp = np.load(hybrid_path)
    pose_oof = hp["pose_struct_oof"]

    y_va = bags[0]["y_val"]
    y_te = bags[0]["y_test"]
    pairs = aligned_pairs(pose_path, features_path)

    val_iso = [b["val_iso"] for b in bags]
    test_iso = [b["test_iso"] for b in bags]

    n = len(bags)

    best_val = -1.0
    best_configs: list[tuple] = []
    pose_weights = [0.3, 0.4, 0.5]

    # Individual stats
    individual = {}
    for name, b in zip(bag_names, bags):
        individual[f"{name}_val_iso_acc"] = float(accuracy_score(y_va, b["val_iso"] > 0.5))
        individual[f"{name}_test_iso_acc"] = float(accuracy_score(y_te, b["test_iso"] > 0.5))
    individual["ag_val_acc"] = float(accuracy_score(y_va, ag["val"] > 0.5))
    individual["ag_test_acc"] = float(accuracy_score(y_te, ag["test"] > 0.5))
    individual["lucky_val_iso_acc"] = float(accuracy_score(y_va, lucky["val_iso"] > 0.5))
    individual["lucky_test_iso_acc"] = float(accuracy_score(y_te, lucky["test_iso"] > 0.5))

    # Grid over bag-weight simplex
    grid = simplex_grid(n, weight_step)
    logger.info("Grid size: %d", len(grid))

    for w in grid:
        if abs(sum(w) - 1.0) > 1e-6:
            continue
        bag_v = sum(w[i] * val_iso[i] for i in range(n))
        bag_t = sum(w[i] * test_iso[i] for i in range(n))
        v_3 = (bag_v + ag["val"] + lucky["val_iso"]) / 3.0
        t_3 = (bag_t + ag["test"] + lucky["test_iso"]) / 3.0
        for wp in pose_weights:
            t_final = t_3.copy()
            for k_idx, t_idx in pairs:
                t_final[t_idx] = wp * pose_oof[k_idx] + (1 - wp) * t_3[t_idx]
            v_acc = accuracy_score(y_va, v_3 > 0.5)
            t_acc = accuracy_score(y_te, t_final > 0.5)
            t_ll = log_loss(y_te, np.clip(t_final, 1e-6, 1 - 1e-6))
            if v_acc > best_val + 1e-9:
                best_val = v_acc
                best_configs = [(w, wp, v_acc, t_acc, t_ll)]
            elif abs(v_acc - best_val) < 1e-9:
                best_configs.append((w, wp, v_acc, t_acc, t_ll))

    # Tie-breaker: pick smallest L2 deviation from uniform (centre of plateau)
    uniform = tuple([1.0 / n] * n)
    def l2(w):
        return sum((wi - u) ** 2 for wi, u in zip(w[0], uniform))
    best_configs.sort(key=lambda c: (l2(c), c[1]))
    chosen = best_configs[0]
    w_bags, w_pose, val_acc, test_acc, test_ll = chosen

    # Compute final probs at chosen config
    bag_v = sum(w_bags[i] * val_iso[i] for i in range(n))
    bag_t = sum(w_bags[i] * test_iso[i] for i in range(n))
    v_3 = (bag_v + ag["val"] + lucky["val_iso"]) / 3.0
    t_3 = (bag_t + ag["test"] + lucky["test_iso"]) / 3.0
    t_final = t_3.copy()
    for k_idx, t_idx in pairs:
        t_final[t_idx] = w_pose * pose_oof[k_idx] + (1 - w_pose) * t_3[t_idx]

    metrics = {
        "bag_names": bag_names,
        "chosen_weights": dict(zip(bag_names, [float(x) for x in w_bags])),
        "w_pose": w_pose,
        "n_aligned": len(pairs),
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "test_logloss": float(test_ll),
        "test_auc": float(roc_auc_score(y_te, t_final)),
        "n_tied_configs": len(best_configs),
        "individual": individual,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(out_dir / "probs.npz", val=v_3, test=t_final, y_val=y_va, y_test=y_te)
    logger.info("Wrote %s", out_dir / "metrics.json")
    return metrics


def _setup_logging(v: int) -> None:
    logging.basicConfig(
        level=logging.INFO if v else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_run(args: argparse.Namespace) -> int:
    bag_pairs = [b.split("=", 1) for b in args.bag]
    bag_names = [p[0] for p in bag_pairs]
    bag_paths = [Path(p[1]) for p in bag_pairs]
    m = run(
        bag_paths, bag_names,
        Path(args.ag), Path(args.lucky), Path(args.hybrid),
        Path(args.pose), Path(args.features),
        Path(args.out_dir),
        weight_step=args.step,
    )
    print(json.dumps(m, indent=2))
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SOTA v3 ensemble (n-bag)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bag", action="append", required=True,
                   help="NAME=path.npz, repeatable")
    r.add_argument("--ag", default="runs/ag_probs.npz")
    r.add_argument("--lucky", default="runs/lucky_probs.npz")
    r.add_argument("--hybrid", default="runs/hybrid_pose_v1/probs.npz")
    r.add_argument("--pose", default="data/processed/pose_features_aligned.parquet")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--step", type=float, default=0.1)
    r.add_argument("--out-dir", default="runs/sota_v3")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
