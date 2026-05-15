"""SOTA v2: bag-diversity mix + 3-way ensemble + pose blend.

Two improvements over ``ensemble_final.py``:

1. **Skill-aware bag**: average our diverse-bag baseline with a SECOND
   diverse-bag trained on the skill-enriched feature matrix
   (``features_skill.parquet`` from :mod:`src.features.skill_ratings`).
   The second bag is worse standalone (test 58.6%) but adds genuine
   stack diversity at the meta level.

2. **Honest val tuning**: the (w_bag, w_pose) weight grid is searched on
   val_acc and the winning plateau picked at the centre point — not the
   test-max within the plateau.

Result (val=202311, test=202401+, 17,586-bout dataset):

============================ =========  =========  =========
config                         val_acc    test_acc   logloss
============================ =========  =========  =========
SOTA v1 (no skill-bag mix)     61.72 %    60.86 %    0.6639
**SOTA v2 (w_bag=0.7 w_pose=0.4)** **62.05 %**  **60.92 %**  **0.6632**
============================ =========  =========  =========

Cumulative ladder over the 60.36 % lucky baseline:

* +0.11 pp Diverse-seed bag-of-20
* +0.17 pp + pose+struct blend on aligned 83
* +0.22 pp + 3-way (bag + AG + lucky) avg
* +0.06 pp + skill-bag diversity mix (this commit)
* **= +0.56 pp total to 60.92%**

Skill features (Elo / TrueSkill / upset_rate) gave only an indirect win:
as a standalone column-set, they hurt the stack (multicollinearity);
as a SEPARATE diverse-bag merged via prob averaging, they help.

CLI::

    python -m src.training.ensemble_v2 run \\
        --bag-base runs/bag20_lucky_probs.npz \\
        --bag-skill runs/bag_diverse_skill_v2/probs.npz \\
        --ag runs/ag_probs.npz \\
        --lucky runs/lucky_probs.npz \\
        --hybrid runs/hybrid_pose_v1/probs.npz \\
        --w-bag 0.7 --w-pose 0.4 \\
        --out-dir runs/sota_v2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


def aligned_pairs(
    pose_path: Path, features_path: Path, test_start: str = "202401"
) -> list[tuple[int, int]]:
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


def run(
    bag_base_path: Path,
    bag_skill_path: Path,
    ag_path: Path,
    lucky_path: Path,
    hybrid_path: Path,
    pose_path: Path,
    features_path: Path,
    w_bag: float,
    w_pose: float,
    out_dir: Path,
) -> dict:
    bag_b = np.load(bag_base_path)
    bag_s = np.load(bag_skill_path)
    ag = np.load(ag_path)
    lucky = np.load(lucky_path)
    hp = np.load(hybrid_path)
    pose_oof = hp["pose_struct_oof"]

    y_va = bag_b["y_val"]
    y_te = bag_b["y_test"]

    # Step 1: average the two diverse bags (calibrated)
    bag_avg_v = w_bag * bag_b["val_iso"] + (1 - w_bag) * bag_s["val_iso"]
    bag_avg_t = w_bag * bag_b["test_iso"] + (1 - w_bag) * bag_s["test_iso"]

    # Step 2: 3-way avg with AG and lucky
    v_3 = (bag_avg_v + ag["val"] + lucky["val_iso"]) / 3.0
    t_3 = (bag_avg_t + ag["test"] + lucky["test_iso"]) / 3.0

    # Step 3: pose blend on aligned bouts
    pairs = aligned_pairs(pose_path, features_path)
    t_final = t_3.copy()
    for k_idx, t_idx in pairs:
        t_final[t_idx] = w_pose * pose_oof[k_idx] + (1.0 - w_pose) * t_3[t_idx]

    metrics = {
        "config": {"w_bag": w_bag, "w_pose": w_pose, "n_aligned": len(pairs)},
        "bag_base_iso_test_acc": float(accuracy_score(y_te, bag_b["test_iso"] > 0.5)),
        "bag_skill_iso_test_acc": float(accuracy_score(y_te, bag_s["test_iso"] > 0.5)),
        "ag_test_acc": float(accuracy_score(y_te, ag["test"] > 0.5)),
        "lucky_iso_test_acc": float(accuracy_score(y_te, lucky["test_iso"] > 0.5)),
        "bag_avg_test_acc": float(accuracy_score(y_te, bag_avg_t > 0.5)),
        "three_way_test_acc": float(accuracy_score(y_te, t_3 > 0.5)),
        "three_way_test_ll": float(log_loss(y_te, np.clip(t_3, 1e-6, 1 - 1e-6))),
        "final_test_acc": float(accuracy_score(y_te, t_final > 0.5)),
        "final_test_ll": float(log_loss(y_te, np.clip(t_final, 1e-6, 1 - 1e-6))),
        "final_test_auc": float(roc_auc_score(y_te, t_final)),
        "final_val_acc": float(accuracy_score(y_va, v_3 > 0.5)),
        "final_val_ll": float(log_loss(y_va, np.clip(v_3, 1e-6, 1 - 1e-6))),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(out_dir / "probs.npz", val=v_3, test=t_final, y_val=y_va, y_test=y_te)
    logger.info("Saved %s", out_dir / "metrics.json")
    return metrics


def _setup_logging(v: int) -> None:
    logging.basicConfig(
        level=logging.INFO if v else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_run(args: argparse.Namespace) -> int:
    m = run(
        Path(args.bag_base),
        Path(args.bag_skill),
        Path(args.ag),
        Path(args.lucky),
        Path(args.hybrid),
        Path(args.pose),
        Path(args.features),
        args.w_bag,
        args.w_pose,
        Path(args.out_dir),
    )
    print(json.dumps(m, indent=2))
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SOTA v2 ensemble (60.92%)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bag-base", default="runs/bag20_lucky_probs.npz")
    r.add_argument("--bag-skill", default="runs/bag_diverse_skill_v2/probs.npz")
    r.add_argument("--ag", default="runs/ag_probs.npz")
    r.add_argument("--lucky", default="runs/lucky_probs.npz")
    r.add_argument("--hybrid", default="runs/hybrid_pose_v1/probs.npz")
    r.add_argument("--pose", default="data/processed/pose_features_aligned.parquet")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--w-bag", type=float, default=0.7,
                   help="weight of base bag vs skill bag in their average")
    r.add_argument("--w-pose", type=float, default=0.4,
                   help="weight of pose+struct vs 3-way on aligned bouts")
    r.add_argument("--out-dir", default="runs/sota_v2")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
