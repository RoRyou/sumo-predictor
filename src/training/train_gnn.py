"""GraphSAGE on rikishi h2h graph for sumo bout outcome prediction.

Graph construction:
  * Nodes: 207 rikishi appearing in our bouts.
  * Edges: every (winner → loser) directed edge from training bouts only
    (no test/val leakage).
  * Node features: static (height, weight, age_at_train_cutoff, heya one-hot,
    shusshin region one-hot) + dynamic (career stats snapshot at train cutoff).

Training:
  * 2-layer GraphSAGE encoder produces 64-dim node embeddings.
  * Edge-prediction head: MLP on concat(emb_A, emb_B, |emb_A - emb_B|).
  * Loss: binary cross-entropy on bout outcome (1 = east wins).
  * Symmetric augmentation: each bout also seen flipped (B is east, A is west).
  * Train on bouts < val_basho; calibrate isotonic on val_basho; eval on test.

Evaluated standalone AND as a new bag/model stream to merge into SOTA v3.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Data cleaning
# ---------------------------------------------------------------------- #
def clean_bouts(bouts: pd.DataFrame) -> pd.DataFrame:
    """Filter out fusen (no-shows) and self/zero bouts."""
    b = bouts.copy()
    b["bashoId"] = b["bashoId"].astype(str)
    n0 = len(b)
    b = b[(b["eastId"] != 0) & (b["westId"] != 0)]
    b = b[b["eastId"] != b["westId"]]
    b = b[b["winnerId"].isin(b["eastId"]) | b["winnerId"].isin(b["westId"])]
    # Reject fusen (default win, not a real bout)
    b = b[b["kimarite"].str.lower() != "fusen"]
    b = b.sort_values(["bashoId", "day", "matchNo"]).reset_index(drop=True)
    logger.info("Cleaned bouts: %d → %d (-%d)", n0, len(b), n0 - len(b))
    return b


def build_rikishi_features(
    rikishis: pd.DataFrame, bout_ids: set[int]
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Filter to rikishi present in bouts; assign sequential node-ids.

    Returns the filtered features DataFrame and a {rikishi_id: node_idx} map.
    """
    r = rikishis[rikishis["id"].isin(bout_ids)].copy()
    r = r.sort_values("id").reset_index(drop=True)
    # Parse birthDate → age in years (relative to a fixed ref date)
    r["birthDate"] = pd.to_datetime(r["birthDate"], errors="coerce", utc=True)
    ref = pd.Timestamp("2025-01-01", tz="UTC")
    r["age_years"] = (ref - r["birthDate"]).dt.days / 365.25
    r["age_years"] = r["age_years"].fillna(r["age_years"].median())

    # Heya: rare-cat threshold of 5 rikishi, else OTHER
    heya_counts = r["heya"].value_counts()
    common_heya = heya_counts[heya_counts >= 3].index
    r["heya_norm"] = r["heya"].where(r["heya"].isin(common_heya), other="OTHER")

    # Shusshin: take the country/prefecture prefix
    def shusshin_region(s):
        if pd.isna(s):
            return "UNK"
        return str(s).split(",")[0].strip()
    r["shusshin_region"] = r["shusshin"].apply(shusshin_region)
    sh_counts = r["shusshin_region"].value_counts()
    common_sh = sh_counts[sh_counts >= 3].index
    r["shusshin_norm"] = r["shusshin_region"].where(
        r["shusshin_region"].isin(common_sh), other="OTHER"
    )

    node_map = {rid: i for i, rid in enumerate(r["id"].tolist())}
    return r, node_map


def make_node_features(r: pd.DataFrame) -> torch.Tensor:
    """Build [N, F] node-feature tensor.

    Numeric: height, weight, age_years, log(career_bouts_train_cutoff).
    One-hot: heya_norm, shusshin_norm.
    """
    # Numeric
    num_cols = ["height", "weight", "age_years"]
    X_num = r[num_cols].astype(float).values
    # standardise
    X_num = (X_num - X_num.mean(axis=0, keepdims=True)) / (X_num.std(axis=0, keepdims=True) + 1e-6)
    # One-hot
    heya_oh = pd.get_dummies(r["heya_norm"], prefix="h").astype(float).values
    sh_oh = pd.get_dummies(r["shusshin_norm"], prefix="s").astype(float).values
    X = np.concatenate([X_num, heya_oh, sh_oh], axis=1)
    return torch.tensor(X, dtype=torch.float32)


# ---------------------------------------------------------------------- #
# Graph construction
# ---------------------------------------------------------------------- #
def build_edge_index(
    bouts: pd.DataFrame, node_map: dict[int, int], cutoff_basho: str
) -> torch.Tensor:
    """Directed edges winner→loser for bouts < cutoff_basho.

    Returns LongTensor [2, E].
    """
    train_bouts = bouts[bouts["bashoId"] < cutoff_basho]
    edges = []
    for r in train_bouts.itertuples(index=False):
        w = int(r.winnerId)
        l = int(r.eastId) if r.winnerId == r.westId else int(r.westId)
        if w in node_map and l in node_map:
            edges.append((node_map[w], node_map[l]))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_bout_pairs(
    bouts: pd.DataFrame,
    node_map: dict[int, int],
    bashos: list[str] | None = None,
    basho_lo: str | None = None,
    basho_hi: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pairs [N, 2], labels [N]) where label=1 iff east wins."""
    b = bouts.copy()
    if bashos is not None:
        b = b[b["bashoId"].isin(bashos)]
    if basho_lo is not None:
        b = b[b["bashoId"] >= basho_lo]
    if basho_hi is not None:
        b = b[b["bashoId"] < basho_hi]
    pairs = []
    labels = []
    for r in b.itertuples(index=False):
        e, w = int(r.eastId), int(r.westId)
        if e not in node_map or w not in node_map:
            continue
        pairs.append((node_map[e], node_map[w]))
        labels.append(int(r.winnerId == e))
    return (
        torch.tensor(pairs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


# ---------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------- #
class SumoGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_nodes: int,
        hidden_dim: int = 64,
        emb_dim: int = 64,
        id_emb_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Learnable per-node ID embedding (each rikishi gets its own latent)
        self.id_emb = nn.Embedding(num_nodes, id_emb_dim)
        nn.init.normal_(self.id_emb.weight, std=0.05)

        self.input_proj = nn.Linear(in_dim + id_emb_dim, hidden_dim)
        self.convs = nn.ModuleList()
        d_prev = hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(d_prev, hidden_dim))
            d_prev = hidden_dim
        self.convs.append(SAGEConv(d_prev, emb_dim))
        self.dropout = dropout
        # Edge predictor: concat(eA, eB, |eA-eB|, eA*eB) → MLP
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        node_ids = torch.arange(num_nodes, device=x.device)
        id_e = self.id_emb(node_ids)
        h = torch.cat([x, id_e], dim=1)
        h = F.relu(self.input_proj(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def predict(
        self, emb: torch.Tensor, pairs: torch.Tensor
    ) -> torch.Tensor:
        eA = emb[pairs[:, 0]]
        eB = emb[pairs[:, 1]]
        h = torch.cat([eA, eB, (eA - eB).abs(), eA * eB], dim=1)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------- #
# Train / eval
# ---------------------------------------------------------------------- #
def train_eval(
    bouts: pd.DataFrame,
    rikishis: pd.DataFrame,
    features: pd.DataFrame,
    val_basho: str = "202311",
    test_start: str = "202401",
    test_end: str = "202412",
    train_start: str = "201501",
    hidden_dim: int = 64,
    emb_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    patience: int = 20,
    symmetric_aug: bool = True,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    bouts = clean_bouts(bouts)
    bouts = bouts[(bouts["bashoId"] >= train_start) & (bouts["bashoId"] <= test_end)].reset_index(drop=True)
    logger.info("After basho filter [%s, %s]: %d bouts", train_start, test_end, len(bouts))
    bout_ids = set(bouts["eastId"].astype(int).tolist() + bouts["westId"].astype(int).tolist())
    r, node_map = build_rikishi_features(rikishis, bout_ids)
    logger.info("Nodes: %d", len(node_map))

    X = make_node_features(r).to(device)
    in_dim = X.shape[1]
    logger.info("Node feature dim: %d", in_dim)

    edge_index = build_edge_index(bouts, node_map, cutoff_basho=val_basho).to(device)
    logger.info("Train edges: %d", edge_index.shape[1])

    # Train pairs from bouts < val_basho
    train_pairs, train_y = build_bout_pairs(
        bouts, node_map, basho_hi=val_basho
    )
    # Val/test pairs ALIGN to features.parquet rows (same order, length)
    feats = features.copy()
    feats["bashoId"] = feats["bashoId"].astype(str)
    val_feats = feats[feats["bashoId"] == val_basho].reset_index(drop=True)
    test_feats = feats[feats["bashoId"] >= test_start].reset_index(drop=True)

    def build_pairs_from_feats(df):
        pairs, ys, mask = [], [], []
        for r in df.itertuples(index=False):
            e, w = int(r.eastId), int(r.westId)
            if e in node_map and w in node_map:
                pairs.append((node_map[e], node_map[w]))
                ys.append(int(r.y))
                mask.append(True)
            else:
                pairs.append((0, 0))  # placeholder; will use mask
                ys.append(int(r.y))
                mask.append(False)
        return (torch.tensor(pairs, dtype=torch.long),
                torch.tensor(ys, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.bool))

    val_pairs, val_y, val_mask = build_pairs_from_feats(val_feats)
    test_pairs, test_y, test_mask = build_pairs_from_feats(test_feats)
    logger.info("Val pairs: %d (valid %d), Test pairs: %d (valid %d)",
                len(val_y), int(val_mask.sum()), len(test_y), int(test_mask.sum()))
    train_pairs, train_y = train_pairs.to(device), train_y.to(device)
    val_pairs, val_y = val_pairs.to(device), val_y.to(device)
    test_pairs, test_y = test_pairs.to(device), test_y.to(device)
    val_mask, test_mask = val_mask.to(device), test_mask.to(device)

    logger.info("Train pairs: %d, val: %d, test: %d",
                len(train_y), len(val_y), len(test_y))

    if symmetric_aug:
        # Add flipped versions: (B, A) with label = 1 - y
        flipped = train_pairs[:, [1, 0]]
        train_pairs_aug = torch.cat([train_pairs, flipped], dim=0)
        train_y_aug = torch.cat([train_y, 1 - train_y], dim=0)
    else:
        train_pairs_aug, train_y_aug = train_pairs, train_y

    model = SumoGNN(
        in_dim, num_nodes=len(node_map),
        hidden_dim=hidden_dim, emb_dim=emb_dim,
        num_layers=num_layers, dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    bad = 0

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        emb = model.encode(X, edge_index)
        logits = model.predict(emb, train_pairs_aug)
        loss = F.binary_cross_entropy_with_logits(logits, train_y_aug)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            emb = model.encode(X, edge_index)
            val_logits = model.predict(emb, val_pairs)
            val_prob = torch.sigmoid(val_logits)
            # Use 0.5 fallback for masked-out (rikishi-not-in-graph) bouts
            val_prob_eff = torch.where(val_mask, val_prob, torch.tensor(0.5, device=device))
            v_acc = accuracy_score(val_y.cpu(), (val_prob_eff > 0.5).cpu())
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                logger.info("Early stop at epoch %d (best %d val_acc=%.4f)",
                            ep, best_epoch, best_val_acc)
                break
        if ep % 10 == 0:
            logger.info("ep=%d loss=%.4f val_acc=%.4f", ep, loss.item(), v_acc)

    # Load best
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb = model.encode(X, edge_index)
        val_prob_t = torch.sigmoid(model.predict(emb, val_pairs))
        test_prob_t = torch.sigmoid(model.predict(emb, test_pairs))
        val_prob = torch.where(val_mask, val_prob_t, torch.tensor(0.5, device=device)).cpu().numpy()
        test_prob = torch.where(test_mask, test_prob_t, torch.tensor(0.5, device=device)).cpu().numpy()

    val_y_np = val_y.cpu().numpy()
    test_y_np = test_y.cpu().numpy()

    # Isotonic calibration
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_prob, val_y_np)
    val_prob_iso = iso.transform(val_prob)
    test_prob_iso = iso.transform(test_prob)

    metrics = {
        "best_epoch": best_epoch,
        "val_acc_raw": float(accuracy_score(val_y_np, val_prob > 0.5)),
        "val_acc_iso": float(accuracy_score(val_y_np, val_prob_iso > 0.5)),
        "test_acc_raw": float(accuracy_score(test_y_np, test_prob > 0.5)),
        "test_acc_iso": float(accuracy_score(test_y_np, test_prob_iso > 0.5)),
        "test_ll_raw": float(log_loss(test_y_np, np.clip(test_prob, 1e-6, 1 - 1e-6))),
        "test_ll_iso": float(log_loss(test_y_np, np.clip(test_prob_iso, 1e-6, 1 - 1e-6))),
        "test_auc": float(roc_auc_score(test_y_np, test_prob)),
    }

    return {
        "metrics": metrics,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "val_prob_iso": val_prob_iso,
        "test_prob_iso": test_prob_iso,
        "val_y": val_y_np,
        "test_y": test_y_np,
    }


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _setup_logging(v: int) -> None:
    logging.basicConfig(
        level=logging.INFO if v else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


def cmd_run(args: argparse.Namespace) -> int:
    bouts = pd.read_parquet(args.bouts)
    rikishis = pd.read_parquet(args.rikishis)
    features = pd.read_parquet(args.features)

    # Bag-of-N: train multiple seeds, average
    results = []
    for s in args.seeds:
        logger.info("=== seed %d ===", s)
        r = train_eval(
            bouts, rikishis, features,
            val_basho=args.val_basho,
            test_start=args.test_start,
            test_end=args.test_end,
            train_start=args.train_start,
            hidden_dim=args.hidden_dim,
            emb_dim=args.emb_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            seed=s,
        )
        results.append(r)
        logger.info("  metrics: %s", r["metrics"])

    val_prob = np.mean([r["val_prob"] for r in results], axis=0)
    test_prob = np.mean([r["test_prob"] for r in results], axis=0)
    val_prob_iso = np.mean([r["val_prob_iso"] for r in results], axis=0)
    test_prob_iso = np.mean([r["test_prob_iso"] for r in results], axis=0)
    val_y = results[0]["val_y"]
    test_y = results[0]["test_y"]

    # Re-calibrate the averaged predictions
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_prob, val_y)
    val_prob_iso2 = iso.transform(val_prob)
    test_prob_iso2 = iso.transform(test_prob)

    agg_metrics = {
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "bagged_raw_val_acc": float(accuracy_score(val_y, val_prob > 0.5)),
        "bagged_raw_test_acc": float(accuracy_score(test_y, test_prob > 0.5)),
        "bagged_raw_test_ll": float(log_loss(test_y, np.clip(test_prob, 1e-6, 1 - 1e-6))),
        "bagged_iso_val_acc": float(accuracy_score(val_y, val_prob_iso2 > 0.5)),
        "bagged_iso_test_acc": float(accuracy_score(test_y, test_prob_iso2 > 0.5)),
        "bagged_iso_test_ll": float(log_loss(test_y, np.clip(test_prob_iso2, 1e-6, 1 - 1e-6))),
        "bagged_test_auc": float(roc_auc_score(test_y, test_prob)),
        "per_seed": [r["metrics"] for r in results],
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(agg_metrics, indent=2))
    np.savez(
        out / "probs.npz",
        val=val_prob, test=test_prob,
        val_iso=val_prob_iso2, test_iso=test_prob_iso2,
        y_val=val_y, y_test=test_y,
    )
    print(json.dumps(agg_metrics, indent=2))
    logger.info("Saved %s", out / "metrics.json")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GraphSAGE on rikishi h2h graph")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bouts", default="data/raw/bouts.parquet")
    r.add_argument("--rikishis", default="data/raw/rikishis.parquet")
    r.add_argument("--features", default="data/processed/features.parquet")
    r.add_argument("--val-basho", default="202311")
    r.add_argument("--test-start", default="202401")
    r.add_argument("--test-end", default="202412")
    r.add_argument("--train-start", default="201501")
    r.add_argument("--hidden-dim", type=int, default=64)
    r.add_argument("--emb-dim", type=int, default=64)
    r.add_argument("--num-layers", type=int, default=2)
    r.add_argument("--dropout", type=float, default=0.3)
    r.add_argument("--lr", type=float, default=5e-3)
    r.add_argument("--weight-decay", type=float, default=1e-4)
    r.add_argument("--epochs", type=int, default=200)
    r.add_argument("--patience", type=int, default=20)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    r.add_argument("--out-dir", default="runs/gnn_v1")
    r.add_argument("-v", "--verbose", action="count", default=1)
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
