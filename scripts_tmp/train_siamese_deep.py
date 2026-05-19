"""Siamese deep model for sumo bout prediction.

Architecture:
1. Per-rikishi features (winrate_A, height_A, pushing_ratio_A, etc.) are encoded
   by a shared MLP tower → 32-dim rikishi embedding `e_A` and `e_B`.
2. Pair features (rank_diff, h2h_*, etc.) go through their own tower.
3. Rikishi ID embeddings (learnable per-rikishi, 16-dim) added to towers.
4. Final classifier on [e_A, e_B, |e_A-e_B|, e_A*e_B, pair_features].

Strict no-leak: trained on bashoId < 202311, val=202311 (early stopping),
test=202401+ (held out for final eval).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from category_encoders import TargetEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


SIDE_A_COLS = [
    "winrate_A_10", "winrate_A_30", "winrate_A_90",
    "pushing_ratio_A", "belt_ratio_A",
    "h2h_count",  # symmetric, fine to include in both
    "streak_A", "record_w_A", "record_l_A",
    "bouts_this_basho_A", "days_since_last_A",
    "career_winrate_A", "career_bouts_A",
    "elo_A", "ts_mu_A", "ts_sigma_A", "ts_skill_A", "upset_rate_A", "bouts_seen_A",
    "prev_kachikoshi_A", "prev_makekoshi_A", "kachi_gap_A", "make_gap_A", "kachi_pressure_A",
    "rank_velocity_A",
]
SIDE_B_COLS = [c.replace("_A", "_B") for c in SIDE_A_COLS]

PAIR_COLS = [
    "rank_diff", "height_diff", "weight_diff", "bmi_diff", "age_diff",
    "winrate_diff_10", "winrate_diff_30", "winrate_diff_90",
    "h2h_winrate", "style_compat", "day_of_basho",
    "elo_diff", "elo_expected_A", "ts_mu_diff", "ts_skill_diff", "upset_rate_diff",
    "rank_velocity_diff", "prev_wins_diff", "prev_kachikoshi_diff",
    "days_since_last_diff", "kachi_pressure_diff",
]


class SiameseDeep(nn.Module):
    def __init__(self, n_side: int, n_pair: int, n_rikishi: int,
                 emb_id: int = 16, hidden: int = 96):
        super().__init__()
        # Per-rikishi feature tower (shared between A and B)
        self.feat_tower = nn.Sequential(
            nn.Linear(n_side, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(hidden, 48), nn.LayerNorm(48), nn.GELU(), nn.Dropout(0.15),
        )
        # Learnable rikishi ID embedding
        self.id_emb = nn.Embedding(n_rikishi + 1, emb_id)
        # Pair feature tower
        self.pair_tower = nn.Sequential(
            nn.Linear(n_pair, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(hidden, 32), nn.LayerNorm(32), nn.GELU(),
        )
        # Final classifier
        # Input: [e_A | e_B | |e_A-e_B| | e_A*e_B | pair_emb]
        # Each rikishi side: 48 (feat) + 16 (id) = 64
        side_dim = 48 + emb_id
        head_in = 4 * side_dim + 32
        self.head = nn.Sequential(
            nn.Linear(head_in, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def _encode_side(self, x_feat, x_id):
        return torch.cat([self.feat_tower(x_feat), self.id_emb(x_id)], dim=-1)

    def forward(self, x_a, x_b, id_a, id_b, x_pair):
        e_a = self._encode_side(x_a, id_a)
        e_b = self._encode_side(x_b, id_b)
        pair = self.pair_tower(x_pair)
        z = torch.cat([e_a, e_b, (e_a - e_b).abs(), e_a * e_b, pair], dim=-1)
        return self.head(z).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_v4.parquet")
    p.add_argument("--val-basho", default="202311")
    p.add_argument("--test-start", default="202401")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="runs/siamese_deep_probs.npz")
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    df["bashoId"] = df["bashoId"].astype(str)

    train_df = df[df["bashoId"] < args.val_basho].copy()
    val_df = df[df["bashoId"] == args.val_basho].copy()
    test_df = df[df["bashoId"] >= args.test_start].copy()
    print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    # Build features
    side_cols_avail = [c for c in SIDE_A_COLS if c in df.columns]
    side_b_cols_avail = [c.replace("_A", "_B") for c in side_cols_avail]
    pair_cols_avail = [c for c in PAIR_COLS if c in df.columns]
    print(f"  side cols: {len(side_cols_avail)}, pair cols: {len(pair_cols_avail)}")

    # Rikishi ID encoding: map unique IDs to 0..N-1
    ids = pd.concat([df["eastId"], df["westId"]]).unique()
    id_map = {int(rid): i + 1 for i, rid in enumerate(sorted(ids))}  # 0 reserved
    n_rikishi = len(id_map)
    print(f"  rikishi: {n_rikishi}")

    def to_id_tensor(df, col):
        return torch.tensor([id_map[int(x)] for x in df[col]], dtype=torch.long)

    # Scale features fit on train
    sc_a = StandardScaler().fit(train_df[side_cols_avail].fillna(0))
    sc_p = StandardScaler().fit(train_df[pair_cols_avail].fillna(0))

    def to_tensors(d):
        x_a = torch.tensor(sc_a.transform(d[side_cols_avail].fillna(0)), dtype=torch.float32)
        x_b = torch.tensor(sc_a.transform(d[side_b_cols_avail].fillna(0).rename(
            columns=dict(zip(side_b_cols_avail, side_cols_avail)))), dtype=torch.float32)
        x_p = torch.tensor(sc_p.transform(d[pair_cols_avail].fillna(0)), dtype=torch.float32)
        id_a = to_id_tensor(d, "eastId")
        id_b = to_id_tensor(d, "westId")
        y = torch.tensor(d["y"].astype(int).values, dtype=torch.float32)
        sw = torch.tensor(d["sample_weight"].values, dtype=torch.float32)
        return x_a, x_b, id_a, id_b, x_p, y, sw

    train = to_tensors(train_df)
    val = to_tensors(val_df)
    test = to_tensors(test_df)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  device: {device}")

    val_probs_seeds, test_probs_seeds = [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SiameseDeep(
            n_side=len(side_cols_avail),
            n_pair=len(pair_cols_avail),
            n_rikishi=n_rikishi,
        ).to(device)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        x_a_tr, x_b_tr, id_a_tr, id_b_tr, x_p_tr, y_tr, sw_tr = [t.to(device) for t in train]
        x_a_va, x_b_va, id_a_va, id_b_va, x_p_va, y_va, _ = [t.to(device) for t in val]
        x_a_te, x_b_te, id_a_te, id_b_te, x_p_te, y_te, _ = [t.to(device) for t in test]
        n_tr = len(y_tr)

        best_val_auc = 0.0
        best_state = None
        patience = 0
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n_tr, device=device)
            ep_loss = 0.0
            for i in range(0, n_tr, args.batch):
                idx = perm[i:i + args.batch]
                opt.zero_grad()
                logits = model(x_a_tr[idx], x_b_tr[idx],
                              id_a_tr[idx], id_b_tr[idx], x_p_tr[idx])
                loss = (loss_fn(logits, y_tr[idx]) * sw_tr[idx]).mean()
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(idx)
            ep_loss /= n_tr

            model.eval()
            with torch.no_grad():
                logits_va = model(x_a_va, x_b_va, id_a_va, id_b_va, x_p_va)
                p_va = torch.sigmoid(logits_va).cpu().numpy()
            val_auc = roc_auc_score(y_va.cpu().numpy(), p_va)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 7:
                break
            if epoch % 5 == 0:
                print(f"  seed {seed} epoch {epoch:3d}  loss={ep_loss:.4f}  val_auc={val_auc:.4f}  best={best_val_auc:.4f}")

        # Load best, predict val + test
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(x_a_va, x_b_va, id_a_va, id_b_va, x_p_va)).cpu().numpy()
            p_te = torch.sigmoid(model(x_a_te, x_b_te, id_a_te, id_b_te, x_p_te)).cpu().numpy()
        print(f"  seed {seed} best_val_auc={best_val_auc:.4f}  test_acc={accuracy_score(y_te.cpu().numpy(), p_te > 0.5):.4f}")
        val_probs_seeds.append(p_va)
        test_probs_seeds.append(p_te)

    v = np.mean(val_probs_seeds, axis=0)
    t = np.mean(test_probs_seeds, axis=0)
    y_va_np = val[-2].numpy()
    y_te_np = test[-2].numpy()
    print(f"\nBagged ({args.seeds} seeds):")
    print(f"  val_acc={accuracy_score(y_va_np, v>0.5):.4f}  val_auc={roc_auc_score(y_va_np, v):.4f}  val_ll={log_loss(y_va_np, np.clip(v,1e-6,1-1e-6)):.4f}")
    print(f"  test_acc={accuracy_score(y_te_np, t>0.5):.4f}  test_auc={roc_auc_score(y_te_np, t):.4f}  test_ll={log_loss(y_te_np, np.clip(t,1e-6,1-1e-6)):.4f}")

    # Isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(v, y_va_np)
    v_iso = iso.transform(v)
    t_iso = iso.transform(t)
    print(f"  iso val_acc={accuracy_score(y_va_np, v_iso>0.5):.4f}  iso val_auc={roc_auc_score(y_va_np, v_iso):.4f}")
    print(f"  iso test_acc={accuracy_score(y_te_np, t_iso>0.5):.4f}  iso test_auc={roc_auc_score(y_te_np, t_iso):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, val=v, test=t, val_iso=v_iso, test_iso=t_iso,
             y_val=y_va_np, y_test=y_te_np)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
