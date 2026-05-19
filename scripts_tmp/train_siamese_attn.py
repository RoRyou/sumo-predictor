"""Deeper Siamese network with cross-attention between two rikishi towers.

Architecture:
- Per-side feature tower (deeper, 96 -> 64 -> 48)
- Learnable rikishi ID embedding (32-dim)
- Cross-attention: e_A attends to e_B and vice-versa
- Pair feature tower (96 -> 32)
- Final head on [e_A, e_B, |e_A-e_B|, e_A*e_B, attn_A, attn_B, pair_emb]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


SIDE_A_COLS = [
    "winrate_A_10", "winrate_A_30", "winrate_A_90",
    "pushing_ratio_A", "belt_ratio_A",
    "h2h_count",
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


class CrossAttnSiamese(nn.Module):
    def __init__(self, n_side: int, n_pair: int, n_rikishi: int,
                 d_model: int = 64, dropout: float = 0.20):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Linear(n_side, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(96, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )
        self.id_emb = nn.Embedding(n_rikishi + 1, 32)
        self.id_proj = nn.Linear(32, d_model)
        # Cross attention: A as query, B as key/value (and vice-versa)
        self.attn_a = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.attn_b = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        # Pair
        self.pair_tower = nn.Sequential(
            nn.Linear(n_pair, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(96, 32), nn.LayerNorm(32), nn.GELU(),
        )
        head_in = 6 * d_model + 32  # e_a, e_b, attn_a, attn_b, |Δ|, ⊙, pair
        self.head = nn.Sequential(
            nn.Linear(head_in, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(96, 32), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

    def _encode(self, x_feat, x_id):
        return self.tower(x_feat) + self.id_proj(self.id_emb(x_id))

    def forward(self, x_a, x_b, id_a, id_b, x_pair):
        e_a = self._encode(x_a, id_a)  # (B, d)
        e_b = self._encode(x_b, id_b)
        # Cross attention (single-token queries)
        e_a_seq = e_a.unsqueeze(1)
        e_b_seq = e_b.unsqueeze(1)
        attn_a, _ = self.attn_a(e_a_seq, e_b_seq, e_b_seq)
        attn_b, _ = self.attn_b(e_b_seq, e_a_seq, e_a_seq)
        attn_a = self.norm_a(attn_a.squeeze(1) + e_a)
        attn_b = self.norm_b(attn_b.squeeze(1) + e_b)
        pair = self.pair_tower(x_pair)
        z = torch.cat([e_a, e_b, attn_a, attn_b, (e_a - e_b).abs(), e_a * e_b, pair], dim=-1)
        return self.head(z).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_v4.parquet")
    p.add_argument("--val-basho", default="202311")
    p.add_argument("--test-start", default="202401")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default="runs/siamese_attn_probs.npz")
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    df["bashoId"] = df["bashoId"].astype(str)
    train_df = df[df["bashoId"] < args.val_basho].copy()
    val_df = df[df["bashoId"] == args.val_basho].copy()
    test_df = df[df["bashoId"] >= args.test_start].copy()

    side_cols = [c for c in SIDE_A_COLS if c in df.columns]
    side_b_cols = [c.replace("_A", "_B") for c in side_cols]
    pair_cols = [c for c in PAIR_COLS if c in df.columns]
    ids = sorted(set(df["eastId"]) | set(df["westId"]))
    id_map = {int(r): i + 1 for i, r in enumerate(ids)}
    n_rikishi = len(id_map)
    print(f"  side={len(side_cols)} pair={len(pair_cols)} rikishi={n_rikishi}")

    sc_a = StandardScaler().fit(train_df[side_cols].fillna(0))
    sc_p = StandardScaler().fit(train_df[pair_cols].fillna(0))

    def to_tensors(d, device):
        x_a = torch.tensor(sc_a.transform(d[side_cols].fillna(0)), dtype=torch.float32).to(device)
        x_b = torch.tensor(sc_a.transform(d[side_b_cols].fillna(0).rename(
            columns=dict(zip(side_b_cols, side_cols)))), dtype=torch.float32).to(device)
        x_p = torch.tensor(sc_p.transform(d[pair_cols].fillna(0)), dtype=torch.float32).to(device)
        id_a = torch.tensor([id_map[int(x)] for x in d["eastId"]], dtype=torch.long).to(device)
        id_b = torch.tensor([id_map[int(x)] for x in d["westId"]], dtype=torch.long).to(device)
        y = torch.tensor(d["y"].astype(int).values, dtype=torch.float32).to(device)
        sw = torch.tensor(d["sample_weight"].values, dtype=torch.float32).to(device)
        return x_a, x_b, id_a, id_b, x_p, y, sw

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  device: {device}")
    train = to_tensors(train_df, device)
    val = to_tensors(val_df, device)
    test = to_tensors(test_df, device)

    val_probs, test_probs = [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = CrossAttnSiamese(
            n_side=len(side_cols), n_pair=len(pair_cols), n_rikishi=n_rikishi,
            d_model=args.d_model, dropout=args.dropout,
        ).to(device)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        x_a_tr, x_b_tr, id_a_tr, id_b_tr, x_p_tr, y_tr, sw_tr = train
        x_a_va, x_b_va, id_a_va, id_b_va, x_p_va, y_va, _ = val
        x_a_te, x_b_te, id_a_te, id_b_te, x_p_te, y_te, _ = test
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
                logits = model(x_a_tr[idx], x_b_tr[idx], id_a_tr[idx], id_b_tr[idx], x_p_tr[idx])
                loss = (loss_fn(logits, y_tr[idx]) * sw_tr[idx]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item() * len(idx)
            sched.step()

            model.eval()
            with torch.no_grad():
                p_va = torch.sigmoid(model(x_a_va, x_b_va, id_a_va, id_b_va, x_p_va)).cpu().numpy()
            val_auc = roc_auc_score(y_va.cpu().numpy(), p_va)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 10:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(x_a_va, x_b_va, id_a_va, id_b_va, x_p_va)).cpu().numpy()
            p_te = torch.sigmoid(model(x_a_te, x_b_te, id_a_te, id_b_te, x_p_te)).cpu().numpy()
        print(f"  seed {seed} val_auc={best_val_auc:.4f}  test_acc={accuracy_score(y_te.cpu().numpy(), p_te>0.5):.4f}")
        val_probs.append(p_va)
        test_probs.append(p_te)

    v = np.mean(val_probs, axis=0)
    t = np.mean(test_probs, axis=0)
    y_va_np = val[-2].cpu().numpy()
    y_te_np = test[-2].cpu().numpy()
    print(f"\nBag-{args.seeds}: val_acc={accuracy_score(y_va_np, v>0.5):.4f} val_auc={roc_auc_score(y_va_np, v):.4f}")
    print(f"           test_acc={accuracy_score(y_te_np, t>0.5):.4f} test_auc={roc_auc_score(y_te_np, t):.4f}")

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(v, y_va_np)
    v_iso = iso.transform(v)
    t_iso = iso.transform(t)
    print(f"  iso val_acc={accuracy_score(y_va_np, v_iso>0.5):.4f} val_auc={roc_auc_score(y_va_np, v_iso):.4f}")
    print(f"  iso test_acc={accuracy_score(y_te_np, t_iso>0.5):.4f} test_auc={roc_auc_score(y_te_np, t_iso):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, val=v, test=t, val_iso=v_iso, test_iso=t_iso,
             y_val=y_va_np, y_test=y_te_np)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
