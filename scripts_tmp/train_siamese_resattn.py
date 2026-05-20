"""Siamese with residual attention blocks (inspired by ALTARN + SiamRAKPN + ResNet).

Each per-side tower:
  - feature → residual MLP block (Linear → LN → GELU → Linear → +residual)
  - Squeeze-and-Excitation channel attention
  - Stack 2 such blocks
  - Final projection to d_side

The shared tower is applied to both sides (siamese symmetry).
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
from sklearn.metrics import accuracy_score, roc_auc_score
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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention. Output ⊙ input."""
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ResMLPBlock(nn.Module):
    """Residual MLP block: x → LN → Linear → GELU → Dropout → Linear → +x → SE."""
    def __init__(self, dim, dropout=0.20):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.fc2 = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock(dim, reduction=4)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.ln(x)
        h = self.fc2(self.dropout(self.act(self.fc1(h))))
        h = x + h  # residual
        h = self.se(h)  # channel attention
        return h


class SiameseResAttn(nn.Module):
    def __init__(self, n_side, n_pair, n_rikishi, d_side=64, d_id=16, n_blocks=2, dropout=0.20):
        super().__init__()
        # Per-side stem
        self.side_proj = nn.Linear(n_side, d_side)
        self.side_blocks = nn.ModuleList([ResMLPBlock(d_side, dropout) for _ in range(n_blocks)])
        self.side_norm = nn.LayerNorm(d_side)

        # Rikishi ID embedding
        self.id_emb = nn.Embedding(n_rikishi + 1, d_id)

        # Pair tower with residual blocks
        self.pair_proj = nn.Linear(n_pair, d_side)
        self.pair_blocks = nn.ModuleList([ResMLPBlock(d_side, dropout) for _ in range(1)])
        self.pair_norm = nn.LayerNorm(d_side)
        self.pair_out = nn.Linear(d_side, 32)

        # Head
        side_dim = d_side + d_id
        head_in = 4 * side_dim + 32
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _encode(self, x_feat, x_id):
        h = self.side_proj(x_feat)
        for blk in self.side_blocks:
            h = blk(h)
        h = self.side_norm(h)
        return torch.cat([h, self.id_emb(x_id)], dim=-1)

    def _encode_pair(self, x_pair):
        h = self.pair_proj(x_pair)
        for blk in self.pair_blocks:
            h = blk(h)
        return self.pair_out(self.pair_norm(h))

    def forward(self, x_a, x_b, id_a, id_b, x_pair):
        e_a = self._encode(x_a, id_a)
        e_b = self._encode(x_b, id_b)
        pair = self._encode_pair(x_pair)
        z = torch.cat([e_a, e_b, (e_a - e_b).abs(), e_a * e_b, pair], dim=-1)
        return self.head(z).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_v4.parquet")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--out", default="runs/siamese_resattn_probs.npz")
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    df["bashoId"] = df["bashoId"].astype(str)
    train_df = df[df["bashoId"] < "202311"].copy()
    val_df = df[df["bashoId"] == "202311"].copy()
    test_df = df[df["bashoId"] >= "202401"].copy()

    side_cols = [c for c in SIDE_A_COLS if c in df.columns]
    side_b_cols = [c.replace("_A", "_B") for c in side_cols]
    pair_cols = [c for c in PAIR_COLS if c in df.columns]
    ids = sorted(set(df["eastId"]) | set(df["westId"]))
    id_map = {int(r): i + 1 for i, r in enumerate(ids)}
    n_rikishi = len(id_map)
    print(f"  side={len(side_cols)} pair={len(pair_cols)} rikishi={n_rikishi} n_blocks={args.n_blocks}")

    sc_a = StandardScaler().fit(train_df[side_cols].fillna(0))
    sc_p = StandardScaler().fit(train_df[pair_cols].fillna(0))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def to_t(d):
        x_a = torch.tensor(sc_a.transform(d[side_cols].fillna(0)), dtype=torch.float32).to(device)
        x_b = torch.tensor(sc_a.transform(d[side_b_cols].fillna(0).rename(
            columns=dict(zip(side_b_cols, side_cols)))), dtype=torch.float32).to(device)
        x_p = torch.tensor(sc_p.transform(d[pair_cols].fillna(0)), dtype=torch.float32).to(device)
        id_a = torch.tensor([id_map[int(x)] for x in d["eastId"]], dtype=torch.long).to(device)
        id_b = torch.tensor([id_map[int(x)] for x in d["westId"]], dtype=torch.long).to(device)
        y = torch.tensor(d["y"].astype(int).values, dtype=torch.float32).to(device)
        sw = torch.tensor(d["sample_weight"].values, dtype=torch.float32).to(device)
        return x_a, x_b, id_a, id_b, x_p, y, sw

    train = to_t(train_df); val = to_t(val_df); test = to_t(test_df)

    val_probs, test_probs = [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model = SiameseResAttn(n_side=len(side_cols), n_pair=len(pair_cols), n_rikishi=n_rikishi,
                               n_blocks=args.n_blocks).to(device)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        x_a_tr, x_b_tr, id_a_tr, id_b_tr, x_p_tr, y_tr, sw_tr = train
        x_a_va, x_b_va, id_a_va, id_b_va, x_p_va, y_va, _ = val
        x_a_te, x_b_te, id_a_te, id_b_te, x_p_te, y_te, _ = test
        n_tr = len(y_tr)

        best_auc = 0; best_state = None; patience = 0
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n_tr, device=device)
            for i in range(0, n_tr, args.batch):
                idx = perm[i:i + args.batch]
                opt.zero_grad()
                logits = model(x_a_tr[idx], x_b_tr[idx], id_a_tr[idx], id_b_tr[idx], x_p_tr[idx])
                loss = (loss_fn(logits, y_tr[idx]) * sw_tr[idx]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                p_va = torch.sigmoid(model(x_a_va, x_b_va, id_a_va, id_b_va, x_p_va)).cpu().numpy()
            auc = roc_auc_score(y_va.cpu().numpy(), p_va)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 10:
                break

        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(x_a_va, x_b_va, id_a_va, id_b_va, x_p_va)).cpu().numpy()
            p_te = torch.sigmoid(model(x_a_te, x_b_te, id_a_te, id_b_te, x_p_te)).cpu().numpy()
        print(f"  seed {seed}: val_auc={best_auc:.4f}  test_acc={accuracy_score(y_te.cpu().numpy(), p_te>0.5):.4f}")
        val_probs.append(p_va); test_probs.append(p_te)

    v = np.mean(val_probs, axis=0); t = np.mean(test_probs, axis=0)
    y_va_np = val[-2].cpu().numpy(); y_te_np = test[-2].cpu().numpy()
    print(f"\nBag-{args.seeds}: val_acc={accuracy_score(y_va_np, v>0.5):.4f} val_auc={roc_auc_score(y_va_np, v):.4f}")
    print(f"            test_acc={accuracy_score(y_te_np, t>0.5):.4f} test_auc={roc_auc_score(y_te_np, t):.4f}")
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(v, y_va_np)
    v_iso = iso.transform(v); t_iso = iso.transform(t)
    print(f"  iso val_acc={accuracy_score(y_va_np, v_iso>0.5):.4f} val_auc={roc_auc_score(y_va_np, v_iso):.4f}")
    print(f"  iso test_acc={accuracy_score(y_te_np, t_iso>0.5):.4f} test_auc={roc_auc_score(y_te_np, t_iso):.4f}")

    np.savez(args.out, val=v, test=t, val_iso=v_iso, test_iso=t_iso,
             y_val=y_va_np, y_test=y_te_np)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
