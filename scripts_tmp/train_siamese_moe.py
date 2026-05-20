"""Mixture-of-Experts Siamese.

Architecture:
- Shared encoder (same as plain siamese): per-side tower + rikishi ID embedding
- K=3 expert heads, each producing a logit
- Gate network on pair-context (rank_diff, day_of_basho, max_elo) → soft weights
- Final logit = Σ_k gate_k * expert_k_logit

Experts specialize implicitly:
- Without hard routing constraints, soft MoE lets each expert find its niche
- Gate learns to route close-rank bouts to one expert, etc.

Aux load-balancing loss to prevent expert collapse.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
GATE_COLS = ["rank_diff", "day_of_basho", "elo_expected_A", "h2h_winrate", "ts_mu_diff"]


class MoESiamese(nn.Module):
    def __init__(self, n_side, n_pair, n_rikishi, n_gate, n_experts=3,
                 emb_id=16, hidden=96, dropout=0.20):
        super().__init__()
        # Shared encoder
        self.feat_tower = nn.Sequential(
            nn.Linear(n_side, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 48), nn.LayerNorm(48), nn.GELU(), nn.Dropout(dropout * 0.75),
        )
        self.id_emb = nn.Embedding(n_rikishi + 1, emb_id)
        self.pair_tower = nn.Sequential(
            nn.Linear(n_pair, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 32), nn.LayerNorm(32), nn.GELU(),
        )

        side_dim = 48 + emb_id
        head_in = 4 * side_dim + 32

        # K expert heads
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, 1),
            ) for _ in range(n_experts)
        ])

        # Gate on pair-context features
        self.gate = nn.Sequential(
            nn.Linear(n_gate, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, n_experts),
        )
        self.n_experts = n_experts

    def _encode(self, x_feat, x_id):
        return torch.cat([self.feat_tower(x_feat), self.id_emb(x_id)], dim=-1)

    def forward(self, x_a, x_b, id_a, id_b, x_pair, x_gate):
        e_a = self._encode(x_a, id_a)
        e_b = self._encode(x_b, id_b)
        pair = self.pair_tower(x_pair)
        z = torch.cat([e_a, e_b, (e_a - e_b).abs(), e_a * e_b, pair], dim=-1)

        # Expert logits: (B, K)
        expert_logits = torch.stack([exp(z).squeeze(-1) for exp in self.experts], dim=-1)
        # Gate weights: (B, K)
        gate_logits = self.gate(x_gate)
        gate_weights = F.softmax(gate_logits, dim=-1)
        # Final = weighted sum
        final = (gate_weights * expert_logits).sum(-1)
        return final, gate_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_v4.parquet")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n-experts", type=int, default=3)
    p.add_argument("--lambda-balance", type=float, default=0.01,
                   help="load-balancing aux loss weight")
    p.add_argument("--out", default="runs/siamese_moe_probs.npz")
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    df["bashoId"] = df["bashoId"].astype(str)
    train_df = df[df["bashoId"] < "202311"].copy()
    val_df = df[df["bashoId"] == "202311"].copy()
    test_df = df[df["bashoId"] >= "202401"].copy()

    side_cols = [c for c in SIDE_A_COLS if c in df.columns]
    side_b_cols = [c.replace("_A", "_B") for c in side_cols]
    pair_cols = [c for c in PAIR_COLS if c in df.columns]
    gate_cols = [c for c in GATE_COLS if c in df.columns]
    ids = sorted(set(df["eastId"]) | set(df["westId"]))
    id_map = {int(r): i + 1 for i, r in enumerate(ids)}
    n_rikishi = len(id_map)
    print(f"  side={len(side_cols)} pair={len(pair_cols)} gate={len(gate_cols)} rikishi={n_rikishi} n_experts={args.n_experts}")

    sc_a = StandardScaler().fit(train_df[side_cols].fillna(0))
    sc_p = StandardScaler().fit(train_df[pair_cols].fillna(0))
    sc_g = StandardScaler().fit(train_df[gate_cols].fillna(0))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def to_t(d):
        x_a = torch.tensor(sc_a.transform(d[side_cols].fillna(0)), dtype=torch.float32).to(device)
        x_b = torch.tensor(sc_a.transform(d[side_b_cols].fillna(0).rename(
            columns=dict(zip(side_b_cols, side_cols)))), dtype=torch.float32).to(device)
        x_p = torch.tensor(sc_p.transform(d[pair_cols].fillna(0)), dtype=torch.float32).to(device)
        x_g = torch.tensor(sc_g.transform(d[gate_cols].fillna(0)), dtype=torch.float32).to(device)
        id_a = torch.tensor([id_map[int(x)] for x in d["eastId"]], dtype=torch.long).to(device)
        id_b = torch.tensor([id_map[int(x)] for x in d["westId"]], dtype=torch.long).to(device)
        y = torch.tensor(d["y"].astype(int).values, dtype=torch.float32).to(device)
        sw = torch.tensor(d["sample_weight"].values, dtype=torch.float32).to(device)
        return x_a, x_b, id_a, id_b, x_p, x_g, y, sw

    train = to_t(train_df); val = to_t(val_df); test = to_t(test_df)

    val_probs, test_probs = [], []
    gate_dist_avg = np.zeros(args.n_experts)  # avg gate distribution on test
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model = MoESiamese(n_side=len(side_cols), n_pair=len(pair_cols),
                            n_rikishi=n_rikishi, n_gate=len(gate_cols),
                            n_experts=args.n_experts).to(device)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        x_a_tr, x_b_tr, id_a_tr, id_b_tr, x_p_tr, x_g_tr, y_tr, sw_tr = train
        x_a_va, x_b_va, id_a_va, id_b_va, x_p_va, x_g_va, y_va, _ = val
        x_a_te, x_b_te, id_a_te, id_b_te, x_p_te, x_g_te, y_te, _ = test
        n_tr = len(y_tr)

        best_auc = 0; best_state = None; patience = 0
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n_tr, device=device)
            for i in range(0, n_tr, args.batch):
                idx = perm[i:i + args.batch]
                opt.zero_grad()
                logits, gate_w = model(x_a_tr[idx], x_b_tr[idx], id_a_tr[idx], id_b_tr[idx],
                                       x_p_tr[idx], x_g_tr[idx])
                main_loss = (bce(logits, y_tr[idx]) * sw_tr[idx]).mean()
                # Load-balancing: encourage gate to use all experts evenly per-batch
                gate_mean = gate_w.mean(dim=0)  # (K,)
                balance_loss = (gate_mean * gate_mean.log().clamp(min=-10)).sum() \
                    + np.log(args.n_experts)  # = -entropy + log(K), zero at uniform
                loss = main_loss + args.lambda_balance * balance_loss
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                logits_va, _ = model(x_a_va, x_b_va, id_a_va, id_b_va, x_p_va, x_g_va)
                p_va = torch.sigmoid(logits_va).cpu().numpy()
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
            logits_va, _ = model(x_a_va, x_b_va, id_a_va, id_b_va, x_p_va, x_g_va)
            p_va = torch.sigmoid(logits_va).cpu().numpy()
            logits_te, gate_te = model(x_a_te, x_b_te, id_a_te, id_b_te, x_p_te, x_g_te)
            p_te = torch.sigmoid(logits_te).cpu().numpy()
            gate_dist_avg += gate_te.cpu().numpy().mean(axis=0) / args.seeds
        print(f"  seed {seed}: val_auc={best_auc:.4f}  test_acc={accuracy_score(y_te.cpu().numpy(), p_te>0.5):.4f}")
        val_probs.append(p_va); test_probs.append(p_te)

    print(f"\nAvg gate distribution on test: {gate_dist_avg}")

    v = np.mean(val_probs, axis=0); t = np.mean(test_probs, axis=0)
    y_va_np = val[-2].cpu().numpy(); y_te_np = test[-2].cpu().numpy()
    print(f"\nBag-{args.seeds}: val_acc={accuracy_score(y_va_np, v>0.5):.4f} val_auc={roc_auc_score(y_va_np, v):.4f}")
    print(f"            test_acc={accuracy_score(y_te_np, t>0.5):.4f} test_auc={roc_auc_score(y_te_np, t):.4f}")
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(v, y_va_np)
    v_iso = iso.transform(v); t_iso = iso.transform(t)
    print(f"  iso val_acc={accuracy_score(y_va_np, v_iso>0.5):.4f} val_auc={roc_auc_score(y_va_np, v_iso):.4f}")
    print(f"  iso test_acc={accuracy_score(y_te_np, t_iso>0.5):.4f} test_auc={roc_auc_score(y_te_np, t_iso):.4f}")

    np.savez(args.out, val=v, test=t, val_iso=v_iso, test_iso=t_iso,
             y_val=y_va_np, y_test=y_te_np, gate_dist=gate_dist_avg)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
