"""Multi-capacity Siamese bagging: train 3 architectures × 5 seeds each = 15 models.

Different capacities (small/medium/large) bring different overfitting trajectories,
yielding diversified errors. Bag them all.
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


class Siamese(nn.Module):
    """Siamese with configurable size."""
    def __init__(self, n_side: int, n_pair: int, n_rikishi: int,
                 hidden: int, side_out: int, pair_out: int,
                 emb_id: int, dropout: float):
        super().__init__()
        self.feat_tower = nn.Sequential(
            nn.Linear(n_side, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, side_out), nn.LayerNorm(side_out), nn.GELU(), nn.Dropout(dropout * 0.75),
        )
        self.id_emb = nn.Embedding(n_rikishi + 1, emb_id)
        self.pair_tower = nn.Sequential(
            nn.Linear(n_pair, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, pair_out), nn.LayerNorm(pair_out), nn.GELU(),
        )
        side_dim = side_out + emb_id
        head_in = 4 * side_dim + pair_out
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _encode(self, x_feat, x_id):
        return torch.cat([self.feat_tower(x_feat), self.id_emb(x_id)], dim=-1)

    def forward(self, x_a, x_b, id_a, id_b, x_pair):
        e_a = self._encode(x_a, id_a)
        e_b = self._encode(x_b, id_b)
        pair = self.pair_tower(x_pair)
        z = torch.cat([e_a, e_b, (e_a - e_b).abs(), e_a * e_b, pair], dim=-1)
        return self.head(z).squeeze(-1)


def train_one(model_cfg, train, val, test, seed, lr, wd, epochs, batch, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Siamese(**model_cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    x_a_tr, x_b_tr, id_a_tr, id_b_tr, x_p_tr, y_tr, sw_tr = train
    x_a_va, x_b_va, id_a_va, id_b_va, x_p_va, y_va, _ = val
    x_a_te, x_b_te, id_a_te, id_b_te, x_p_te, y_te, _ = test
    n_tr = len(y_tr)

    best_auc = 0
    best_state = None
    patience = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits = model(x_a_tr[idx], x_b_tr[idx], id_a_tr[idx], id_b_tr[idx], x_p_tr[idx])
            loss = (loss_fn(logits, y_tr[idx]) * sw_tr[idx]).mean()
            loss.backward()
            opt.step()
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
        if patience >= 7:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(model(x_a_va, x_b_va, id_a_va, id_b_va, x_p_va)).cpu().numpy()
        p_te = torch.sigmoid(model(x_a_te, x_b_te, id_a_te, id_b_te, x_p_te)).cpu().numpy()
    return p_va, p_te, best_auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_v4.parquet")
    p.add_argument("--val-basho", default="202311")
    p.add_argument("--test-start", default="202401")
    p.add_argument("--seeds-per", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--out", default="runs/siamese_multi_probs.npz")
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

    cfgs = [
        # (name, kwargs, lr, wd)
        ("small",  dict(hidden=64, side_out=32, pair_out=24, emb_id=12, dropout=0.15), 1e-3, 1e-4),
        ("medium", dict(hidden=96, side_out=48, pair_out=32, emb_id=16, dropout=0.20), 1e-3, 1e-4),
        ("large",  dict(hidden=128, side_out=64, pair_out=48, emb_id=24, dropout=0.25), 8e-4, 2e-4),
    ]

    val_probs, test_probs = [], []
    for name, kw, lr, wd in cfgs:
        for s in range(args.seeds_per):
            cfg = dict(n_side=len(side_cols), n_pair=len(pair_cols), n_rikishi=n_rikishi, **kw)
            pv, pt, auc = train_one(cfg, train, val, test, seed=s,
                                     lr=lr, wd=wd, epochs=args.epochs, batch=256, device=device)
            print(f"  {name} seed {s}: val_auc={auc:.4f} test_acc={accuracy_score(test[-2].cpu().numpy(), pt>0.5):.4f}")
            val_probs.append(pv); test_probs.append(pt)

    v = np.mean(val_probs, axis=0); t = np.mean(test_probs, axis=0)
    y_va_np = val[-2].cpu().numpy(); y_te_np = test[-2].cpu().numpy()
    print(f"\nMulti-capacity bag (15 models): val_acc={accuracy_score(y_va_np, v>0.5):.4f} val_auc={roc_auc_score(y_va_np, v):.4f}")
    print(f"                                  test_acc={accuracy_score(y_te_np, t>0.5):.4f} test_auc={roc_auc_score(y_te_np, t):.4f}")
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(v, y_va_np)
    v_iso = iso.transform(v); t_iso = iso.transform(t)
    print(f"  iso val_acc={accuracy_score(y_va_np, v_iso>0.5):.4f} val_auc={roc_auc_score(y_va_np, v_iso):.4f}")
    print(f"  iso test_acc={accuracy_score(y_te_np, t_iso>0.5):.4f} test_auc={roc_auc_score(y_te_np, t_iso):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, val=v, test=t, val_iso=v_iso, test_iso=t_iso,
             y_val=y_va_np, y_test=y_te_np)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
