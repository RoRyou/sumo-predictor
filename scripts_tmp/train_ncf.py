"""Neural Collaborative Filtering: pure rikishi-ID model.

No features. Just rikishi IDs and outcomes.
- Each rikishi gets two embeddings: 'offensive' (e_off) and 'defensive' (e_def)
- P(east wins) = sigmoid(<e_off_east, e_def_west> - <e_off_west, e_def_east> + bias)

This is the matrix factorization view of rikishi vs rikishi:
- e_off_east · e_def_west = how good east attacks west's defense
- e_off_west · e_def_east = how good west attacks east's defense
- Diff = relative attack advantage = log-odds of east winning

Bias term captures global east-side advantage.

Why this is FRESH:
- All existing models use hand-crafted features (rank_diff, winrate, Elo)
- NCF learns rikishi-pair-specific signals from raw outcomes
- Should capture "rikishi A has style that troubles rikishi B" patterns
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


class NCF(nn.Module):
    def __init__(self, n_rikishi, d=16, dropout=0.10):
        super().__init__()
        self.off_emb = nn.Embedding(n_rikishi + 1, d)
        self.def_emb = nn.Embedding(n_rikishi + 1, d)
        self.bias = nn.Parameter(torch.zeros(1))
        # Initialize small
        nn.init.normal_(self.off_emb.weight, std=0.05)
        nn.init.normal_(self.def_emb.weight, std=0.05)
        self.drop = nn.Dropout(dropout)

    def forward(self, id_e, id_w):
        e_off = self.drop(self.off_emb(id_e))
        e_def = self.drop(self.def_emb(id_e))
        w_off = self.drop(self.off_emb(id_w))
        w_def = self.drop(self.def_emb(id_w))
        # east attacks west = e_off · w_def
        e_attack = (e_off * w_def).sum(-1)
        w_attack = (w_off * e_def).sum(-1)
        return e_attack - w_attack + self.bias


class NCF_Deep(nn.Module):
    """NCF with non-linear interaction (MLP on concat instead of dot product)."""
    def __init__(self, n_rikishi, d=24, dropout=0.20):
        super().__init__()
        self.emb = nn.Embedding(n_rikishi + 1, d)
        nn.init.normal_(self.emb.weight, std=0.05)
        # MLP on [e_east, e_west, e_east * e_west, e_east - e_west]
        self.head = nn.Sequential(
            nn.Linear(4 * d, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, id_e, id_w):
        e = self.emb(id_e)
        w = self.emb(id_w)
        x = torch.cat([e, w, e * w, e - w], dim=-1)
        return self.head(x).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_v4.parquet")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--arch", choices=["dot", "deep"], default="dot")
    p.add_argument("--out", default="runs/ncf_probs.npz")
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    df["bashoId"] = df["bashoId"].astype(str)
    train_df = df[df["bashoId"] < "202311"].copy()
    val_df = df[df["bashoId"] == "202311"].copy()
    test_df = df[df["bashoId"] >= "202401"].copy()

    ids = sorted(set(df["eastId"]) | set(df["westId"]))
    id_map = {int(r): i + 1 for i, r in enumerate(ids)}
    n_rikishi = len(id_map)
    print(f"  rikishi={n_rikishi} arch={args.arch} d={args.d}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def to_t(d):
        id_e = torch.tensor([id_map[int(x)] for x in d["eastId"]], dtype=torch.long).to(device)
        id_w = torch.tensor([id_map[int(x)] for x in d["westId"]], dtype=torch.long).to(device)
        y = torch.tensor(d["y"].astype(int).values, dtype=torch.float32).to(device)
        sw = torch.tensor(d["sample_weight"].values, dtype=torch.float32).to(device)
        return id_e, id_w, y, sw

    train = to_t(train_df); val = to_t(val_df); test = to_t(test_df)

    val_probs, test_probs = [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        if args.arch == "dot":
            model = NCF(n_rikishi, d=args.d).to(device)
        else:
            model = NCF_Deep(n_rikishi, d=args.d).to(device)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        ide_tr, idw_tr, y_tr, sw_tr = train
        ide_va, idw_va, y_va, _ = val
        ide_te, idw_te, y_te, _ = test
        n_tr = len(y_tr)

        best_auc = 0; best_state = None; patience = 0
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n_tr, device=device)
            for i in range(0, n_tr, args.batch):
                idx = perm[i:i + args.batch]
                opt.zero_grad()
                logits = model(ide_tr[idx], idw_tr[idx])
                loss = (loss_fn(logits, y_tr[idx]) * sw_tr[idx]).mean()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p_va = torch.sigmoid(model(ide_va, idw_va)).cpu().numpy()
            auc = roc_auc_score(y_va.cpu().numpy(), p_va)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 20:
                break

        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(ide_va, idw_va)).cpu().numpy()
            p_te = torch.sigmoid(model(ide_te, idw_te)).cpu().numpy()
        print(f"  seed {seed}: val_auc={best_auc:.4f}  test_acc={accuracy_score(y_te.cpu().numpy(), p_te>0.5):.4f}")
        val_probs.append(p_va); test_probs.append(p_te)

    v = np.mean(val_probs, axis=0); t = np.mean(test_probs, axis=0)
    y_va_np = val[2].cpu().numpy(); y_te_np = test[2].cpu().numpy()
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
