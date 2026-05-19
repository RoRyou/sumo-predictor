"""FT-Transformer (mini) for tabular bout prediction.

Each numerical feature becomes a token (linear projection to d_model),
then self-attention layers process them, with [CLS] token for classification.
Categorical IDs (rikishi) get their own learnable embeddings as tokens too.

Trained on bashoId < 202311, val=202311 (early stopping by val_AUC), test=202401+.
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


# All v4 features (no per-side split — let attention figure it out)
def get_feature_cols(df):
    DROP = {'bashoId','day','matchNo','eastId','westId','y','kimarite','sample_weight',
            'heya_A','heya_B','shusshin_A','shusshin_B'}
    return [c for c in df.columns if c not in DROP and df[c].dtype != 'object']


class FTTransformerMini(nn.Module):
    def __init__(self, n_num: int, n_rikishi: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        # Numerical feature token: linear + learnable bias per-feature
        self.num_proj = nn.Linear(1, d_model)
        self.feat_pos = nn.Parameter(torch.randn(n_num, d_model) * 0.02)
        # Rikishi ID embedding (two tokens: east, west)
        self.id_emb = nn.Embedding(n_rikishi + 1, d_model)
        # CLS token
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.n_num = n_num

    def forward(self, x_num, id_e, id_w):
        B = x_num.size(0)
        # x_num: (B, n_num) → per-feature token (B, n_num, d_model)
        num_tok = self.num_proj(x_num.unsqueeze(-1)) + self.feat_pos.unsqueeze(0)
        # ID embeddings
        e_tok = self.id_emb(id_e).unsqueeze(1)
        w_tok = self.id_emb(id_w).unsqueeze(1)
        # CLS
        cls = self.cls.expand(B, -1, -1)
        # Combine: [CLS, east_id, west_id, num_tokens...]
        z = torch.cat([cls, e_tok, w_tok, num_tok], dim=1)
        z = self.encoder(z)
        return self.head(z[:, 0]).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_v4.parquet")
    p.add_argument("--val-basho", default="202311")
    p.add_argument("--test-start", default="202401")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="runs/ft_transformer_v4_probs.npz")
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    df["bashoId"] = df["bashoId"].astype(str)

    train_df = df[df["bashoId"] < args.val_basho].copy()
    val_df = df[df["bashoId"] == args.val_basho].copy()
    test_df = df[df["bashoId"] >= args.test_start].copy()
    print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    num_cols = get_feature_cols(df)
    print(f"  numerical features: {len(num_cols)}")

    # Rikishi ID encoding
    ids = sorted(set(df["eastId"]) | set(df["westId"]))
    id_map = {int(r): i + 1 for i, r in enumerate(ids)}
    n_rikishi = len(id_map)
    print(f"  rikishi: {n_rikishi}")

    sc = StandardScaler().fit(train_df[num_cols].fillna(0))

    def to_tensors(d, device):
        x = torch.tensor(sc.transform(d[num_cols].fillna(0)), dtype=torch.float32).to(device)
        id_e = torch.tensor([id_map[int(x)] for x in d["eastId"]], dtype=torch.long).to(device)
        id_w = torch.tensor([id_map[int(x)] for x in d["westId"]], dtype=torch.long).to(device)
        y = torch.tensor(d["y"].astype(int).values, dtype=torch.float32).to(device)
        sw = torch.tensor(d["sample_weight"].values, dtype=torch.float32).to(device)
        return x, id_e, id_w, y, sw

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  device: {device}")
    train = to_tensors(train_df, device)
    val = to_tensors(val_df, device)
    test = to_tensors(test_df, device)

    val_probs_seeds, test_probs_seeds = [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = FTTransformerMini(
            n_num=len(num_cols), n_rikishi=n_rikishi,
            d_model=args.d_model, n_heads=4, n_layers=args.n_layers,
        ).to(device)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        x_tr, ie_tr, iw_tr, y_tr, sw_tr = train
        x_va, ie_va, iw_va, y_va, _ = val
        x_te, ie_te, iw_te, y_te, _ = test
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
                logits = model(x_tr[idx], ie_tr[idx], iw_tr[idx])
                loss = (loss_fn(logits, y_tr[idx]) * sw_tr[idx]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item() * len(idx)
            ep_loss /= n_tr
            sched.step()

            model.eval()
            with torch.no_grad():
                p_va = torch.sigmoid(model(x_va, ie_va, iw_va)).cpu().numpy()
            val_auc = roc_auc_score(y_va.cpu().numpy(), p_va)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 8:
                break
            if epoch % 5 == 0:
                print(f"  seed {seed} epoch {epoch:3d}  loss={ep_loss:.4f}  val_auc={val_auc:.4f}  best={best_val_auc:.4f}")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(x_va, ie_va, iw_va)).cpu().numpy()
            p_te = torch.sigmoid(model(x_te, ie_te, iw_te)).cpu().numpy()
        print(f"  seed {seed} best_val_auc={best_val_auc:.4f}  test_acc={accuracy_score(y_te.cpu().numpy(), p_te>0.5):.4f}")
        val_probs_seeds.append(p_va)
        test_probs_seeds.append(p_te)

    v = np.mean(val_probs_seeds, axis=0)
    t = np.mean(test_probs_seeds, axis=0)
    y_va_np = val[-2].cpu().numpy()
    y_te_np = test[-2].cpu().numpy()
    print(f"\nBagged ({args.seeds} seeds):")
    print(f"  val_acc={accuracy_score(y_va_np, v>0.5):.4f}  val_auc={roc_auc_score(y_va_np, v):.4f}")
    print(f"  test_acc={accuracy_score(y_te_np, t>0.5):.4f}  test_auc={roc_auc_score(y_te_np, t):.4f}")

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
