"""Pose tower for Route B: Bi-LSTM × 2 + Attention Pooling.

Mirrors readme §6.3.  Adds:

* a confidence-aware mask path in :class:`AttentionPooling` (readme T12)
* a thin :class:`PoseTowerClassifier` head for standalone binary
  training on bout outcomes (later replaced by the fusion MLP).

Shapes
------
* input  : ``(B, T, feat_dim)`` -- typically ``feat_dim=40``
* output : ``(B, embed_dim)`` from :class:`PoseTower`,
            ``(B,)`` probabilities from :class:`PoseTowerClassifier`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Confidence-weighted soft attention over the time axis.

    Parameters
    ----------
    dim
        Hidden size of the input sequence.

    Inputs
    ------
    h : (B, T, dim)
    mask : (B, T) bool, optional
        ``False`` positions are zeroed out before softmax (used for
        padded sequences and -- per readme T12 -- low-confidence frames).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1)
        )

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        scores = self.attn(h).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=-1)        # (B, T)
        pooled = torch.bmm(w.unsqueeze(1), h).squeeze(1)  # (B, dim)
        return pooled


class PoseTower(nn.Module):
    """Bi-LSTM × 2 + Attention Pooling.  Outputs ``f_pose ∈ R^128``."""

    def __init__(
        self,
        feat_dim: int = 40,
        hidden: int = 128,
        num_layers: int = 2,
        embed_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.input_norm = nn.LayerNorm(feat_dim)
        self.lstm = nn.LSTM(
            feat_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_pool = AttentionPooling(hidden * 2)
        self.proj = nn.Linear(hidden * 2, embed_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (B, T, feat_dim)
        x = self.input_norm(x)
        h, _ = self.lstm(x)                    # (B, T, 2H)
        pooled = self.attn_pool(h, mask=mask)  # (B, 2H)
        return self.proj(pooled)               # (B, embed_dim)


class PoseTowerClassifier(nn.Module):
    """``PoseTower`` + small binary head for standalone Route-B training."""

    def __init__(
        self,
        feat_dim: int = 40,
        hidden: int = 128,
        embed_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.tower = PoseTower(
            feat_dim=feat_dim,
            hidden=hidden,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        emb = self.tower(x, mask=mask)         # (B, embed_dim)
        return self.head(emb).squeeze(-1)      # (B,) logits
