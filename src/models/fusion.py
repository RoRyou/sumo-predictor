"""Two-tower fusion model — combines Route A (structural) + Route B (pose).

Mirrors readme §6.4.  Three components:

* :class:`StructTower` — learnable projection of raw structural features
  (``rank_diff``, winrate diffs, h2h, etc.) into ``struct_dim``.
  In practice the structural side is dominated by the gradient-boosted
  ensemble from :mod:`src.training.train_struct`; this tower's job is to
  let the *deep* fusion model see the structural signal as an embedding.
  Concretely we pass in **both** the raw features AND the stacked
  ensemble's predicted probability (as a logit), so the tower can lean on
  the ensemble's confidence while still capturing residual nonlinearity.
* :class:`SumoFusionModel` — the §6.4 fusion MLP.  Wraps a frozen-or-fine-
  tuned :class:`~src.models.temporal.PoseTower` on the B side and the
  struct tower on the A side; outputs ``P(east wins)``.

Shapes
------
* struct in : ``(B, struct_in_dim)`` (raw + stacked prob/logit)
* pose in   : ``(B, T, pose_feat_dim)``
* mask      : ``(B, T)`` optional
* output    : ``(B,)`` probabilities

This module focuses on the architecture.  Training data plumbing lives
in :mod:`src.training.train_fusion`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.temporal import PoseTower


class StructTower(nn.Module):
    """Small MLP that maps raw structural features → ``embed_dim`` embedding.

    Designed to be combined with the GBDT ensemble's stacked logit — pass
    that logit as one of the input features (it's typically the strongest
    single signal).
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class SumoFusionModel(nn.Module):
    """Two-tower fusion model: structural + pose → P(east wins).

    Parameters
    ----------
    struct_in_dim:
        Number of raw structural input features (including stacked GBDT
        logit and any auxiliary scalars).
    pose_feat_dim:
        Per-frame pose feature dimensionality (default 40, matching
        :mod:`src.features.kinematics`).
    pose_tower:
        Optional pre-built :class:`~src.models.temporal.PoseTower`.  If
        None, one is constructed with default settings.
    struct_embed:
        Hidden width of the struct tower output.
    pose_embed:
        Hidden width of the pose tower output (must match the supplied
        ``pose_tower`` if you pass one).
    hidden:
        Fusion-MLP hidden size.
    dropout:
        Fusion-MLP dropout.

    Notes
    -----
    * The pose tower can be frozen for stage-1 fusion training (set
      ``model.freeze_pose_tower(True)``); typically a brief fine-tune
      stage 2 follows with all weights unfrozen.
    * Output is a probability via sigmoid for parity with readme §6.4.
      For numerical stability during training, prefer to drive the model
      with raw logits and apply :class:`~nn.BCEWithLogitsLoss` — use
      :meth:`forward_logits` for that path.
    """

    def __init__(
        self,
        struct_in_dim: int,
        pose_feat_dim: int = 40,
        pose_tower: PoseTower | None = None,
        struct_embed: int = 32,
        pose_embed: int = 128,
        hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.struct_tower = StructTower(
            in_dim=struct_in_dim,
            embed_dim=struct_embed,
            dropout=dropout,
        )
        self.pose_tower = pose_tower or PoseTower(
            feat_dim=pose_feat_dim,
            embed_dim=pose_embed,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(struct_embed + pose_embed, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    # ------------------------------------------------------------------ #
    def freeze_pose_tower(self, frozen: bool = True) -> None:
        for p in self.pose_tower.parameters():
            p.requires_grad_(not frozen)

    def freeze_struct_tower(self, frozen: bool = True) -> None:
        for p in self.struct_tower.parameters():
            p.requires_grad_(not frozen)

    # ------------------------------------------------------------------ #
    def forward_logits(
        self,
        struct_x: torch.Tensor,
        pose_x: torch.Tensor,
        pose_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return raw logits (use for ``BCEWithLogitsLoss``)."""
        s = self.struct_tower(struct_x)
        p = self.pose_tower(pose_x, mask=pose_mask)
        z = self.fusion(torch.cat([s, p], dim=-1))
        return z.squeeze(-1)

    def forward(
        self,
        struct_x: torch.Tensor,
        pose_x: torch.Tensor,
        pose_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``P(east wins)``."""
        return torch.sigmoid(
            self.forward_logits(struct_x, pose_x, pose_mask=pose_mask)
        )
