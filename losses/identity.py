from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class IdentityLoss(nn.Module):
    """Lightweight identity embedding loss for self-contained experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        for param in self.embedder.parameters():
            param.requires_grad = False
        self.embedder.eval()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.embedder(x)
        return feat.flatten(1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_emb = self.embed(pred)
        target_emb = self.embed(target)
        pred_emb = F.normalize(pred_emb, dim=1)
        target_emb = F.normalize(target_emb, dim=1)
        return 1.0 - (pred_emb * target_emb).sum(dim=1).mean()
