from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VGGPerceptualLoss(nn.Module):
    """Compact perceptual proxy without external pretrained downloads."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feat = self.features(pred)
        target_feat = self.features(target)
        return F.l1_loss(pred_feat, target_feat)
