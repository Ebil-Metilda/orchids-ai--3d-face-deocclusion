from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, down: bool = True) -> None:
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DualBranchGenerator(nn.Module):
    def __init__(self, in_channels: int = 7) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64, down=True)
        self.enc2 = ConvBlock(64, 128, down=True)
        self.enc3 = ConvBlock(128, 256, down=True)

        self.dec1 = ConvBlock(256, 128, down=False)
        self.dec2 = ConvBlock(128, 64, down=False)

        self.mask_head = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid())
        self.face_head = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True), nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, occluded: torch.Tensor, prior: torch.Tensor, coarse_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([occluded, prior, coarse_mask], dim=1)
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        d1 = self.dec1(h3)
        d2 = self.dec2(d1)

        refined_mask = self.mask_head(d2)
        generated_face = self.face_head(d2)
        output = occluded * (1.0 - refined_mask) + generated_face * refined_mask
        return output, refined_mask, generated_face
