from __future__ import annotations

import torch


def compute_occlusion_mask(input_image: torch.Tensor, reconstructed: torch.Tensor, threshold: float) -> torch.Tensor:
    diff = torch.abs(input_image - reconstructed)
    per_pixel = diff.mean(dim=1, keepdim=True)
    return (per_pixel > threshold).float()
