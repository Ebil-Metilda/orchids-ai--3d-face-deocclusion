from __future__ import annotations

import torch


def smoothness_regularization(mask: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
    dy = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
    return dx + dy
