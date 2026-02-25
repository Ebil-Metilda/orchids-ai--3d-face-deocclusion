from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def psnr(pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    mse = torch.clamp(mse, min=1e-8)
    return 20.0 * torch.log10(torch.tensor(max_value, device=pred.device)) - 10.0 * torch.log10(mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    score = numerator / (denominator + 1e-8)
    return score.mean()


def identity_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    emb_a = F.normalize(emb_a, dim=1)
    emb_b = F.normalize(emb_b, dim=1)
    return (emb_a * emb_b).sum(dim=1).mean()
