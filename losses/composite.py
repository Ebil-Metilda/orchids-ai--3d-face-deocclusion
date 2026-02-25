from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .identity import IdentityLoss
from .perceptual import VGGPerceptualLoss
from .regularization import smoothness_regularization


@dataclass
class LossWeights:
    lambda_l1: float
    lambda_perceptual: float
    lambda_identity: float
    lambda_adv: float
    lambda_smooth: float


class CompositeGeneratorLoss:
    def __init__(self, weights: LossWeights, device: torch.device) -> None:
        self.weights = weights
        self.perceptual = VGGPerceptualLoss().to(device)
        self.identity = IdentityLoss().to(device)

    def __call__(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        refined_mask: torch.Tensor,
        pred_fake: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l1 = F.l1_loss(output, target)

        perceptual = output.new_tensor(0.0)
        if self.weights.lambda_perceptual > 0.0:
            perceptual = self.perceptual(output, target)

        identity = output.new_tensor(0.0)
        if self.weights.lambda_identity > 0.0:
            identity = self.identity(output, target)

        adv = output.new_tensor(0.0)
        if self.weights.lambda_adv > 0.0:
            if pred_fake is None:
                raise ValueError("pred_fake is required when adversarial loss is enabled")
            adv = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))

        smooth = smoothness_regularization(refined_mask)

        total = (
            self.weights.lambda_l1 * l1
            + self.weights.lambda_perceptual * perceptual
            + self.weights.lambda_identity * identity
            + self.weights.lambda_adv * adv
            + self.weights.lambda_smooth * smooth
        )

        details = {
            "l1": float(l1.detach().item()),
            "perceptual": float(perceptual.detach().item()),
            "identity": float(identity.detach().item()),
            "adversarial": float(adv.detach().item()),
            "smoothness": float(smooth.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, details


def discriminator_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
    real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
    fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
    return 0.5 * (real + fake)
