from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AblationVariant:
    name: str
    use_prior: bool
    use_identity_loss: bool
    use_adversarial_loss: bool


VARIANT_ORDER = ["full", "no_prior", "no_identity", "no_adversarial"]

ABLATION_VARIANTS: dict[str, AblationVariant] = {
    "full": AblationVariant(
        name="full",
        use_prior=True,
        use_identity_loss=True,
        use_adversarial_loss=True,
    ),
    "no_prior": AblationVariant(
        name="no_prior",
        use_prior=False,
        use_identity_loss=True,
        use_adversarial_loss=True,
    ),
    "no_identity": AblationVariant(
        name="no_identity",
        use_prior=True,
        use_identity_loss=False,
        use_adversarial_loss=True,
    ),
    "no_adversarial": AblationVariant(
        name="no_adversarial",
        use_prior=True,
        use_identity_loss=True,
        use_adversarial_loss=False,
    ),
}


def get_ablation_variant(name: str) -> AblationVariant:
    if name not in ABLATION_VARIANTS:
        available = ", ".join(sorted(ABLATION_VARIANTS.keys()))
        raise ValueError(f"Unknown ablation variant '{name}'. Available: {available}")
    return ABLATION_VARIANTS[name]


def get_variant_index(name: str) -> int:
    if name not in VARIANT_ORDER:
        available = ", ".join(VARIANT_ORDER)
        raise ValueError(f"Unknown ablation variant '{name}'. Available: {available}")
    return VARIANT_ORDER.index(name)


def resolve_variants(config_variants: list[str] | None) -> list[str]:
    if not config_variants:
        return VARIANT_ORDER.copy()
    resolved: list[str] = []
    for name in config_variants:
        resolved.append(get_ablation_variant(str(name)).name)
    return resolved


def compute_prior(
    occluded: torch.Tensor,
    reconstructor: torch.nn.Module | None,
    use_prior: bool,
) -> torch.Tensor:
    if use_prior:
        if reconstructor is None:
            raise ValueError("Reconstructor must be provided when use_prior=True")
        return reconstructor(occluded)

    # No-prior ablation uses a weak 2D blur baseline instead of learned 3D prior.
    return F.avg_pool2d(occluded, kernel_size=11, stride=1, padding=5)
