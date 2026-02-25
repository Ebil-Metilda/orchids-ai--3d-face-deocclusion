from __future__ import annotations

import random
from typing import Tuple

import torch


def _rand_bounds(size: int) -> Tuple[int, int]:
    a = random.randint(0, size - 1)
    b = random.randint(a, size)
    return a, b


def apply_synthetic_occlusion(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies random synthetic occlusions and returns occluded image and mask.

    Mask uses 1 for occluded regions and 0 for visible regions.
    """
    _, h, w = image.shape
    mask = torch.zeros((1, h, w), dtype=image.dtype)
    occluded = image.clone()

    occlusion_type = random.choice(["rectangle", "stripe", "sunglasses", "random_blocks"])

    if occlusion_type == "rectangle":
        y1, y2 = _rand_bounds(h)
        x1, x2 = _rand_bounds(w)
        if (y2 - y1) < h // 6:
            y2 = min(h, y1 + h // 4)
        if (x2 - x1) < w // 6:
            x2 = min(w, x1 + w // 4)
        mask[:, y1:y2, x1:x2] = 1.0

    elif occlusion_type == "stripe":
        y_center = random.randint(h // 3, (2 * h) // 3)
        thickness = random.randint(max(2, h // 12), max(4, h // 6))
        y1 = max(0, y_center - thickness // 2)
        y2 = min(h, y_center + thickness // 2)
        mask[:, y1:y2, :] = 1.0

    elif occlusion_type == "sunglasses":
        y1 = h // 3
        y2 = h // 2
        x1 = w // 6
        x2 = (5 * w) // 6
        mask[:, y1:y2, x1:x2] = 1.0

    else:  # random_blocks
        for _ in range(random.randint(2, 5)):
            bh = random.randint(max(4, h // 12), max(8, h // 5))
            bw = random.randint(max(4, w // 12), max(8, w // 5))
            y = random.randint(0, max(0, h - bh))
            x = random.randint(0, max(0, w - bw))
            mask[:, y : y + bh, x : x + bw] = 1.0

    noise = torch.rand_like(image) * 0.15
    fill = torch.zeros_like(image) + random.uniform(0.0, 0.3)
    occluded = image * (1.0 - mask) + (fill + noise) * mask
    occluded = torch.clamp(occluded, 0.0, 1.0)
    return occluded, mask
