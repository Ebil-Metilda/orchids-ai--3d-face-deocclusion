from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

from .occlusion import apply_synthetic_occlusion


def _pil_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


class FaceOcclusionDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int,
        synthetic_count: int = 0,
        use_synthetic_if_missing: bool = True,
        transform: Optional[object] = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform

        self.paths = []
        if self.root.exists():
            self.paths = [
                p
                for p in self.root.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]

        self.synthetic_count = synthetic_count if (use_synthetic_if_missing and not self.paths) else 0
        self.length = len(self.paths) if self.paths else self.synthetic_count

        if self.length == 0:
            raise ValueError(
                f"No images found in {root} and synthetic generation disabled or zero synthetic samples."
            )

    def __len__(self) -> int:
        return self.length

    def _make_synthetic_face(self) -> Image.Image:
        size = self.image_size
        base = Image.new("RGB", (size, size), (235, 210, 190))
        draw = ImageDraw.Draw(base)

        draw.ellipse([size * 0.2, size * 0.1, size * 0.8, size * 0.9], fill=(245, 220, 200))
        eye_y = int(size * 0.42)
        eye_dx = int(size * 0.13)
        eye_r = int(size * 0.03)
        cx = size // 2
        draw.ellipse([cx - eye_dx - eye_r, eye_y - eye_r, cx - eye_dx + eye_r, eye_y + eye_r], fill=(30, 30, 30))
        draw.ellipse([cx + eye_dx - eye_r, eye_y - eye_r, cx + eye_dx + eye_r, eye_y + eye_r], fill=(30, 30, 30))

        mouth_y = int(size * 0.68)
        draw.arc([size * 0.38, mouth_y - 10, size * 0.62, mouth_y + 20], start=10, end=170, fill=(140, 50, 50), width=2)
        return base

    def _load_image(self, index: int) -> Image.Image:
        if self.paths:
            return Image.open(self.paths[index]).convert("RGB")
        return self._make_synthetic_face()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self._load_image(index)
        clean = _pil_to_tensor(image, self.image_size)
        occluded, mask = apply_synthetic_occlusion(clean)
        return occluded, clean, mask
