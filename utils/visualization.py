from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw
import torch


def _to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    t = torch.clamp(tensor, 0.0, 1.0)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_panel(images: Dict[str, torch.Tensor], file_path: str, max_items: int = 4) -> None:
    rows: List[List[Image.Image]] = []
    for _, tensor in images.items():
        batch = tensor[:max_items].detach().cpu()
        rows.append([_to_pil_rgb(item) for item in batch])

    if not rows:
        raise ValueError("No images provided for panel.")

    cell_w, cell_h = rows[0][0].size
    n_rows = len(rows)
    n_cols = max_items
    canvas = Image.new("RGB", (n_cols * cell_w, n_rows * cell_h), color=(255, 255, 255))

    for r, row in enumerate(rows):
        for c in range(min(n_cols, len(row))):
            canvas.paste(row[c], (c * cell_w, r * cell_h))

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(file_path)


def draw_ablation_table(results: List[Dict[str, float]], output_path: str) -> None:
    headers = ["Variant", "PSNR", "SSIM", "L1", "Identity"]
    rows = [
        [
            str(r["variant"]),
            f"{float(r['psnr']):.4f}",
            f"{float(r['ssim']):.4f}",
            f"{float(r['l1']):.4f}",
            f"{float(r['identity_similarity']):.6f}",
        ]
        for r in results
    ]

    col_widths = [220, 130, 130, 130, 160]
    row_h = 48
    title_h = 70
    width = sum(col_widths)
    height = title_h + row_h * (len(rows) + 1)

    img = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, title_h], fill=(30, 41, 59))
    draw.text((20, 24), "Ablation Comparison (Orchids AI)", fill=(241, 245, 249))

    y = title_h
    x = 0
    for idx, header in enumerate(headers):
        draw.rectangle([x, y, x + col_widths[idx], y + row_h], fill=(226, 232, 240), outline=(148, 163, 184), width=1)
        draw.text((x + 10, y + 15), header, fill=(15, 23, 42))
        x += col_widths[idx]

    for row_idx, row in enumerate(rows):
        y = title_h + row_h * (row_idx + 1)
        x = 0
        shade = (255, 255, 255) if row_idx % 2 == 0 else (248, 250, 252)
        for col_idx, value in enumerate(row):
            draw.rectangle([x, y, x + col_widths[col_idx], y + row_h], fill=shade, outline=(203, 213, 225), width=1)
            draw.text((x + 10, y + 15), value, fill=(30, 41, 59))
            x += col_widths[col_idx]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def draw_ablation_visual_comparison(
    panels: Dict[str, str],
    output_path: str,
    section: str = "restored",
) -> None:
    variant_names = list(panels.keys())
    if not variant_names:
        raise ValueError("No panel images found for visual comparison.")

    loaded: List[tuple[str, Image.Image]] = []
    for name in variant_names:
        panel_path = Path(panels[name])
        if not panel_path.exists():
            continue
        loaded.append((name, Image.open(panel_path).convert("RGB")))

    if not loaded:
        raise FileNotFoundError("No valid panel image paths found.")

    # Panels are stacked as rows: input, prior, mask, restored.
    row_index = {"input": 0, "prior": 1, "mask": 2, "restored": 3}.get(section, 3)

    crops: List[tuple[str, Image.Image]] = []
    for name, panel in loaded:
        w, h = panel.size
        row_h = h // 4
        y0 = row_index * row_h
        crop = panel.crop((0, y0, w, y0 + row_h))
        crops.append((name, crop))

    label_h = 46
    gap = 10
    single_w, single_h = crops[0][1].size
    total_w = single_w * len(crops) + gap * (len(crops) - 1)
    total_h = single_h + label_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for name, crop in crops:
        draw.rectangle([x, 0, x + single_w, label_h], fill=(241, 245, 249))
        draw.text((x + 10, 14), name, fill=(15, 23, 42))
        canvas.paste(crop, (x, label_h))
        x += single_w + gap

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_json(data: object, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
