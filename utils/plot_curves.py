from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if abs(v_max - v_min) < 1e-9:
        return [0.5 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def draw_loss_curves(history_path: str, output_path: str) -> None:
    history_file = Path(history_path)
    if not history_file.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    with history_file.open("r", encoding="utf-8") as handle:
        history: List[Dict[str, float]] = json.load(handle)

    keys = ["train_gen", "train_disc", "train_recon", "val_l1"]
    width, height = 960, 540
    margin = 60
    chart_w = width - 2 * margin
    chart_h = height - 2 * margin

    image = Image.new("RGB", (width, height), (250, 252, 255))
    draw = ImageDraw.Draw(image)

    draw.rectangle([margin, margin, margin + chart_w, margin + chart_h], outline=(40, 40, 40), width=2)
    draw.text((margin, 20), "Orchids AI Loss Curves", fill=(20, 20, 20))

    palette = {
        "train_gen": (220, 53, 69),
        "train_disc": (13, 110, 253),
        "train_recon": (25, 135, 84),
        "val_l1": (255, 153, 0),
    }

    for idx, key in enumerate(keys):
        values = [float(item.get(key, 0.0)) for item in history]
        scaled = _normalize(values)
        if len(scaled) < 2:
            continue

        points = []
        for i, v in enumerate(scaled):
            x = margin + int(i * chart_w / (len(scaled) - 1))
            y = margin + chart_h - int(v * chart_h)
            points.append((x, y))
        draw.line(points, fill=palette[key], width=3)
        draw.text((margin + 10, margin + 10 + idx * 18), key, fill=palette[key])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
