from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from test import evaluate
from train import train_variant
from utils.ablation import resolve_variants
from utils.config import load_config
from utils.runtime import ensure_dir
from utils.visualization import draw_ablation_table, draw_ablation_visual_comparison, save_json


def _variant_output_dir(cfg: Dict, variant_name: str) -> Path:
    base = Path(cfg["experiment"]["output_dir"])
    if variant_name == "full":
        return base
    subdir = str(cfg.get("ablation", {}).get("output_subdir", "ablations"))
    return base / subdir / variant_name


def run_all_ablations(config_path: str = "config.yaml") -> List[Dict[str, float]]:
    cfg = load_config(config_path)
    ablation_cfg = cfg.get("ablation", {})
    variants = resolve_variants(ablation_cfg.get("variants"))

    results: List[Dict[str, float]] = []
    panel_paths: Dict[str, str] = {}

    for variant in variants:
        variant_out = _variant_output_dir(cfg, variant)
        ensure_dir(str(variant_out))

        checkpoint = train_variant(config_path, variant_name=variant, output_dir=str(variant_out))
        metrics = evaluate(config_path, str(checkpoint), variant_name=variant, output_dir=str(variant_out))
        results.append(metrics)

        panel_path = variant_out / "eval" / f"review_panel_{variant}.png"
        panel_paths[variant] = str(panel_path)

    results_json_path = str(ablation_cfg.get("results_json", "outputs/eval/ablation_results.json"))
    table_png_path = str(ablation_cfg.get("table_png", "outputs/eval/ablation_table.png"))
    panel_png_path = str(ablation_cfg.get("panel_png", "outputs/eval/ablation_visual_comparison.png"))

    save_json(results, results_json_path)
    draw_ablation_table(results, table_png_path)
    draw_ablation_visual_comparison(panel_paths, panel_png_path, section="restored")

    print("Ablation run complete")
    print(f"Results JSON: {results_json_path}")
    print(f"Table PNG: {table_png_path}")
    print(f"Visual comparison: {panel_png_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Orchids AI ablation suite")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    run_all_ablations(args.config)


if __name__ == "__main__":
    main()
