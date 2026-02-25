from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from data.dataset import FaceOcclusionDataset
from models.gan import DualBranchGenerator
from models.reconstruction import ReconstructionPriorNet, pytorch3d_is_available
from models.segmentation import compute_occlusion_mask
from utils.ablation import compute_prior, get_ablation_variant
from utils.config import load_config
from utils.metrics import identity_similarity, l1_score, psnr, ssim
from utils.runtime import ensure_dir, resolve_device
from utils.visualization import save_panel


class TinyIdentityEmbedder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 2, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


def evaluate(
    config_path: str,
    checkpoint_path: Optional[str],
    variant_name: str = "full",
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    cfg = load_config(config_path)
    variant = get_ablation_variant(variant_name)

    base_output_dir = Path(output_dir) if output_dir else Path(cfg["experiment"]["output_dir"])
    eval_dir = base_output_dir / "eval"
    ensure_dir(str(eval_dir))

    device = resolve_device(str(cfg["training"]["device"]))
    threshold = float(cfg["training"]["occlusion_threshold"])

    ckpt_path = Path(checkpoint_path or cfg["inference"]["checkpoint"])
    checkpoint = torch.load(ckpt_path, map_location=device)

    use_pytorch3d = bool(checkpoint.get("use_pytorch3d_prior", cfg["training"].get("use_pytorch3d_prior", False)))

    reconstructor = None
    if variant.use_prior:
        reconstructor = ReconstructionPriorNet(
            use_pytorch3d=use_pytorch3d,
            image_size=int(cfg["data"]["image_size"]),
        ).to(device)
        recon_state = checkpoint.get("reconstructor")
        if recon_state is not None:
            reconstructor.load_state_dict(recon_state)
        reconstructor.eval()

    generator = DualBranchGenerator().to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    dataset = FaceOcclusionDataset(
        root=cfg["data"]["val_root"],
        image_size=int(cfg["data"]["image_size"]),
        synthetic_count=int(cfg["data"]["val_synthetic_samples"]),
        use_synthetic_if_missing=bool(cfg["data"]["use_synthetic_if_missing"]),
    )
    loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
    embedder = TinyIdentityEmbedder().to(device).eval()

    stats = {"psnr": 0.0, "ssim": 0.0, "l1": 0.0, "identity_similarity": 0.0}
    batches = 0

    panel_path = eval_dir / f"review_panel_{variant_name}.png"
    panel_cache: Dict[str, torch.Tensor] | None = None

    with torch.no_grad():
        for idx, (occluded, clean, _) in enumerate(loader):
            occluded = occluded.to(device)
            clean = clean.to(device)

            prior = compute_prior(occluded, reconstructor, variant.use_prior)
            mask = compute_occlusion_mask(occluded, prior, threshold)
            restored, refined_mask, _ = generator(occluded, prior, mask)

            stats["psnr"] += float(psnr(restored, clean).item())
            stats["ssim"] += float(ssim(restored, clean).item())
            stats["l1"] += float(l1_score(restored, clean).item())
            stats["identity_similarity"] += float(identity_similarity(embedder(restored), embedder(clean)).item())
            batches += 1

            if idx == 0:
                panel_cache = {
                    "input": occluded,
                    "prior": prior,
                    "mask": refined_mask,
                    "restored": restored,
                }
                save_panel(panel_cache, str(panel_path), max_items=int(cfg["inference"]["num_examples"]))

    averaged = {k: v / max(batches, 1) for k, v in stats.items()}
    averaged["variant"] = variant_name
    averaged["pytorch3d_prior_requested"] = bool(use_pytorch3d)
    averaged["pytorch3d_available"] = bool(pytorch3d_is_available())

    metrics_path = eval_dir / f"metrics_{variant_name}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(averaged, handle, indent=2)

    # Keep backward-compatible output names for full model.
    if variant_name == "full":
        with (eval_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(averaged, handle, indent=2)
        if panel_cache is not None:
            save_panel(panel_cache, str(eval_dir / "review_panel.png"), max_items=int(cfg["inference"]["num_examples"]))

    print(f"Evaluation complete for variant={variant_name}")
    print(json.dumps(averaged, indent=2))
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--variant", default="full", choices=["full", "no_prior", "no_identity", "no_adversarial"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    evaluate(args.config, args.checkpoint, args.variant, args.output_dir)


if __name__ == "__main__":
    main()
