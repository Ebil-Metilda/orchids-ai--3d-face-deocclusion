from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FaceOcclusionDataset
from losses.composite import CompositeGeneratorLoss, LossWeights, discriminator_loss
from models.gan import DualBranchGenerator, PatchDiscriminator
from models.reconstruction import ReconstructionPriorNet, pytorch3d_is_available
from models.segmentation import compute_occlusion_mask
from utils.ablation import compute_prior, get_ablation_variant, get_variant_index
from utils.config import load_config
from utils.runtime import ensure_dir, resolve_device, set_seed
from utils.visualization import save_panel


def build_loaders(cfg: Dict) -> tuple[DataLoader, DataLoader]:
    image_size = int(cfg["data"]["image_size"])
    train_dataset = FaceOcclusionDataset(
        root=cfg["data"]["train_root"],
        image_size=image_size,
        synthetic_count=int(cfg["data"]["synthetic_samples"]),
        use_synthetic_if_missing=bool(cfg["data"]["use_synthetic_if_missing"]),
    )
    val_dataset = FaceOcclusionDataset(
        root=cfg["data"]["val_root"],
        image_size=image_size,
        synthetic_count=int(cfg["data"]["val_synthetic_samples"]),
        use_synthetic_if_missing=bool(cfg["data"]["use_synthetic_if_missing"]),
    )
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"]["num_workers"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def _variant_output_dir(cfg: Dict, variant_name: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    if variant_name == "full":
        return Path(cfg["experiment"]["output_dir"])
    subdir = str(cfg.get("ablation", {}).get("output_subdir", "ablations"))
    return Path(cfg["experiment"]["output_dir"]) / subdir / variant_name


def train_variant(config_path: str, variant_name: str = "full", output_dir: str | None = None) -> Path:
    cfg = load_config(config_path)
    variant = get_ablation_variant(variant_name)

    seed_offset = int(cfg.get("ablation", {}).get("seed_offset_per_variant", 100))
    variant_seed = int(cfg["experiment"]["seed"]) + seed_offset * get_variant_index(variant_name)
    set_seed(variant_seed)

    run_output_dir = _variant_output_dir(cfg, variant_name, output_dir)
    checkpoints_dir = run_output_dir / "checkpoints"
    previews_dir = run_output_dir / "previews"
    ensure_dir(str(checkpoints_dir))
    ensure_dir(str(previews_dir))

    device = resolve_device(str(cfg["training"]["device"]))
    print(f"Using device: {device}")
    print(f"Training variant: {variant_name}")

    use_pytorch3d = bool(cfg["training"].get("use_pytorch3d_prior", False))
    if use_pytorch3d:
        print(f"PyTorch3D prior requested. available={pytorch3d_is_available()}")

    train_loader, val_loader = build_loaders(cfg)

    reconstructor = None
    if variant.use_prior:
        reconstructor = ReconstructionPriorNet(
            use_pytorch3d=use_pytorch3d,
            image_size=int(cfg["data"]["image_size"]),
        ).to(device)

    generator = DualBranchGenerator().to(device)
    discriminator = PatchDiscriminator().to(device) if variant.use_adversarial_loss else None

    opt_recon = None
    if reconstructor is not None:
        opt_recon = Adam(
            reconstructor.parameters(),
            lr=float(cfg["training"]["learning_rate_g"]),
            betas=tuple(cfg["training"]["betas"]),
        )

    opt_g = Adam(generator.parameters(), lr=float(cfg["training"]["learning_rate_g"]), betas=tuple(cfg["training"]["betas"]))
    opt_d = None
    if discriminator is not None:
        opt_d = Adam(discriminator.parameters(), lr=float(cfg["training"]["learning_rate_d"]), betas=tuple(cfg["training"]["betas"]))

    loss_weights = LossWeights(
        lambda_l1=float(cfg["training"]["lambda_l1"]),
        lambda_perceptual=float(cfg["training"]["lambda_perceptual"]),
        lambda_identity=float(cfg["training"]["lambda_identity"]) if variant.use_identity_loss else 0.0,
        lambda_adv=float(cfg["training"]["lambda_adv"]) if variant.use_adversarial_loss else 0.0,
        lambda_smooth=float(cfg["training"]["lambda_smooth"]),
    )
    generator_loss_fn = CompositeGeneratorLoss(weights=loss_weights, device=device)
    recon_loss_fn = nn.L1Loss()

    epochs = int(cfg["training"]["epochs"])
    threshold = float(cfg["training"]["occlusion_threshold"])
    grad_clip = float(cfg["training"]["grad_clip"])

    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        if reconstructor is not None:
            reconstructor.train()
        generator.train()
        if discriminator is not None:
            discriminator.train()

        running = {
            "recon": 0.0,
            "gen": 0.0,
            "disc": 0.0,
            "l1": 0.0,
            "perceptual": 0.0,
            "identity": 0.0,
            "adversarial": 0.0,
            "smoothness": 0.0,
        }
        steps = 0

        progress = tqdm(train_loader, desc=f"[{variant_name}] Epoch {epoch}/{epochs}")
        for occluded, clean, true_mask in progress:
            occluded = occluded.to(device)
            clean = clean.to(device)
            true_mask = true_mask.to(device)

            prior = compute_prior(occluded, reconstructor, variant.use_prior)
            coarse_mask = compute_occlusion_mask(occluded, prior, threshold=threshold)

            fake, refined_mask, _ = generator(occluded, prior.detach(), coarse_mask)

            loss_d = occluded.new_tensor(0.0)
            if discriminator is not None and opt_d is not None:
                pred_real = discriminator(clean)
                pred_fake_detached = discriminator(fake.detach())
                loss_d = discriminator_loss(pred_real, pred_fake_detached)
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                opt_d.step()

            loss_recon = occluded.new_tensor(0.0)
            if reconstructor is not None and opt_recon is not None:
                loss_recon = recon_loss_fn(prior, clean)
                opt_recon.zero_grad(set_to_none=True)
                loss_recon.backward()
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), grad_clip)
                opt_recon.step()

            pred_fake = discriminator(fake) if discriminator is not None else None
            loss_g, detail = generator_loss_fn(fake, clean, refined_mask, pred_fake)
            mask_supervision = nn.functional.l1_loss(refined_mask, true_mask)
            loss_g = loss_g + 0.5 * mask_supervision

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            opt_g.step()

            steps += 1
            running["recon"] += float(loss_recon.detach().item())
            running["gen"] += float(loss_g.detach().item())
            running["disc"] += float(loss_d.detach().item())
            running["l1"] += detail["l1"]
            running["perceptual"] += detail["perceptual"]
            running["identity"] += detail["identity"]
            running["adversarial"] += detail["adversarial"]
            running["smoothness"] += detail["smoothness"]

            progress.set_postfix({"g": running["gen"] / steps, "d": running["disc"] / max(steps, 1), "r": running["recon"] / max(steps, 1)})

        epoch_log = {f"train_{k}": v / max(steps, 1) for k, v in running.items()}

        if reconstructor is not None:
            reconstructor.eval()
        generator.eval()
        with torch.no_grad():
            val_occluded, val_clean, _ = next(iter(val_loader))
            val_occluded = val_occluded.to(device)
            val_clean = val_clean.to(device)
            val_prior = compute_prior(val_occluded, reconstructor, variant.use_prior)
            val_mask = compute_occlusion_mask(val_occluded, val_prior, threshold=threshold)
            val_output, _, _ = generator(val_occluded, val_prior, val_mask)

            val_l1 = nn.functional.l1_loss(val_output, val_clean)
            epoch_log["val_l1"] = float(val_l1.item())

            save_panel(
                {
                    "input": val_occluded,
                    "prior": val_prior,
                    "coarse_mask": val_mask,
                    "restored": val_output,
                },
                str(previews_dir / f"epoch_{epoch:03d}.png"),
            )

        history.append(epoch_log)

        ckpt_path = checkpoints_dir / "latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "reconstructor": reconstructor.state_dict() if reconstructor is not None else None,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict() if discriminator is not None else None,
                "config": cfg,
                "variant": variant_name,
                "use_pytorch3d_prior": use_pytorch3d,
                "pytorch3d_available": pytorch3d_is_available(),
            },
            ckpt_path,
        )

        print(f"[{variant_name}] Epoch {epoch} metrics: {epoch_log}")

    with (run_output_dir / "loss_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Training complete for variant={variant_name}. Checkpoint: {checkpoints_dir / 'latest.pt'}")
    return checkpoints_dir / "latest.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Orchids AI model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--variant", default="full", choices=["full", "no_prior", "no_identity", "no_adversarial"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    train_variant(args.config, args.variant, args.output_dir)


if __name__ == "__main__":
    main()
