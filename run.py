from __future__ import annotations

import argparse

from run_ablation import run_all_ablations
from test import evaluate
from train import train_variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchids AI training/eval launcher")
    parser.add_argument("mode", choices=["train", "test", "all", "ablation"], nargs="?", default="all")
    parser.add_argument("--variant", default="full", choices=["full", "no_prior", "no_identity", "no_adversarial"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if args.mode == "train":
        train_variant(args.config, variant_name=args.variant)
    elif args.mode == "test":
        evaluate(args.config, None, variant_name=args.variant)
    elif args.mode == "ablation":
        run_all_ablations(args.config)
    else:
        ckpt = train_variant(args.config, variant_name=args.variant)
        evaluate(args.config, str(ckpt), variant_name=args.variant)


if __name__ == "__main__":
    main()
