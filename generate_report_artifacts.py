from __future__ import annotations

from pathlib import Path

from utils.plot_curves import draw_loss_curves


def main() -> None:
    output_dir = Path("outputs")
    history = output_dir / "loss_history.json"
    if not history.exists():
        raise FileNotFoundError("outputs/loss_history.json not found. Run training first.")

    draw_loss_curves(str(history), str(output_dir / "eval" / "loss_curves.png"))
    print("Generated outputs/eval/loss_curves.png")


if __name__ == "__main__":
    main()
