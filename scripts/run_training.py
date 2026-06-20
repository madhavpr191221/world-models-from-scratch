"""
Real VICReg training run — entry point.

Usage (from project root):
    uv run python scripts/run_training.py

To resume an interrupted run:
    uv run python scripts/run_training.py --resume checkpoints/vicreg/final.pt

Checkpoints saved to cfg.checkpoint_dir (default: checkpoints/vicreg/):
    epoch_0009.pt, epoch_0019.pt, ... -- periodic, every checkpoint_every_n_epochs
    best.pt                          -- lowest L_total seen so far
    final.pt                         -- always saved at the end

Loss curves logged to cfg.log_dir/loss_log.csv as training runs --
visualize anytime with scripts/plot_loss_curves.py (works mid-run too).
"""

import argparse

from jepa_world_models.vic_reg_loss.config import VICRegConfig
from jepa_world_models.vic_reg_loss.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VICReg on STL-10")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint file to resume training from"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the number of epochs from config (for quick experiments)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override num_workers from config (try 4 to test parallel data loading)"
    )
    args = parser.parse_args()

    cfg = VICRegConfig()

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()