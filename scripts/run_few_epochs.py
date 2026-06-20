"""
Multi-epoch smoke test for VICReg training — runs several full epochs
to see whether L_var peaks and reverses, as the theory predicts it
eventually must once L_inv stops dominating the gradient.

Usage (from project root):
    uv run python scripts/run_few_epochs.py
"""

import torch

from jepa_world_models.contrastive_learning import ViTEncoder, Projector
from jepa_world_models.vic_reg_loss.config import VICRegConfig
from jepa_world_models.vic_reg_loss.loss import VICRegLoss
from jepa_world_models.vic_reg_loss.train import (
    build_dataloader,
    build_model,
    build_optimizer,
    warmup_lr,
    set_lr,
    train_one_epoch,
)

N_EPOCHS = 5  # run 5 full epochs, not just 1


def main() -> None:
    cfg = VICRegConfig()

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        cfg.device = "cpu"

    print(f"Running {N_EPOCHS} epochs, lr={cfg.lr}, warmup_epochs={cfg.warmup_epochs}\n")

    loader = build_dataloader(cfg)
    encoder, projector = build_model(cfg)
    all_params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = build_optimizer(cfg, all_params)
    loss_fn = VICRegLoss(
        lambda_=cfg.lambda_, mu=cfg.mu, nu=cfg.nu, gamma=cfg.gamma, eps=cfg.eps
    )

    epoch_summaries = []

    for epoch in range(N_EPOCHS):
        # apply warmup schedule
        lr = warmup_lr(epoch, cfg)
        set_lr(optimizer, lr)
        print(f"\n{'='*70}\nEPOCH {epoch} (lr={lr:.6f})\n{'='*70}")

        avg = train_one_epoch(
            encoder=encoder,
            projector=projector,
            loss_fn=loss_fn,
            loader=loader,
            optimizer=optimizer,
            device=cfg.device,
            cfg=cfg,
            epoch=epoch,
            use_amp=True,
        )
        epoch_summaries.append(avg)

    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL EPOCHS")
    print("=" * 70)
    print(f"{'Epoch':<8}{'L_total':<12}{'L_inv':<12}{'L_var':<12}{'L_cov':<12}")
    for i, s in enumerate(epoch_summaries):
        print(f"{i:<8}{s['total']:<12.4f}{s['inv']:<12.4f}{s['var']:<12.4f}{s['cov']:<12.4f}")


if __name__ == "__main__":
    main()