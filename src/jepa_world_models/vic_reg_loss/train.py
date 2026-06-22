import csv
from dataclasses import asdict
from pathlib import Path

import torch
import time 
from torch.utils.data import DataLoader
from tqdm import tqdm
from jepa_world_models.contrastive_learning import ViTEncoder, Projector
from jepa_world_models.contrastive_learning.augmentations import vicreg_augmentation
from jepa_world_models.data.stl10 import STL10Unlabeled
from jepa_world_models.vic_reg_loss.config import VICRegConfig
from jepa_world_models.vic_reg_loss.loss import VICRegLoss

def build_dataloader(cfg: VICRegConfig) -> DataLoader:
    """
    Constructs the STL-10 unlabeled DataLoader with the VICReg two-view
    augmentation pipeline.

    GPU utilization note: num_workers and pin_memory affect how fast data
    reaches the GPU, not how the GPU itself is used. If the GPU sits idle
    waiting for data (a "data-bound" training loop), increasing
    num_workers helps. On Windows, num_workers > 0 can be unstable with
    CUDA -- start at 0, increase only if profiling shows the GPU is
    starved (see Section 6 timing notes).
    """
    transform = vicreg_augmentation(image_size=cfg.image_size)
    dataset = STL10Unlabeled(root=cfg.data_root, transform=transform)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,  # BatchNorm1d requires B >= 2; drop a ragged last batch of B=1
    )


def build_model(cfg: VICRegConfig) -> tuple[ViTEncoder, Projector]:
    """
    Constructs the encoder and projector, moves both to cfg.device.

    GPU utilization note: this is where the model actually lands on the
    GPU. Both encoder and projector must be on the SAME device as the
    data, or PyTorch raises a device mismatch error at the first forward
    pass.
    """
    encoder = ViTEncoder(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
    ).to(cfg.device)

    projector = Projector(
        in_dim=cfg.embed_dim,
        proj_dim=cfg.proj_dim,
    ).to(cfg.device)

    return encoder, projector

def build_optimizer(cfg: VICRegConfig, params) -> torch.optim.Adam:
    """
    Adam optimizer over the combined encoder + projector parameters.

    params should be the union of encoder.parameters() and
    projector.parameters() -- both are trained jointly, since the
    VICReg loss backpropagates through the projector into the encoder.

    weight_decay=1e-6 is intentionally light. Heavy L2 regularization
    can fight against the variance term's goal of keeping representation
    spread -- shrinking weights toward zero pulls against maintaining
    high variance per dimension.
    """
    return torch.optim.Adam(
        params,
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )


def warmup_lr(epoch: int, cfg: VICRegConfig) -> float:
    """
    Linear warmup: ramps learning rate from 0 to cfg.lr over the first
    cfg.warmup_epochs epochs, then holds constant at cfg.lr.

    epoch is 0-indexed. At epoch=0, lr starts near 0 (not exactly 0, to
    avoid a completely dead first step); at epoch=warmup_epochs and
    beyond, lr = cfg.lr exactly.
    """
    if epoch >= cfg.warmup_epochs:
        return cfg.lr
    # linear ramp: epoch 0 -> small fraction of lr, epoch warmup_epochs-1 -> nearly lr
    return cfg.lr * (epoch + 1) / cfg.warmup_epochs


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Updates the learning rate for all parameter groups in-place."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_one_epoch(
    encoder: ViTEncoder,
    projector: Projector,
    loss_fn: VICRegLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    cfg: VICRegConfig,
    epoch: int,
    use_amp: bool = True,
    log_csv_path: str = "logs/vicreg/loss_log.csv",
) -> dict:
    """
    Runs one full epoch of VICReg training.

    Logs every step's loss components to a CSV file (log_csv_path),
    appended across epochs, so the full training trajectory can be
    plotted afterward with scripts/plot_loss_curves.py -- this is the
    primary diagnostic tool for catching collapse: L_var trending UP
    toward gamma and L_cov trending toward 0 simultaneously, even while
    L_total looks like it's improving, is the collapse signature.
    """
    encoder.train()
    projector.train()

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    running_total, running_inv, running_var, running_cov = 0.0, 0.0, 0.0, 0.0
    n_batches = len(loader)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    epoch_start = time.time()

    # Set up CSV logging -- create with header if this is the very first
    # write, otherwise append (so multi-epoch runs accumulate one
    # continuous log rather than overwriting each epoch).
    log_path = Path(log_csv_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    csv_file = open(log_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["epoch", "step", "global_step", "L_total", "L_inv", "L_var", "L_cov"])

    progress_bar = tqdm(
        enumerate(loader),
        total=n_batches,
        desc=f"Epoch {epoch + 1}",
        unit="batch",
    )

    for step, (view_a, view_b) in progress_bar:
        view_a = view_a.to(device, non_blocking=True)
        view_b = view_b.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            combined = torch.cat([view_a, view_b], dim=0)
            z_combined = projector(encoder(combined))
            z_a, z_b = z_combined.chunk(2, dim=0)
            loss_out = loss_fn(z_a, z_b)

        scaler.scale(loss_out.total).backward()
        scaler.step(optimizer)
        scaler.update()

        running_total += loss_out.total.item()
        running_inv += loss_out.inv.item()
        running_var += loss_out.var.item()
        running_cov += loss_out.cov.item()

        global_step = epoch * n_batches + step
        csv_writer.writerow([
            epoch + 1, step, global_step,
            loss_out.total.item(), loss_out.inv.item(),
            loss_out.var.item(), loss_out.cov.item(),
        ])

        mem_mib = (
            torch.cuda.memory_allocated() / (1024**2) if device == "cuda" else 0.0
        )
        progress_bar.set_postfix({
            "L_tot": f"{loss_out.total.item():.3f}",
            "L_inv": f"{loss_out.inv.item():.3f}",
            "L_var": f"{loss_out.var.item():.3f}",
            "L_cov": f"{loss_out.cov.item():.4f}",
            "mem_MiB": f"{mem_mib:.0f}",
        })

    csv_file.close()

    epoch_time = time.time() - epoch_start
    peak_mem_mib = (
        torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
    )

    avg = {
        "total": running_total / n_batches,
        "inv": running_inv / n_batches,
        "var": running_var / n_batches,
        "cov": running_cov / n_batches,
        "epoch_time_sec": epoch_time,
        "peak_mem_mib": peak_mem_mib,
        "images_per_sec": (n_batches * cfg.batch_size) / epoch_time,
    }

    print(
        f"\n  Epoch {epoch + 1} done in {epoch_time:.1f}s "
        f"({avg['images_per_sec']:.1f} images/sec) | "
        f"avg L_total={avg['total']:.4f} | peak GPU mem: {peak_mem_mib:.1f} MiB\n"
    )

    return avg


def save_checkpoint(
    encoder: ViTEncoder,
    projector: Projector,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    avg_loss: float,
    checkpoint_dir: str,
    filename: str,
    cfg: VICRegConfig | None = None,
) -> str:
    """
    Saves a full training checkpoint: encoder weights, projector weights,
    optimizer state (so resuming continues Adam's momentum correctly,
    not from scratch), and the epoch number.

    Returns the full path the checkpoint was saved to.
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    full_path = checkpoint_path / filename

    torch.save({
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "projector_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "avg_loss": avg_loss,
        "config": asdict(cfg) if cfg is not None else None,
    }, full_path)

    return str(full_path)


def load_model_from_checkpoint(
    checkpoint_file: str,
    device: str | None = None,
) -> tuple[VICRegConfig, ViTEncoder, Projector, dict]:
    """
    Load encoder and projector weights from a checkpoint, reconstructing the
    exact model configuration when it is stored in the checkpoint.
    """
    checkpoint = torch.load(checkpoint_file, map_location=device or "cpu")
    cfg_data = checkpoint.get("config")
    cfg = VICRegConfig(**cfg_data) if cfg_data is not None else VICRegConfig()
    if device is not None:
        cfg.device = device

    encoder, projector = build_model(cfg)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    projector.load_state_dict(checkpoint["projector_state_dict"])
    encoder.eval()
    projector.eval()
    return cfg, encoder, projector, checkpoint


def load_checkpoint(
    checkpoint_file: str,
    encoder: ViTEncoder,
    projector: Projector,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> int:
    """
    Loads a checkpoint in-place into encoder, projector, and optimizer.
    Returns the epoch number the checkpoint was saved at, so training
    can resume from epoch+1.
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    projector.load_state_dict(checkpoint["projector_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


def train(cfg: VICRegConfig, resume_from: str | None = None) -> None:
    """
    Full VICReg training run: builds the pipeline, runs cfg.epochs epochs
    with warmup, saves a checkpoint every cfg.checkpoint_every_n_epochs
    epochs, and separately tracks + saves the single best checkpoint
    (lowest average L_total seen so far).

    resume_from: optional path to a checkpoint file to resume training
    from (e.g. after an interrupted run).
    """
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        cfg.device = "cpu"

    print(cfg)
    if cfg.device == "cuda":
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    loader = build_dataloader(cfg)
    encoder, projector = build_model(cfg)
    all_params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = build_optimizer(cfg, all_params)
    loss_fn = VICRegLoss(
        lambda_=cfg.lambda_, mu=cfg.mu, nu=cfg.nu, gamma=cfg.gamma, eps=cfg.eps
    )

    start_epoch = 0
    best_loss = float("inf")

    if resume_from is not None:
        start_epoch = load_checkpoint(resume_from, encoder, projector, optimizer, cfg.device) + 1
        print(f"Resumed from {resume_from}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        lr = warmup_lr(epoch, cfg)
        set_lr(optimizer, lr)
        print(f"\n{'='*70}\nEPOCH {epoch+1}/{cfg.epochs - 1} (lr={lr:.6f})\n{'='*70}")

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

        # Save periodic checkpoint
        if (epoch + 1) % cfg.checkpoint_every_n_epochs == 0:
            path = save_checkpoint(
                encoder, projector, optimizer, epoch, avg["total"],
                cfg.checkpoint_dir, f"epoch_{epoch + 1:04d}.pt", cfg=cfg,
            )
            print(f"  Saved periodic checkpoint: {path}")

        # Save best checkpoint (overwrites previous best)
        if avg["total"] < best_loss:
            best_loss = avg["total"]
            path = save_checkpoint(
                encoder, projector, optimizer, epoch, avg["total"],
                cfg.checkpoint_dir, "best.pt", cfg=cfg,
            )
            print(f"  New best L_total={best_loss:.4f}, saved: {path}")

    # Always save a final checkpoint at the end of training -- guard
    # against the loop never running (e.g. resuming a checkpoint whose
    # epoch is already >= cfg.epochs), which would leave `avg` undefined.
    if "avg" not in locals():
        print(
            f"\nWARNING: training loop did not run (start_epoch={start_epoch} "
            f">= cfg.epochs={cfg.epochs}). Nothing to save as 'final' -- "
            f"the resumed checkpoint is already the most recent state."
        )
    else:
        final_path = save_checkpoint(
            encoder, projector, optimizer, cfg.epochs - 1, avg["total"],
            cfg.checkpoint_dir, "final.pt", cfg=cfg,
        )
        print(f"\nTraining complete. Final checkpoint: {final_path}")
        print(f"Best checkpoint (L_total={best_loss:.4f}): {cfg.checkpoint_dir}/best.pt")
