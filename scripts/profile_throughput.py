"""
Profiling script: measures data-loading time vs GPU compute time
separately, across different num_workers values, to determine whether
training is data-bound (GPU waiting for batches) or compute-bound
(GPU is the bottleneck) -- and whether increasing num_workers helps.

Usage (from project root):
    uv run python scripts/profile_throughput.py

Runs a short number of steps (not a full epoch) at each num_workers
value tested, reporting:
    - time spent waiting for the DataLoader to produce a batch
    - time spent in the forward + backward GPU computation
    - overall images/sec throughput
    - peak GPU memory

This directly answers "is the GPU sitting idle waiting for data" --
if data-loading time dominates, increasing num_workers should help;
if compute time dominates, num_workers won't matter much.
"""

import time

import torch
from tqdm import tqdm

from jepa_world_models.contrastive_learning import ViTEncoder, Projector
from jepa_world_models.contrastive_learning.augmentations import vicreg_augmentation
from jepa_world_models.data.stl10 import STL10Unlabeled
from jepa_world_models.vic_reg_loss.config import VICRegConfig
from jepa_world_models.vic_reg_loss.loss import VICRegLoss
from torch.utils.data import DataLoader

N_PROFILE_STEPS = 30  # short run, just enough to get a stable average
NUM_WORKERS_TO_TEST = [0, 2, 4, 8]

def profile_one_config(cfg: VICRegConfig, num_workers: int) -> dict:
    """Runs N_PROFILE_STEPS steps with the given num_workers, returning
    timing breakdown."""
    cfg.num_workers = num_workers

    print(f"  building dataset and dataloader (num_workers={num_workers})...")
    transform = vicreg_augmentation(image_size=cfg.image_size)
    dataset = STL10Unlabeled(root=cfg.data_root, transform=transform)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(num_workers > 0),
        drop_last=True,
    )
    print(f"  dataset ready: {len(dataset):,} images, {len(loader)} batches/epoch")

    print("  building model...")
    encoder = ViTEncoder(
        image_size=cfg.image_size, patch_size=cfg.patch_size,
        in_channels=cfg.in_channels, embed_dim=cfg.embed_dim,
        depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
    ).to(cfg.device)
    projector = Projector(in_dim=cfg.embed_dim, proj_dim=cfg.proj_dim).to(cfg.device)
    loss_fn = VICRegLoss(
        lambda_=cfg.lambda_, mu=cfg.mu, nu=cfg.nu, gamma=cfg.gamma, eps=cfg.eps
    )
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=cfg.lr
    )
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    print("  model ready, starting timed steps...")

    encoder.train()
    projector.train()

    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    data_wait_time = 0.0
    compute_time = 0.0

    loader_iter = iter(loader)
    overall_start = time.time()

    progress = tqdm(range(N_PROFILE_STEPS), desc=f"  workers={num_workers}", unit="step")

    for step in progress:
        t0 = time.time()
        view_a, view_b = next(loader_iter)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        data_wait_time += (t1 - t0)

        view_a = view_a.to(cfg.device, non_blocking=True)
        view_b = view_b.to(cfg.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        t2 = time.time()
        with torch.amp.autocast('cuda', enabled=True):
            combined = torch.cat([view_a, view_b], dim=0)
            z_combined = projector(encoder(combined))
            z_a, z_b = z_combined.chunk(2, dim=0)
            loss_out = loss_fn(z_a, z_b)

        scaler.scale(loss_out.total).backward()
        scaler.step(optimizer)
        scaler.update()

        if cfg.device == "cuda":
            torch.cuda.synchronize()
        t3 = time.time()
        compute_time += (t3 - t2)

        progress.set_postfix({
            "data_ms": f"{(t1-t0)*1000:.0f}",
            "compute_ms": f"{(t3-t2)*1000:.0f}",
        })

    overall_time = time.time() - overall_start
    peak_mem_mib = (
        torch.cuda.max_memory_allocated() / (1024**2) if cfg.device == "cuda" else 0.0
    )

    return {
        "num_workers": num_workers,
        "data_wait_time": data_wait_time,
        "compute_time": compute_time,
        "overall_time": overall_time,
        "images_per_sec": (N_PROFILE_STEPS * cfg.batch_size) / overall_time,
        "peak_mem_mib": peak_mem_mib,
        "data_fraction": data_wait_time / overall_time,
    }

def main() -> None:
    cfg = VICRegConfig()
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        cfg.device = "cpu"

    print(f"Profiling {N_PROFILE_STEPS} steps per config, "
          f"batch_size={cfg.batch_size}\n")

    results = []
    for nw in NUM_WORKERS_TO_TEST:
        print(f"Testing num_workers={nw}...")
        try:
            r = profile_one_config(cfg, nw)
            results.append(r)
            print(
                f"  data_wait={r['data_wait_time']:.2f}s "
                f"({r['data_fraction']*100:.1f}% of total) | "
                f"compute={r['compute_time']:.2f}s | "
                f"throughput={r['images_per_sec']:.1f} img/s | "
                f"peak_mem={r['peak_mem_mib']:.1f} MiB\n"
            )
        except Exception as e:
            print(f"  FAILED with num_workers={nw}: {e}\n")

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'workers':<10}{'data%':<10}{'img/s':<12}{'peak_mem_MiB':<15}")
    for r in results:
        print(f"{r['num_workers']:<10}{r['data_fraction']*100:<10.1f}"
              f"{r['images_per_sec']:<12.1f}{r['peak_mem_mib']:<15.1f}")

    if results:
        best = max(results, key=lambda r: r["images_per_sec"])
        print(f"\nBest throughput: num_workers={best['num_workers']} "
              f"({best['images_per_sec']:.1f} img/s)")
        if results[0]["data_fraction"] > 0.3:
            print(
                f"\nDIAGNOSIS: at num_workers=0, {results[0]['data_fraction']*100:.0f}% "
                f"of time was spent waiting for data -- training IS data-bound. "
                f"Set num_workers={best['num_workers']} in config.py for better throughput."
            )
        else:
            print(
                f"\nDIAGNOSIS: data-loading was already a small fraction of total time "
                f"even at num_workers=0 -- training is closer to compute-bound. "
                f"num_workers won't help much; the GPU itself is the bottleneck."
            )


if __name__ == "__main__":
    main()