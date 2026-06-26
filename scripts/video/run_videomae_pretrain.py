from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_world_models.analysis.videomae_pipeline import pretrain_videomae


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain a compact VideoMAE model.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--source-split", type=str, default="train")
    parser.add_argument("--subset-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="logs/videomae")
    parser.add_argument("--cache-dir", type=str, default="logs/videomae/cache")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = pretrain_videomae(
        data_root=args.data_root,
        source_split=args.source_split,
        subset_size=args.subset_size,
        image_size=args.image_size,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        device=args.device,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_recon_pct = max(0.0, (1.0 - float(result.train_loss)) * 100.0)
    val_recon_pct = max(0.0, (1.0 - float(result.val_loss)) * 100.0)
    (out / "result.json").write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")
    print(f"Wrote VideoMAE report to {out / 'result.json'}")
    print(f"train_loss={result.train_loss:.6f}")
    print(f"val_loss={result.val_loss:.6f}")
    print(f"train_recon_score={train_recon_pct:.2f}%")
    print(f"val_recon_score={val_recon_pct:.2f}%")
    print(f"best_checkpoint={result.checkpoint_path}")
    print(f"final_checkpoint={result.final_checkpoint_path}")


if __name__ == "__main__":
    main()
