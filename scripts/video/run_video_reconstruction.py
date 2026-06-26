from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_world_models.analysis.video_reconstruction import (
    build_reconstruction_head,
    build_tubelet_bank,
    load_clip_from_video_path,
    reconstruct_clip_with_decoder,
    reconstruct_clip_with_bank,
    save_reconstruction_artifacts,
)
from jepa_world_models.analysis.videomae_pipeline import SomethingSomethingVideoDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a masked VideoMAE reconstruction demo.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--source-split", type=str, default="train")
    parser.add_argument("--subset-size", type=int, default=512)
    parser.add_argument("--bank-batch-size", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-hidden-dim", type=int, default=None)
    parser.add_argument("--head-blocks", type=int, default=2)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--mask-mode", type=str, default="middle", choices=["middle", "random"])
    parser.add_argument("--reconstruction-mode", type=str, default="decoder", choices=["decoder", "retrieval"])
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--video-index", type=int, default=0)
    parser.add_argument("--cache-dir", type=str, default="logs/video_reconstruction/cache")
    parser.add_argument("--output-dir", type=str, default="logs/video_reconstruction")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bank = None
    head = None
    if args.reconstruction_mode == "retrieval":
        bank = build_tubelet_bank(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            source_split=args.source_split,
            subset_size=args.subset_size,
            image_size=args.image_size,
            num_frames=args.num_frames,
            batch_size=args.bank_batch_size,
            cache_dir=args.cache_dir,
        )
    else:
        head, head_bundle = build_reconstruction_head(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            source_split=args.source_split,
            subset_size=args.subset_size,
            image_size=args.image_size,
            num_frames=args.num_frames,
            batch_size=args.bank_batch_size,
            epochs=args.head_epochs,
            lr=args.head_lr,
            hidden_dim=args.head_hidden_dim,
            num_blocks=args.head_blocks,
            dropout=args.head_dropout,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )

    if args.video_path:
        clip = load_clip_from_video_path(args.video_path, num_frames=args.num_frames, image_size=args.image_size)
        source_name = Path(args.video_path).stem
    else:
        dataset = SomethingSomethingVideoDataset(
            data_root=args.data_root,
            split=args.source_split,
            image_size=args.image_size,
            num_frames=args.num_frames,
            limit=max(args.video_index + 1, 1),
            seed=args.seed,
            cache_dir=args.cache_dir,
        )
        sample = dataset[args.video_index]
        clip = sample["clip"]
        source_name = sample["video_id"]

    if args.reconstruction_mode == "retrieval":
        result = reconstruct_clip_with_bank(
            checkpoint_path=args.checkpoint,
            bank=bank,
            clip=clip,
            mask_ratio=args.mask_ratio,
            mask_mode=args.mask_mode,
            seed=args.seed,
        )
    else:
        result = reconstruct_clip_with_decoder(
            checkpoint_path=args.checkpoint,
            head=head,
            clip=clip,
            mask_ratio=args.mask_ratio,
            mask_mode=args.mask_mode,
            seed=args.seed,
        )
    run_dir = output_dir / f"{source_name}_{args.mask_mode}_{int(args.mask_ratio * 100)}"
    payload = save_reconstruction_artifacts(result, run_dir)
    payload.update(
        {
            "source_name": source_name,
            "mask_ratio": args.mask_ratio,
            "mask_mode": args.mask_mode,
            "reconstruction_mode": args.reconstruction_mode,
        }
    )
    if head is not None:
        payload["head_checkpoint"] = head_bundle.checkpoint_path
        payload["head_train_loss"] = head_bundle.train_loss
        payload["head_val_loss"] = head_bundle.val_loss
        payload["head_best_epoch"] = head_bundle.best_epoch
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote reconstruction artifacts to {run_dir}")
    print(f"original_video={payload['original_video']}")
    print(f"masked_video={payload['masked_video']}")
    print(f"reconstructed_video={payload['reconstructed_video']}")
    if "psnr_db" in payload:
        print(f"psnr_db={payload['psnr_db']:.3f}")


if __name__ == "__main__":
    main()

