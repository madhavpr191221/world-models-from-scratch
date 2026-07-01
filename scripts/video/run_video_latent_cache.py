from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_world_models.analysis.video_world_model import (
    _latent_sequence_bank_cache_path,
    build_latent_sequence_bank,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode video clips into cached latent sequences.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the frozen video encoder checkpoint.")
    parser.add_argument("--data-root", type=str, default="data", help="Data root containing Something-Something V2.")
    parser.add_argument("--source-split", type=str, default="train", help="Dataset split to sample videos from.")
    parser.add_argument("--subset-size", type=int, default=128, help="Number of usable videos to sample.")
    parser.add_argument("--image-size", type=int, default=224, help="Frame resize target.")
    parser.add_argument("--context-seconds", type=float, default=5.0, help="Observed context duration in seconds.")
    parser.add_argument("--future-seconds", type=float, default=1.5, help="Prediction horizon in seconds.")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Target sampling rate in frames per second.")
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=1,
        help="Batch size used while extracting cached latent sequences.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="logs/video_world_model/cache",
        help="Directory used for cached latent sequences.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/video_world_model/cache_summary.json",
        help="Path to the JSON summary of the cached latent bank.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override, for example cuda or cpu.")
    return parser


def _seconds_to_frame_count(seconds: float, sample_fps: float) -> int:
    frames = max(2, int(round(seconds * sample_fps)))
    return frames + (frames % 2)


def main() -> None:
    args = build_parser().parse_args()
    context_frames = _seconds_to_frame_count(args.context_seconds, args.sample_fps)
    future_frames = _seconds_to_frame_count(args.future_seconds, args.sample_fps)
    total_frames = context_frames + future_frames

    bank = build_latent_sequence_bank(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        source_split=args.source_split,
        subset_size=args.subset_size,
        image_size=args.image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
        sample_fps=args.sample_fps,
        batch_size=args.feature_batch_size,
        cache_dir=args.cache_dir,
        seed=args.seed,
        device=args.device,
    )

    cache_path = _latent_sequence_bank_cache_path(
        cache_dir=args.cache_dir,
        checkpoint_path=args.checkpoint,
        source_split=args.source_split,
        subset_size=args.subset_size,
        image_size=args.image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
        sample_fps=args.sample_fps,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_path": bank.checkpoint_path,
        "cache_path": str(cache_path),
        "cache_manifest_path": str(cache_path.with_suffix(".json")),
        "source_split": bank.source_split,
        "subset_size": bank.subset_size,
        "image_size": bank.image_size,
        "sample_fps": bank.sample_fps,
        "context_frames": bank.context_frames,
        "future_frames": bank.future_frames,
        "total_frames": bank.total_frames,
        "latent_shape": list(bank.context_latents.shape),
        "latent_dim": bank.latent_dim,
        "num_samples": len(bank.sample_indices),
        "encoder_fingerprint": bank.encoder_fingerprint,
        "cache_format_version": bank.cache_format_version,
        "cache_kind": bank.cache_kind,
        "created_at": bank.created_at,
        "content_hash": bank.content_hash,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote latent cache summary to {output}")
    print(f"latent_shape={tuple(bank.context_latents.shape)}")
    print(f"cache_format_version={bank.cache_format_version}")
    print(f"cache_kind={bank.cache_kind}")
    print(f"encoder_fingerprint={bank.encoder_fingerprint}")
    print(f"content_hash={bank.content_hash}")


if __name__ == "__main__":
    main()
