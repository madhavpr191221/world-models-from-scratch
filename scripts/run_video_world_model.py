from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_world_models.analysis.video_world_model import train_video_world_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a JEPA-style latent video world model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the frozen VideoMAE checkpoint.")
    parser.add_argument("--data-root", type=str, default="data", help="Data root containing Something-Something V2.")
    parser.add_argument("--source-split", type=str, default="train", help="Source split used to sample videos.")
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
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used to train the predictor.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of predictor epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the predictor.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden width of the predictor.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer encoder layers.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Predictor dropout rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="logs/video_world_model/cache",
        help="Directory used for cached latent sequences.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/video_world_model",
        help="Directory used for checkpoints and reports.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override, for example cuda or cpu.")
    parser.add_argument(
        "--output",
        type=str,
        default="logs/video_world_model/result.json",
        help="Path to the JSON summary report.",
    )
    return parser


def _seconds_to_frame_count(seconds: float, sample_fps: float) -> int:
    frames = max(2, int(round(seconds * sample_fps)))
    return frames + (frames % 2)


def main() -> None:
    args = build_parser().parse_args()
    context_frames = _seconds_to_frame_count(args.context_seconds, args.sample_fps)
    future_frames = _seconds_to_frame_count(args.future_seconds, args.sample_fps)
    total_frames = context_frames + future_frames

    result = train_video_world_model(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        source_split=args.source_split,
        subset_size=args.subset_size,
        image_size=args.image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
        feature_batch_size=args.feature_batch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        seed=args.seed,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        device=args.device,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    print(f"Wrote world model report to {output}")
    print(f"train_loss={result.train_loss:.6f}")
    print(f"val_loss={result.val_loss:.6f}")
    print(f"test_loss={result.test_loss:.6f}")
    print(f"train_metrics={result.train_metrics}")
    print(f"test_metrics={result.test_metrics}")
    print(f"baseline_metrics={result.baseline_metrics}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"context_seconds={args.context_seconds:g}")
    print(f"future_seconds={args.future_seconds:g}")
    print(f"sample_fps={args.sample_fps:g}")
    print(f"derived_context_frames={context_frames}")
    print(f"derived_future_frames={future_frames}")
    print(f"derived_total_frames={total_frames}")
    print(f"predictions_path={result.predictions_path}")


if __name__ == "__main__":
    main()
