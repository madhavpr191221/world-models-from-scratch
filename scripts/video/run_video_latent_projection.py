from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_world_models.analysis.video_latent_projection import (
    build_latent_projection_engine,
    save_latent_projection_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project latent video trajectories into 2D.")
    parser.add_argument("--world-model-checkpoint", type=str, required=True, help="Path to the trained latent world-model checkpoint.")
    parser.add_argument("--data-root", type=str, default="data", help="Root containing Something-Something V2.")
    parser.add_argument("--source-split", type=str, default="train", help="Dataset split to sample from.")
    parser.add_argument("--subset-size", type=int, default=256, help="Number of videos to sample into the latent bank.")
    parser.add_argument("--image-size", type=int, default=224, help="Frame resize target used for latent extraction.")
    parser.add_argument("--context-seconds", type=float, default=5.0, help="Observed context duration in seconds.")
    parser.add_argument("--future-seconds", type=float, default=1.5, help="Prediction horizon in seconds.")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Sampling rate in frames per second.")
    parser.add_argument("--feature-batch-size", type=int, default=1, help="Batch size used while building the latent cache.")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to project and report.")
    parser.add_argument("--projection-method", type=str, default="pca", choices=["pca", "tsne"], help="2D projection method.")
    parser.add_argument("--background-sample-size", type=int, default=512, help="How many background clips to include in the rendered cloud.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and projection.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override, for example cuda or cpu.")
    parser.add_argument("--cache-dir", type=str, default="logs/video_world_model/cache", help="Directory used for the cached latent bank.")
    parser.add_argument("--output-dir", type=str, default="logs/video_latent_projection", help="Directory for reports and artifacts.")
    parser.add_argument("--output", type=str, default="logs/video_latent_projection/result.json", help="Path to the JSON report.")
    return parser


def _seconds_to_frame_count(seconds: float, sample_fps: float) -> int:
    frames = max(2, int(round(seconds * sample_fps)))
    return frames + (frames % 2)


def main() -> None:
    args = build_parser().parse_args()
    context_frames = _seconds_to_frame_count(args.context_seconds, args.sample_fps)
    future_frames = _seconds_to_frame_count(args.future_seconds, args.sample_fps)
    total_frames = context_frames + future_frames

    engine = build_latent_projection_engine(
        world_model_checkpoint=args.world_model_checkpoint,
        data_root=args.data_root,
        source_split=args.source_split,
        subset_size=args.subset_size,
        image_size=args.image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
        feature_batch_size=args.feature_batch_size,
        cache_dir=args.cache_dir,
        seed=args.seed,
        device=args.device,
    )
    result = engine.analyze_index(
        args.index,
        projection_method=args.projection_method,
        background_sample_size=args.background_sample_size,
        seed=args.seed,
    )

    output = Path(args.output)
    save_latent_projection_report(output, result)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Wrote latent projection report to {output}")
    print(f"projection_method={result.projection_method}")
    print(f"train_metrics={result.metrics}")
    print(f"baseline_metrics={result.baseline_metrics}")
    print(f"context_seconds={args.context_seconds:g}")
    print(f"future_seconds={args.future_seconds:g}")
    print(f"sample_fps={args.sample_fps:g}")
    print(f"derived_context_frames={context_frames}")
    print(f"derived_future_frames={future_frames}")
    print(f"derived_total_frames={total_frames}")
    print(f"video_url={result.query['video_url']}")


if __name__ == "__main__":
    main()
