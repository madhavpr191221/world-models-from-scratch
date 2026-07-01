from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_world_models.analysis.video_world_model_reporting import write_rollout_validation_plots
from jepa_world_models.analysis.video_world_model_validation import (
    build_rollout_validation_report,
    write_rollout_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent rollout validation for the video world model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained latent world-model checkpoint.")
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
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="logs/video_world_model/cache",
        help="Directory used for cached latent sequences.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/video_world_model/rollout_validation.json",
        help="Path to the validation report.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on the number of validation batches to process.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override, for example cuda or cpu.")
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default=None,
        help="Directory for rollout validation plots. Defaults to <output-dir>/plots.",
    )
    return parser.parse_args()


def _seconds_to_frame_count(seconds: float, sample_fps: float) -> int:
    frames = max(2, int(round(seconds * sample_fps)))
    return frames + (frames % 2)


def main() -> None:
    args = parse_args()
    context_frames = _seconds_to_frame_count(args.context_seconds, args.sample_fps)
    future_frames = _seconds_to_frame_count(args.future_seconds, args.sample_fps)
    total_frames = context_frames + future_frames
    report = build_rollout_validation_report(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        source_split=args.source_split,
        subset_size=args.subset_size,
        image_size=args.image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
        sample_fps=args.sample_fps,
        feature_batch_size=args.feature_batch_size,
        seed=args.seed,
        batch_limit=args.max_batches,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    output = write_rollout_validation_report(args.output, report)
    plot_dir = Path(args.plot_output_dir) if args.plot_output_dir is not None else Path(args.output).parent / "plots"
    plot_paths = write_rollout_validation_plots(report, plot_dir)
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(f"Wrote rollout validation report to {output}")
    print(f"num_samples={payload['num_samples']}")
    print(f"context_frames={payload['context_frames']}")
    print(f"future_frames={payload['future_frames']}")
    print(f"total_frames={payload['total_frames']}")
    print(f"latent_dim={payload['latent_dim']}")
    print(f"device={payload['device']}")
    print(f"predictor_mode={payload['predictor_mode']}")
    print(f"context_lag_steps={payload['context_lag_steps']}")
    print(f"rollout_validation_plot={plot_paths['rollout_validation']}")
    print(f"rollout_spectrum_plot={plot_paths['rollout_spectrum']}")


if __name__ == "__main__":
    main()
