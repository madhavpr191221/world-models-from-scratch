from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_world_models.analysis.video_rollout_horizon_sweep import run_rollout_horizon_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a rollout horizon sweep for a trained latent world model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained latent world-model checkpoint.")
    parser.add_argument("--data-root", type=str, default="data", help="Data root containing Something-Something V2.")
    parser.add_argument("--source-split", type=str, default="train", help="Source split used to sample videos.")
    parser.add_argument("--subset-size", type=int, default=128, help="Number of usable videos to sample.")
    parser.add_argument("--image-size", type=int, default=224, help="Frame resize target.")
    parser.add_argument("--context-seconds", type=float, default=4.0, help="Observed context duration in seconds.")
    parser.add_argument(
        "--future-seconds-grid",
        nargs="+",
        type=float,
        required=True,
        help="Future horizons to sweep, for example 1.5 3.0 4.5 6.0.",
    )
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Target sampling rate in frames per second.")
    parser.add_argument("--feature-batch-size", type=int, default=1, help="Batch size used for latent extraction.")
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
        default="logs/video_world_model/horizon_sweep",
        help="Directory used for sweep reports and plots.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override, for example cuda or cpu.")
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=1,
        help="Maximum number of batches to evaluate per horizon.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_rollout_horizon_sweep(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        source_split=args.source_split,
        subset_size=args.subset_size,
        image_size=args.image_size,
        context_seconds=args.context_seconds,
        future_seconds_grid=args.future_seconds_grid,
        sample_fps=args.sample_fps,
        feature_batch_size=args.feature_batch_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_limit=args.batch_limit,
    )
    summary_path = Path(args.output_dir) / "horizon_sweep_summary.json"
    print(f"Wrote horizon sweep to {args.output_dir}")
    print(f"summary_path={summary_path}")
    print(json.dumps(result.to_json(), indent=2))


if __name__ == "__main__":
    main()

