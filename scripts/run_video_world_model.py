from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from jepa_world_models.analysis.common import resolve_device
from jepa_world_models.analysis.video_world_model import train_video_world_model
from jepa_world_models.analysis.video_world_model_plots import write_video_world_model_plots
from jepa_world_models.analysis.video_world_model_validation import (
    build_rollout_validation_report,
    write_rollout_validation_report,
)

warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self\.use_nested_tensor is False because encoder_layer\.norm_first was True",
    category=UserWarning,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a JEPA-style latent video world model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the frozen VideoMAE checkpoint.")
    parser.add_argument("--data-root", type=str, default="data", help="Data root containing Something-Something V2.")
    parser.add_argument("--source-split", type=str, default="train", help="Source split used to sample videos.")
    parser.add_argument(
        "--validation-source-split",
        type=str,
        default=None,
        help="Optional physical validation split folder name, for example validation.",
    )
    parser.add_argument(
        "--test-source-split",
        type=str,
        default=None,
        help="Optional physical test split folder name, for example test.",
    )
    parser.add_argument("--subset-size", type=int, default=128, help="Number of usable videos to sample.")
    parser.add_argument("--image-size", type=int, default=224, help="Frame resize target.")
    parser.add_argument("--context-seconds", type=float, default=5.0, help="Observed context duration in seconds.")
    parser.add_argument("--future-seconds", type=float, default=1.5, help="Prediction horizon in seconds.")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Target sampling rate in frames per second.")
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=8,
        help="Batch size used while extracting cached latent sequences.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers used for latent extraction and predictor training.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used to train the predictor.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of predictor epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the predictor.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden width of the predictor.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer encoder layers.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Predictor dropout rate.")
    parser.add_argument(
        "--predictor-name",
        type=str,
        default="causal_transformer",
        choices=("causal_transformer", "mamba", "cross_attention"),
        help="Temporal predictor family to train.",
    )
    parser.add_argument(
        "--predictor-mode",
        type=str,
        default="context",
        choices=("context", "one-lag"),
        help="Temporal predictor mode: multi-context or single-lag autoregression.",
    )
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
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating PNG plots after training.",
    )
    parser.add_argument(
        "--skip-rollout-validation",
        action="store_true",
        help="Skip the post-training rollout validation pass.",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default=None,
        help="Directory for generated PNG plots. Defaults to <output-dir>/plots.",
    )
    parser.add_argument(
        "--validation-output",
        type=str,
        default=None,
        help="Path to the rollout validation JSON. Defaults to <output-dir>/rollout_validation.json.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect a torch.profiler trace during the training run.",
    )
    parser.add_argument(
        "--profile-output-dir",
        type=str,
        default="logs/video_world_model/profile",
        help="Directory for profiler artifacts.",
    )
    parser.add_argument(
        "--profile-trace-name",
        type=str,
        default="video_world_model_trace.json",
        help="Filename for the exported Chrome trace.",
    )
    parser.add_argument(
        "--profile-table-sort-by",
        type=str,
        default="self_cuda_time_total",
        help="Profiler table column used for sorting.",
    )
    parser.add_argument(
        "--profile-table-row-limit",
        type=int,
        default=30,
        help="Maximum number of rows to print in the profiler table.",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default="logs/video_world_model/profile/profile_summary.json",
        help="Path to a JSON summary of the profiler run.",
    )
    return parser


def _seconds_to_frame_count(seconds: float, sample_fps: float) -> int:
    frames = max(2, int(round(seconds * sample_fps)))
    return frames + (frames % 2)


def _run_training(args: argparse.Namespace):
    context_frames = _seconds_to_frame_count(args.context_seconds, args.sample_fps)
    future_frames = _seconds_to_frame_count(args.future_seconds, args.sample_fps)
    total_frames = context_frames + future_frames

    result = train_video_world_model(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        source_split=args.source_split,
        validation_source_split=args.validation_source_split,
        test_source_split=args.test_source_split,
        subset_size=args.subset_size,
        image_size=args.image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
        sample_fps=args.sample_fps,
        feature_batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        predictor_name=args.predictor_name,
        predictor_mode=args.predictor_mode,
        seed=args.seed,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
    return result, context_frames, future_frames, total_frames


def _write_summary(output: Path, args: argparse.Namespace, result, context_frames: int, future_frames: int, total_frames: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    print(f"Wrote world model report to {output}")
    print(f"train_loss={result.train_loss:.6f}")
    print(f"val_loss={result.val_loss:.6f}")
    print(f"test_loss={result.test_loss:.6f}")
    print(f"train_metrics={result.train_metrics}")
    print(f"test_metrics={result.test_metrics}")
    print(f"baseline_metrics={result.baseline_metrics}")
    print(f"baseline_verdict={result.baseline_verdict}")
    test_verdict = result.baseline_verdict["test"]
    print(f"baseline_test_verdict={test_verdict['verdict']}")
    print(f"baseline_test_normalized_latent_mse_better={test_verdict['normalized_latent_mse']['better_than_best_baseline']}")
    print(f"baseline_test_latent_mse_better={test_verdict['latent_mse']['better_than_best_baseline']}")
    print(f"baseline_test_cosine_similarity_better={test_verdict['cosine_similarity']['better_than_best_baseline']}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"encoder_name={result.encoder_name}")
    print(f"predictor_name={result.predictor_name}")
    print(f"cache_path={result.cache_path}")
    print(f"cache_manifest_path={result.cache_manifest_path}")
    print(f"predictor_mode={result.predictor_mode}")
    print(f"context_seconds={args.context_seconds:g}")
    print(f"future_seconds={args.future_seconds:g}")
    print(f"sample_fps={args.sample_fps:g}")
    print(f"derived_context_frames={context_frames}")
    print(f"derived_future_frames={future_frames}")
    print(f"derived_total_frames={total_frames}")
    print(f"predictions_path={result.predictions_path}")


def _run_profiled_training(args: argparse.Namespace, device_obj: torch.device):
    activities = [ProfilerActivity.CPU]
    if device_obj.type == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    profile_dir = Path(args.profile_output_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    trace_path = profile_dir / args.profile_trace_name

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("train_video_world_model"):
            result, context_frames, future_frames, total_frames = _run_training(args)

    sort_by = args.profile_table_sort_by
    if sort_by.startswith("self_cuda") and ProfilerActivity.CUDA not in activities:
        sort_by = "self_cpu_time_total"

    prof.export_chrome_trace(str(trace_path))
    print(prof.key_averages().table(sort_by=sort_by, row_limit=args.profile_table_row_limit))

    profile_output = Path(args.profile_output)
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profile_summary = {
        "checkpoint_path": result.checkpoint_path,
        "trace_path": str(trace_path),
        "sort_by": sort_by,
        "context_seconds": args.context_seconds,
        "future_seconds": args.future_seconds,
        "sample_fps": args.sample_fps,
        "derived_context_frames": context_frames,
        "derived_future_frames": future_frames,
        "derived_total_frames": total_frames,
        "device": str(device_obj),
        "train_loss": result.train_loss,
        "val_loss": result.val_loss,
        "test_loss": result.test_loss,
    }
    profile_output.write_text(json.dumps(profile_summary, indent=2), encoding="utf-8")
    print(f"Trace written to {trace_path}")
    print(f"Profile summary written to {profile_output}")
    return result, context_frames, future_frames, total_frames


def _postprocess_outputs(args: argparse.Namespace, result, context_frames: int, future_frames: int, total_frames: int) -> None:
    plot_dir = Path(args.plot_output_dir) if args.plot_output_dir is not None else Path(args.output_dir) / "plots"
    validation_report = None
    validation_path: Path | None = None

    if not args.skip_rollout_validation:
        validation_path = Path(args.validation_output) if args.validation_output is not None else Path(args.output_dir) / "rollout_validation.json"
        validation_report = build_rollout_validation_report(
            checkpoint_path=result.checkpoint_path,
            data_root=args.data_root,
            source_split=args.test_source_split or args.source_split,
            subset_size=args.subset_size,
            image_size=args.image_size,
            total_frames=total_frames,
            context_frames=context_frames,
            future_frames=future_frames,
            sample_fps=args.sample_fps,
            feature_batch_size=args.feature_batch_size,
            seed=args.seed,
            batch_limit=None,
            cache_dir=args.cache_dir,
            device=args.device,
        )
        validation_path = write_rollout_validation_report(validation_path, validation_report)
        print(f"Wrote rollout validation report to {validation_path}")

    if not args.skip_plots:
        plot_paths = write_video_world_model_plots(result, plot_dir, validation_report=validation_report)
        for name, path in plot_paths.items():
            print(f"{name}_plot={path}")


def main() -> None:
    args = build_parser().parse_args()
    device_obj = resolve_device(args.device)

    if args.profile:
        result, context_frames, future_frames, total_frames = _run_profiled_training(args, device_obj)
    else:
        result, context_frames, future_frames, total_frames = _run_training(args)

    _write_summary(Path(args.output), args, result, context_frames, future_frames, total_frames)
    _postprocess_outputs(args, result, context_frames, future_frames, total_frames)


if __name__ == "__main__":
    main()
