from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jepa_world_models.analysis.video_latent_models import EncoderSpec, build_video_encoder
from jepa_world_models.analysis.video_world_model import train_video_world_model
from jepa_world_models.analysis.video_world_model_plots import write_video_world_model_plots
from jepa_world_models.analysis.video_world_model_validation import (
    build_rollout_validation_report,
    write_rollout_validation_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the encoder/predictor latent-dynamics sweep.")
    parser.add_argument("--data-root", type=str, default="data", help="Data root containing Something-Something V2.")
    parser.add_argument("--source-split", type=str, default="train", help="Split used to sample clips.")
    parser.add_argument(
        "--validation-source-split",
        type=str,
        default=None,
        help="Optional physical validation split folder name. Omit to use an internal split.",
    )
    parser.add_argument(
        "--test-source-split",
        type=str,
        default=None,
        help="Optional physical test split folder name. Omit to use an internal split.",
    )
    parser.add_argument("--subset-size", type=int, default=256, help="Number of usable videos to sample per run.")
    parser.add_argument("--image-size", type=int, default=224, help="Frame resize target.")
    parser.add_argument("--context-seconds", type=float, default=4.0, help="Observed context duration in seconds.")
    parser.add_argument("--future-seconds", type=float, default=2.0, help="Prediction horizon in seconds.")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Target sampling rate in frames per second.")
    parser.add_argument("--feature-batch-size", type=int, default=1, help="Batch size used for latent extraction.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers used for latent extraction and predictor training.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size used to train the predictor.")
    parser.add_argument("--epochs", type=int, default=3, help="Predictor epochs per run.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the predictor.")
    parser.add_argument("--hidden-dim", type=int, default=192, help="Predictor hidden width.")
    parser.add_argument("--num-layers", type=int, default=4, help="Predictor depth.")
    parser.add_argument("--num-heads", type=int, default=4, help="Predictor attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Predictor dropout rate.")
    parser.add_argument(
        "--context-lag-steps",
        type=int,
        default=None,
        help="Number of recent latent steps fed to the predictor. Omit to use the full context window.",
    )
    parser.add_argument(
        "--context-lag-grid",
        nargs="+",
        type=int,
        default=None,
        help="Optional lag sweep grid. Each value is run independently as context_lag_steps.",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=["swin", "timesformer", "vivit", "hybrid"],
        choices=("swin", "timesformer", "vivit", "hybrid"),
        help="Encoder families to sweep.",
    )
    parser.add_argument(
        "--predictors",
        nargs="+",
        default=["one_lag_mlp", "causal_transformer", "gru", "tcn", "mamba", "cross_attention"],
        choices=("one_lag_mlp", "causal_transformer", "gru", "tcn", "mamba", "cross_attention"),
        help="Predictor families to sweep.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="logs/video_latent_dynamics",
        help="Root folder for pair-specific outputs.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="logs/video_latent_dynamics/cache",
        help="Shared cache root for latent sequence banks.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, for example cuda or cpu.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--encoder-pretrained",
        action="store_true",
        help="Use pretrained torchvision weights for the Swin encoder family when available.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG plot generation for the sweep runs.",
    )
    return parser


def _seconds_to_frame_count(seconds: float, sample_fps: float) -> int:
    frames = max(2, int(round(seconds * sample_fps)))
    return frames + (frames % 2)


def _save_encoder_checkpoint(
    *,
    encoder_name: str,
    output_dir: Path,
    latent_dim: int,
    pretrained: bool,
) -> Path:
    encoder = build_video_encoder(
        EncoderSpec(
            name=encoder_name,
            latent_dim=latent_dim,
            pretrained=pretrained,
        )
    )
    encoder_path = output_dir / f"encoder_{encoder_name}.pt"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "encoder_name": encoder_name,
                "latent_dim": latent_dim,
                "tubelet_size": getattr(encoder, "tubelet_size", 2),
                "pretrained": pretrained,
                "variant": "t",
            },
            "model_state": encoder.state_dict(),
        },
        encoder_path,
    )
    return encoder_path


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    cache_root = Path(args.cache_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    context_frames = _seconds_to_frame_count(args.context_seconds, args.sample_fps)
    future_frames = _seconds_to_frame_count(args.future_seconds, args.sample_fps)
    total_frames = context_frames + future_frames
    lag_grid = list(args.context_lag_grid) if args.context_lag_grid is not None else [args.context_lag_steps]

    summary: list[dict[str, object]] = []
    for encoder_name in args.encoders:
        pair_results: dict[str, object] = {"encoder": encoder_name, "runs": []}
        for predictor_name in args.predictors:
            for lag_steps in lag_grid:
                lag_name = "all" if lag_steps is None else f"lag{lag_steps}"
                pair_name = f"{encoder_name}-{predictor_name}-{lag_name}"
                pair_dir = output_root / pair_name
                pair_dir.mkdir(parents=True, exist_ok=True)
                encoder_ckpt = _save_encoder_checkpoint(
                    encoder_name=encoder_name,
                    output_dir=pair_dir,
                    latent_dim=192,
                    pretrained=bool(args.encoder_pretrained and encoder_name == "swin"),
                )
                result = train_video_world_model(
                    checkpoint_path=encoder_ckpt,
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
                    predictor_name=predictor_name,
                    predictor_mode="context",
                    context_lag_steps=lag_steps,
                    seed=args.seed,
                    cache_dir=cache_root,
                    output_dir=pair_dir,
                    device=args.device,
                )
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
                    cache_dir=cache_root,
                    device=args.device,
                )
                validation_path = write_rollout_validation_report(pair_dir / "rollout_validation.json", validation_report)
                if not args.skip_plots:
                    plot_paths = write_video_world_model_plots(result, pair_dir / "plots", validation_report=validation_report)
                else:
                    plot_paths = {}
                run_payload = result.to_json()
                run_payload["encoder_checkpoint_path"] = str(encoder_ckpt)
                run_payload["pair_name"] = pair_name
                run_payload["validation_report_path"] = str(validation_path)
                run_payload["plot_paths"] = {key: str(value) for key, value in plot_paths.items()}
                (pair_dir / "result.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
                pair_results["runs"].append(
                    {
                        "pair_name": pair_name,
                        "encoder_checkpoint_path": str(encoder_ckpt),
                        "checkpoint_path": result.checkpoint_path,
                        "checkpoint_dir": result.checkpoint_dir,
                        "context_lag_steps": result.context_lag_steps,
                        "train_loss": result.train_loss,
                        "val_loss": result.val_loss,
                        "test_loss": result.test_loss,
                        "test_verdict": result.baseline_verdict["test"]["verdict"],
                        "validation_report_path": str(validation_path),
                    }
                )
                print(f"[{pair_name}] train={result.train_loss:.6f} val={result.val_loss:.6f} test={result.test_loss:.6f}")
        summary.append(pair_results)

    (output_root / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote sweep summary to {output_root / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
