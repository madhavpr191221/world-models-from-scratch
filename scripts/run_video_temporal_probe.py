from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_world_models.analysis.video_temporal_probe import train_forward_reverse_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a forward-vs-reversed temporal probe.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the frozen VICReg checkpoint.")
    parser.add_argument("--data-root", type=str, default="data", help="Data root that contains Something-Something V2.")
    parser.add_argument("--source-split", type=str, default="train", help="Source split used to sample the usable subset.")
    parser.add_argument("--image-size", type=int, default=96, help="Frame resize target.")
    parser.add_argument("--num-frames", type=int, default=32, help="Frames sampled per clip.")
    parser.add_argument("--subset-size", type=int, default=2000, help="Total number of usable source videos to sample before automatic 70/20/10 split.")
    parser.add_argument("--batch-size", type=int, default=64, help="Probe batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of probe epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the probe.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subset selection and training.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/video_temporal_probe",
        help="Directory for probe checkpoints and reports.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/video_temporal_probe/result.json",
        help="Path for the JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_forward_reverse_probe(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        source_split=args.source_split,
        image_size=args.image_size,
        num_frames=args.num_frames,
        subset_size=args.subset_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")

    print(f"Wrote temporal probe report to {output}")
    print(f"train_accuracy={result.train_accuracy:.4f}")
    print(f"val_accuracy={result.val_accuracy:.4f}")
    print(f"test_accuracy={result.test_accuracy:.4f}")
    print(f"feature_shape={result.feature_shape}")
    print(f"best_checkpoint={result.checkpoint_path}")


if __name__ == "__main__":
    main()
