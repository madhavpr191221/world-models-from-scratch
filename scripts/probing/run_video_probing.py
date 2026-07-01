"""
Run the first video SSL probe on Something-Something V2.

This script tests whether frozen clip features already contain temporal order.
The primary task is forward-vs-reversed classification on short clips.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from jepa_world_models.analysis.video_probing import run_temporal_direction_probe
from jepa_world_models.data.video import SomethingSomethingV2Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen temporal direction probe.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/vicreg/best.pt",
        help="Path to a VicReg checkpoint.",
    )
    parser.add_argument(
        "--train-split",
        default="data/20bn-something-something-download-package-labels/labels/train.json",
        help="Path to train.json.",
    )
    parser.add_argument(
        "--validation-split",
        default="data/20bn-something-something-download-package-labels/labels/validation.json",
        help="Path to validation.json.",
    )
    parser.add_argument(
        "--labels",
        default="data/20bn-something-something-download-package-labels/labels/labels.json",
        help="Path to labels.json.",
    )
    parser.add_argument(
        "--video-root",
        default="data/something_v2/20bn-something-something-v2",
        help="Folder containing .webm clips.",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu; defaults to auto-detect.")
    parser.add_argument("--max-train-clips", type=int, default=256)
    parser.add_argument("--max-validation-clips", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        default="logs/video_probing",
        help="Directory for video probing outputs.",
    )
    parser.add_argument(
        "--cache-dir",
        default="logs/video_probing/cache",
        help="Directory for cached video features.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute cached clip features even if they already exist.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    train_dataset = SomethingSomethingV2Dataset(
        split_path=args.train_split,
        labels_path=args.labels,
        video_root=args.video_root,
        image_size=args.image_size,
        num_frames=args.num_frames,
    )
    validation_dataset = SomethingSomethingV2Dataset(
        split_path=args.validation_split,
        labels_path=args.labels,
        video_root=args.video_root,
        image_size=args.image_size,
        num_frames=args.num_frames,
    )

    output = run_temporal_direction_probe(
        checkpoint_path=args.checkpoint,
        train_dataset=train_dataset,
        test_dataset=validation_dataset,
        device=args.device,
        max_train_clips=args.max_train_clips,
        max_test_clips=args.max_validation_clips,
        seed=args.seed,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [asdict(row) for row in output["results"]]
    summary = {
        "checkpoint": output["checkpoint"],
        "device": output["device"],
        "image_size": output["image_size"],
        "num_frames": output["num_frames"],
        "cache_dir": args.cache_dir,
        "train_clips": len(output["train_split"].selected_clip_ids),
        "validation_clips": len(output["test_split"].selected_clip_ids),
        "selected_train_video_ids": output["train_split"].selected_clip_ids,
        "selected_validation_video_ids": output["test_split"].selected_clip_ids,
        "results": results,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _write_csv(output_dir / "results.csv", results)

    print(f"Wrote video probing artifacts to {output_dir}")
    print(f"Feature cache dir: {args.cache_dir}")
    for row in results:
        print(
            f"{row['feature_view']:>8} | task={row['task']} | "
            f"train_acc={row['train_accuracy']:.4f} | test_acc={row['test_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
