from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class SplitManifest:
    split: str
    count: int
    source_root: str
    destination_root: str
    created_at: str
    video_ids: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move Something-Something V2 videos into train/validation/test folders.")
    parser.add_argument(
        "--source-root",
        type=str,
        default="data/something_v2/20bn-something-something-v2",
        help="Flat source directory containing the .webm videos.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/data_videos",
        help="Destination root for train/validation/test folders.",
    )
    parser.add_argument("--train-count", type=int, default=50000, help="Number of videos to move into train.")
    parser.add_argument("--validation-count", type=int, default=10000, help="Number of videos to move into validation.")
    parser.add_argument("--test-count", type=int, default=5000, help="Number of videos to move into test.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected split without moving files.")
    return parser


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    source_root = _safe_resolve(Path(args.source_root))
    output_root = _safe_resolve(Path(args.output_root))

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_files = sorted(source_root.glob("*.webm"))
    total_required = args.train_count + args.validation_count + args.test_count
    if len(source_files) < total_required:
        raise ValueError(f"Need at least {total_required} videos, found {len(source_files)}.")

    rng = random.Random(args.seed)
    shuffled = source_files[:]
    rng.shuffle(shuffled)

    train_files = shuffled[: args.train_count]
    validation_files = shuffled[args.train_count : args.train_count + args.validation_count]
    test_files = shuffled[args.train_count + args.validation_count : total_required]

    split_map = {
        "train": train_files,
        "validation": validation_files,
        "test": test_files,
    }

    summary: dict[str, object] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "created_at": _utc_now_iso(),
        "splits": {},
    }

    for split, files in split_map.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        video_ids = [path.stem for path in files]
        manifest = SplitManifest(
            split=split,
            count=len(files),
            source_root=str(source_root),
            destination_root=str(split_dir),
            created_at=_utc_now_iso(),
            video_ids=video_ids,
        )
        summary["splits"][split] = asdict(manifest)
        _write_json(output_root / f"{split}_manifest.json", asdict(manifest))
        if args.dry_run:
            continue
        for src in files:
            dst = split_dir / src.name
            if dst.exists():
                continue
            shutil.move(str(src), str(dst))

    _write_json(output_root / "split_summary.json", summary)
    print(json.dumps({k: len(v) for k, v in split_map.items()}, indent=2))
    print(f"Split summary written to {output_root / 'split_summary.json'}")


if __name__ == "__main__":
    main()
