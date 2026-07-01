from __future__ import annotations

import argparse
import shutil
from pathlib import Path


VIDEO_EXTENSIONS = {".webm", ".mp4", ".avi", ".mov", ".mkv"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy a small set of demo videos into a single folder.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="data/demo_videos")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument(
        "--preferred-subdir",
        type=str,
        default="kinetics400",
        help="Optional subdirectory to search first before scanning the whole data tree.",
    )
    return parser


def iter_video_files(root: Path, preferred_subdir: str) -> list[Path]:
    preferred = root / preferred_subdir
    candidates: list[Path] = []
    if preferred.exists():
        candidates.extend(sorted(p for p in preferred.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS))
    candidates.extend(sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = iter_video_files(data_root, args.preferred_subdir)
    if not videos:
        raise SystemExit(f"No video files found under {data_root}")

    copied = 0
    for src in videos:
        dst = output_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        copied += 1
        print(f"Copied {src} -> {dst}")
        if copied >= args.count:
            break

    print(f"Copied {copied} videos into {output_dir}")


if __name__ == "__main__":
    main()
