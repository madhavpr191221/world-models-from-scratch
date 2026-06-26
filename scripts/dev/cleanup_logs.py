from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean generated log directories.")
    parser.add_argument(
        "--root",
        type=str,
        default="logs",
        help="Root logs directory to clean.",
    )
    parser.add_argument(
        "--keep",
        nargs="*",
        default=["videomae_large", "video_world_model"],
        help="Directory names to keep under the logs root.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the matching directories. Without this flag, the script is a dry run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Logs root does not exist: {root}")

    keep = {".gitignore", *args.keep}
    targets: list[Path] = []
    for child in root.iterdir():
        if child.name in keep:
            continue
        targets.append(child)

    if not targets:
        print(f"No cleanup needed under {root}")
        return

    print(f"Logs root: {root}")
    print(f"Keeping: {sorted(keep)}")
    print("Targets:")
    for target in targets:
        print(f"  {target}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to delete the targets.")
        return

    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    print(f"Deleted {len(targets)} item(s) under {root}")


if __name__ == "__main__":
    main()
