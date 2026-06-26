from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_world_models.analysis.video_dynamics import build_video_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video latent-dynamics demo payload.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained VicReg checkpoint.")
    parser.add_argument("--data-root", type=str, default="data", help="Data root with Something-Something V2 files.")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to analyze.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of neighbors to return.")
    parser.add_argument("--bank-size", type=int, default=512, help="Number of clips to sample into the bank.")
    parser.add_argument(
        "--output",
        type=str,
        default="logs/video_dynamics/demo.json",
        help="Where to write the demo payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = build_video_engine(
        args.checkpoint,
        data_root=args.data_root,
        bank_size=args.bank_size,
    )
    output = engine.write_demo_payload(args.index, args.output, top_k=args.top_k)
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(f"Wrote demo payload to {output}")
    print(f"Query: {payload['query']['video_id']} -> {payload['query']['label_text']}")
    print(f"Frames: {len(payload['trajectory'])}")
    print(f"Global neighbors: {len(payload['global_neighbors'])}")


if __name__ == "__main__":
    main()
