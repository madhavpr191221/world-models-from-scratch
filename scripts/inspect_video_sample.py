"""
Inspect one Something-Something V2 clip.

This script is a smoke test for the video data path:

- load the JSON metadata
- resolve the matching `.webm` file
- decode a short clip
- sample a fixed number of frames
- save the sampled frames and a contact sheet

Usage:
    uv run python scripts/inspect_video_sample.py

Outputs:
    logs/video_smoke_test/
        clip_<video_id>/
            frame_00.png
            frame_01.png
            ...
            contact_sheet.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from jepa_world_models.data.video import SomethingSomethingV2Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one Something-Something V2 clip.")
    parser.add_argument(
        "--split",
        default="data/20bn-something-something-download-package-labels/labels/train.json",
        help="Path to the split JSON file.",
    )
    parser.add_argument(
        "--labels",
        default="data/20bn-something-something-download-package-labels/labels/labels.json",
        help="Path to labels.json.",
    )
    parser.add_argument(
        "--video-root",
        default="data/something_v2/20bn-something-something-v2",
        help="Folder containing <video_id>.webm files.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/video_smoke_test",
        help="Where to write sampled frames.",
    )
    parser.add_argument("--index", type=int, default=None, help="Dataset index to inspect.")
    parser.add_argument("--video-id", default=None, help="Specific video id to inspect.")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample.")
    parser.add_argument("--image-size", type=int, default=96, help="Resize each frame to this size.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used when picking a sample.")
    return parser.parse_args()


def save_frame_grid(clip: torch.Tensor, out_path: Path) -> None:
    frames = []
    for frame in clip:
        tensor = frame.clamp(0.0, 1.0)
        frames.append(Image.fromarray((tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy())))

    if not frames:
        raise ValueError("No frames to save")

    widths = [frame.width for frame in frames]
    heights = [frame.height for frame in frames]
    canvas = Image.new("RGB", (sum(widths), max(heights)), color=(12, 12, 16))
    x = 0
    for idx, frame in enumerate(frames):
        canvas.paste(frame, (x, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((x + 4, 4), f"{idx}", fill=(255, 255, 255))
        x += frame.width
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = SomethingSomethingV2Dataset(
        split_path=args.split,
        labels_path=args.labels,
        video_root=args.video_root,
        image_size=args.image_size,
        num_frames=args.num_frames,
    )

    if args.video_id is not None:
        record = next((item for item in dataset.records if item.video_id == args.video_id), None)
        if record is None:
            raise SystemExit(f"Unknown video id: {args.video_id}")
        index = dataset.records.index(record)
    elif args.index is not None:
        index = args.index
    else:
        index = random.randrange(len(dataset))

    record = dataset.get_record(index)
    clip, label_id, label_text, video_id = dataset.load_clip(record.video_id)

    output_dir = Path(args.output_dir) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video id: {video_id}")
    print(f"Label id: {label_id}")
    print(f"Template: {label_text}")
    print(f"Instance label: {record.instance_label}")
    print(f"Clip shape: {tuple(clip.shape)}")
    print(f"Video path: {record.path}")

    for frame_index, frame in enumerate(clip):
        frame_path = output_dir / f"frame_{frame_index:02d}.png"
        Image.fromarray((frame.clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0).cpu().numpy())).save(frame_path)

    contact_sheet = output_dir / "contact_sheet.png"
    save_frame_grid(clip, contact_sheet)
    print(f"Saved frames to: {output_dir}")
    print(f"Saved contact sheet to: {contact_sheet}")


if __name__ == "__main__":
    main()
