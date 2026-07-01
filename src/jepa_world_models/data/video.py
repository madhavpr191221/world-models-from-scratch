from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T


@dataclass(slots=True)
class VideoRecord:
    video_id: str
    instance_label: str
    template_text: str
    label_id: int
    path: Path
    placeholders: tuple[str, ...] = ()


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_label_map(labels_path: str | Path) -> tuple[dict[str, int], tuple[str, ...]]:
    raw = _load_json(labels_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected labels.json to contain a dict, got {type(raw).__name__}")

    items = sorted(((str(label), int(label_id)) for label, label_id in raw.items()), key=lambda item: item[1])
    label_to_id = {label: label_id for label, label_id in items}
    classes = tuple(label for label, _ in items)
    return label_to_id, classes


def _normalize_template(text: str) -> str:
    return " ".join(text.replace("[", "").replace("]", "").split())


def _load_split_records(split_path: str | Path) -> list[dict[str, Any]]:
    raw = _load_json(split_path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected {split_path} to contain a list, got {type(raw).__name__}")
    return raw


def _decode_video_frames(path: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_image().convert("RGB"))
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames


def _sample_indices(num_frames: int, total_frames: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if total_frames == 1:
        return [0] * num_frames
    indices = torch.linspace(0, total_frames - 1, steps=num_frames).round().long().tolist()
    return [int(index) for index in indices]


def _frames_to_tensor(
    frames: list[Image.Image],
    *,
    image_size: int,
) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    tensors = [transform(frame) for frame in frames]
    return torch.stack(tensors, dim=0)


class SomethingSomethingV2Dataset(Dataset):
    def __init__(
        self,
        *,
        split_path: str | Path,
        labels_path: str | Path,
        video_root: str | Path,
        image_size: int = 96,
        num_frames: int = 8,
    ) -> None:
        self.split_path = Path(split_path)
        self.labels_path = Path(labels_path)
        self.video_root = Path(video_root)
        self.image_size = image_size
        self.num_frames = num_frames

        self.label_to_id, self.classes = _load_label_map(self.labels_path)
        self.missing_paths = 0
        self.records = self._load_records()

    def _load_records(self) -> list[VideoRecord]:
        records: list[VideoRecord] = []
        for row in _load_split_records(self.split_path):
            video_id = str(row["id"])
            instance_label = str(row["label"])
            template_text = _normalize_template(str(row.get("template", instance_label)))
            label_id = self.label_to_id[template_text]
            video_path = self.video_root / f"{video_id}.webm"
            if not video_path.exists():
                self.missing_paths += 1
                continue
            records.append(
                VideoRecord(
                    video_id=video_id,
                    instance_label=instance_label,
                    template_text=template_text,
                    label_id=label_id,
                    path=video_path,
                    placeholders=tuple(str(x) for x in row.get("placeholders", [])),
                )
            )
        if not records:
            raise RuntimeError(f"No records found in {self.split_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return self.load_clip(record.video_id)

    def get_record(self, index: int) -> VideoRecord:
        return self.records[index]

    def get_video_path(self, index: int) -> Path:
        return self.records[index].path

    def load_clip(self, video_id: str, *, num_frames: int | None = None) -> tuple[torch.Tensor, int, str, str]:
        record = next((item for item in self.records if item.video_id == video_id), None)
        if record is None:
            raise KeyError(f"Unknown video id: {video_id}")
        return self.load_clip_from_path(
            record.path,
            record.label_id,
            record.template_text,
            record.video_id,
            num_frames=num_frames,
        )

    def load_clip_from_path(
        self,
        path: str | Path,
        label_id: int,
        label_text: str,
        video_id: str,
        *,
        num_frames: int | None = None,
    ) -> tuple[torch.Tensor, int, str, str]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        frames = _decode_video_frames(path)
        count = num_frames or self.num_frames
        indices = _sample_indices(count, len(frames))
        selected = [frames[idx] for idx in indices]
        clip = _frames_to_tensor(selected, image_size=self.image_size)
        return clip, label_id, label_text, video_id

    def describe(self, index: int) -> str:
        record = self.records[index]
        return f"{record.video_id}: {record.template_text} | {record.instance_label}"
