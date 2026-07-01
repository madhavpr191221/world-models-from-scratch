from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import json
import math
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "labels" in data and isinstance(data["labels"], list):
            return data["labels"]
        if "database" in data and isinstance(data["database"], dict):
            items = []
            for key, value in data["database"].items():
                item = dict(value)
                item["video_id"] = key
                items.append(item)
            return items
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON format in {path}")


def _load_json_file(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_video_root(data_root: Path, split: str | None = None) -> Path:
    if split is not None:
        split_candidates = [
            data_root / "data_videos" / split,
            data_root / "videos" / split,
        ]
        resolved_split = _first_existing(split_candidates)
        if resolved_split is not None:
            return resolved_split
    candidates = [
        data_root / "something_v2" / "20bn-something-something-v2",
        data_root / "20bn-something-something-v2",
    ]
    resolved = _first_existing(candidates)
    if resolved is None:
        raise FileNotFoundError("Could not find a video root.")
    return resolved


def _resolve_labels_path(data_root: Path, split: str) -> Path:
    candidates = [
        data_root / "20bn-something-something-download-package-labels" / "labels" / f"{split}.json",
        data_root / "20bn-something-something-download-package-labels" / f"{split}.json",
    ]
    resolved = _first_existing(candidates)
    if resolved is None:
        raise FileNotFoundError(f"Could not find label metadata for split={split}.")
    return resolved


def _read_video_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _read_sampled_video_frames(video_path: Path, frame_indices: Sequence[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames: list[np.ndarray] = []
    try:
        for index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _video_is_decodable(video_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    try:
        ok, _ = cap.read()
        return bool(ok)
    finally:
        cap.release()


def _sample_frame_indices(num_frames_total: int, num_frames: int) -> list[int]:
    if num_frames_total <= 0:
        return []
    if num_frames_total >= num_frames:
        return np.linspace(0, num_frames_total - 1, num_frames).round().astype(int).tolist()
    repeats = int(math.ceil(num_frames / num_frames_total))
    indices = list(range(num_frames_total)) * repeats
    return indices[:num_frames]


def _resize_and_normalize(frame: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def load_clip_from_video_path(
    video_path: str | Path,
    num_frames: int = 16,
    image_size: int = 224,
) -> torch.Tensor:
    video_path = Path(video_path)
    frames = _read_video_frames(video_path)
    if not frames:
        raise RuntimeError(f"Could not decode video: {video_path}")
    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int).tolist()
    selected = []
    for idx in indices:
        frame = _resize_and_normalize(frames[idx], image_size)
        selected.append(frame)
    return torch.stack(selected, dim=0)


@dataclass(frozen=True)
class VideoSample:
    video_id: str
    path: str
    label: str


class SomethingSomethingVideoDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        image_size: int = 224,
        num_frames: int = 16,
        limit: int | None = None,
        seed: int = 0,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.video_root = _resolve_video_root(self.data_root, split)
        self.labels_path = None
        if self.video_root.name not in {"train", "validation", "test"}:
            self.labels_path = _resolve_labels_path(self.data_root, split)
        self.image_size = image_size
        self.num_frames = num_frames
        self.rng = random.Random(seed)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.samples = self._load_samples(limit=limit)

    def _sample_cache_path(self, limit: int | None) -> Path | None:
        if self.cache_dir is None:
            return None
        split_key = self.split if self.labels_path is None else self.labels_path.stem
        key = f"{split_key}_{self.image_size}_{self.num_frames}_{limit or 'all'}"
        return self.cache_dir / "video_samples" / f"{key}.json"

    def _load_samples(self, limit: int | None) -> list[VideoSample]:
        cache_path = self._sample_cache_path(limit)
        if cache_path is not None and cache_path.exists():
            payload = _load_json_file(cache_path)
            return [VideoSample(**item) for item in payload]

        if self.video_root.name in {"train", "validation", "test"}:
            samples: list[VideoSample] = []
            for video_path in sorted(self.video_root.glob("*.webm")):
                samples.append(VideoSample(video_id=video_path.stem, path=str(video_path), label=""))
                if limit is not None and len(samples) >= limit:
                    break
            if cache_path is not None:
                _write_json_file(cache_path, [sample.__dict__ for sample in samples])
            return samples if limit is None else samples[:limit]

        if self.labels_path is None:
            raise FileNotFoundError(f"Could not find label metadata or split folder for split={self.split}.")
        metadata = _load_json(self.labels_path)
        samples: list[VideoSample] = []
        for item in metadata:
            video_id = str(item.get("video_id") or item.get("id") or item.get("id_video") or item.get("video") or "")
            if not video_id:
                continue
            video_path = self.video_root / f"{video_id}.webm"
            if not video_path.exists() or not _video_is_decodable(video_path):
                continue
            label = str(item.get("template") or item.get("label") or item.get("text") or "")
            samples.append(VideoSample(video_id=video_id, path=str(video_path), label=label))
            if limit is not None and len(samples) >= limit:
                break

        if cache_path is not None:
            _write_json_file(cache_path, [sample.__dict__ for sample in samples])
        return samples if limit is None else samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        video_path = Path(sample.path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not decode video: {sample.path}")
        try:
            frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if frame_total <= 0:
            raise RuntimeError(f"Could not decode video: {sample.path}")
        frame_indices = _sample_frame_indices(frame_total, self.num_frames)
        frames = _read_sampled_video_frames(video_path, frame_indices)
        if len(frames) != len(frame_indices):
            full_frames = _read_video_frames(video_path)
            if not full_frames:
                raise RuntimeError(f"Could not decode video: {sample.path}")
            max_index = len(full_frames) - 1
            safe_indices = [min(int(i), max_index) for i in frame_indices]
            frames = [full_frames[i] for i in safe_indices]
        clip = torch.stack([_resize_and_normalize(frame, self.image_size) for frame in frames], dim=0)
        return {
            "clip": clip,
            "video_id": sample.video_id,
            "path": sample.path,
            "label": sample.label,
            "frame_count": torch.tensor(frame_total, dtype=torch.int64),
        }
