from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image
import torch
from sklearn.decomposition import PCA
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from jepa_world_models.analysis.common import load_checkpointed_models, l2_normalize, resolve_device


@dataclass(slots=True)
class VideoExample:
    index: int
    video_id: str
    filename: str
    label_id: int
    label_text: str
    template: str | None = None


@dataclass(slots=True)
class VideoRecord:
    index: int
    video_id: str
    label_id: int
    label_text: str
    template: str | None
    path: Path


@dataclass(slots=True)
class NeighborItem:
    rank: int
    index: int
    video_id: str
    filename: str
    label_id: int
    label_text: str
    score: float


@dataclass(slots=True)
class FramePoint:
    frame_index: int
    x: float
    y: float
    step_distance: float


@dataclass(slots=True)
class VideoDynamicsResult:
    query: dict[str, Any]
    bounds: dict[str, float]
    trajectory: list[FramePoint]
    global_neighbors: list[NeighborItem]
    frame_neighbors: list[list[NeighborItem]]

    def to_json(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "bounds": self.bounds,
            "trajectory": [asdict(point) for point in self.trajectory],
            "global_neighbors": [asdict(item) for item in self.global_neighbors],
            "frame_neighbors": [[asdict(item) for item in items] for items in self.frame_neighbors],
        }


@dataclass(slots=True)
class VideoClipBank:
    features: Tensor
    indices: Tensor
    video_ids: list[str]
    filenames: list[str]
    label_ids: Tensor
    label_texts: list[str]
    templates: list[str | None]

    def to_cache(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "indices": self.indices,
            "video_ids": self.video_ids,
            "filenames": self.filenames,
            "label_ids": self.label_ids,
            "label_texts": self.label_texts,
            "templates": self.templates,
        }

    @classmethod
    def from_cache(cls, payload: dict[str, Any]) -> "VideoClipBank":
        return cls(
            features=payload["features"],
            indices=payload["indices"],
            video_ids=list(payload["video_ids"]),
            filenames=list(payload["filenames"]),
            label_ids=payload["label_ids"],
            label_texts=list(payload["label_texts"]),
            templates=list(payload.get("templates", [])),
        )


class VideoSplitDataset:
    def __init__(self, records: list[VideoRecord], image_size: int = 96, num_frames: int = 16) -> None:
        self.records = records
        self.image_size = image_size
        self.num_frames = num_frames
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _decode_frames(self, path: Path) -> list[Image.Image]:
        container = av.open(str(path))
        frames: list[Image.Image] = []
        try:
            for frame in container.decode(video=0):
                frames.append(frame.to_image())
        finally:
            container.close()
        return frames

    def _sample_indices(self, total_frames: int) -> list[int]:
        if total_frames <= self.num_frames:
            return list(range(total_frames))
        return np.linspace(0, total_frames - 1, num=self.num_frames, dtype=int).tolist()

    def __getitem__(self, index: int) -> tuple[Tensor, int, str, str, str | None]:
        record = self.records[index]
        frames = self._decode_frames(record.path)
        if not frames:
            raise RuntimeError(f"No frames decoded from {record.path}")
        selected = [frames[i] for i in self._sample_indices(len(frames))]
        clip = torch.stack([self.transform(frame) for frame in selected], dim=0)
        return clip, record.label_id, record.label_text, record.video_id, record.template


def _default_data_root() -> Path:
    return Path("data")


def _video_root(data_root: str | Path) -> Path:
    return Path(data_root) / "something_v2" / "20bn-something-something-v2"


def _labels_root(data_root: str | Path) -> Path:
    return Path(data_root) / "20bn-something-something-download-package-labels" / "labels"


def _label_template_path(labels_root: Path) -> Path:
    return labels_root / "labels.json"


def _split_path(labels_root: Path, split: str) -> Path:
    return labels_root / f"{split}.json"


def _safe_json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_label_map(labels_root: Path) -> dict[str, str]:
    labels_path = _label_template_path(labels_root)
    payload = _safe_json_load(labels_path)
    return {str(key): str(value) for key, value in payload.items()}


def _normalize_template(template: str) -> str:
    return template.replace("[", "").replace("]", "")


def _build_dataset(
    *,
    data_root: str | Path,
    split: str,
    image_size: int,
):
    labels_root = _labels_root(data_root)
    video_root = _video_root(data_root)
    label_map = _load_label_map(labels_root)
    split_payload = _safe_json_load(_split_path(labels_root, split))
    if isinstance(split_payload, dict) and "labels" in split_payload:
        split_entries = split_payload["labels"]
    else:
        split_entries = split_payload

    records: list[VideoRecord] = []
    for index, entry in enumerate(split_entries):
        video_id = str(entry.get("id") or entry.get("video_id"))
        template = entry.get("template")
        label_text = str(entry.get("label", video_id))
        template_key = _normalize_template(template) if template else None
        label_id = int(label_map.get(template_key, label_map.get(template, -1))) if template else -1
        path = video_root / f"{video_id}.webm"
        if not path.exists():
            continue
        records.append(
            VideoRecord(
                index=index,
                video_id=video_id,
                label_id=label_id,
                label_text=label_text,
                template=template_key,
                path=path,
            )
        )

    return VideoSplitDataset(records, image_size=image_size), video_root, labels_root, label_map


def _encode_frames(encoder, frames: Tensor, device: torch.device) -> Tensor:
    frames = frames.to(device)
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape [T, C, H, W], got {tuple(frames.shape)}")
    with torch.inference_mode():
        features = encoder(frames)
    return features.detach().float().cpu()


def _pool_frames(frame_features: Tensor) -> Tensor:
    return frame_features.mean(dim=0, keepdim=True)


def _fit_projection(bank_features: Tensor, query_features: Tensor) -> PCA:
    combined = torch.cat([bank_features.float(), query_features.float()], dim=0).numpy()
    pca = PCA(n_components=2, random_state=0)
    pca.fit(combined)
    return pca


def _project_2d(pca: PCA, features: Tensor) -> np.ndarray:
    return pca.transform(features.float().numpy())


def _make_neighbor_items(
    *,
    scores: Tensor,
    bank: VideoClipBank,
    top_k: int,
) -> list[NeighborItem]:
    values, indices = torch.topk(scores, k=min(top_k, scores.numel()))
    items: list[NeighborItem] = []
    for rank, (score, bank_index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        items.append(
            NeighborItem(
                rank=rank,
                index=int(bank.indices[bank_index].item()),
                video_id=bank.video_ids[bank_index],
                filename=bank.filenames[bank_index],
                label_id=int(bank.label_ids[bank_index].item()),
                label_text=bank.label_texts[bank_index],
                score=float(score),
            )
        )
    return items


class VideoDynamicsEngine:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        data_root: str | Path = "data",
        device: str | torch.device | None = None,
        image_size: int = 96,
        train_split: str = "train",
        query_split: str = "validation",
        bank_size: int = 512,
        cache_dir: str | Path = "logs/video_dynamics",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.data_root = Path(data_root)
        self.device = resolve_device(device)
        self.image_size = image_size
        self.train_split = train_split
        self.query_split = query_split
        self.bank_size = bank_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bank_cache_path = self.cache_dir / "clip_bank.pt"

        self.cfg, self.encoder, self.projector, self.checkpoint = load_checkpointed_models(
            self.checkpoint_path,
            device=self.device,
        )
        self.encoder.eval()
        self.projector.eval()

        self.query_dataset, self.query_video_root, _, self.label_map = _build_dataset(
            data_root=self.data_root,
            split=self.query_split,
            image_size=self.image_size,
        )
        self.train_dataset, self.train_video_root, _, _ = _build_dataset(
            data_root=self.data_root,
            split=self.train_split,
            image_size=self.image_size,
        )

        self._bank: VideoClipBank | None = None
        self._pca: PCA | None = None

    def _dataset_len(self, dataset) -> int:
        return len(dataset)

    def _get_item(self, dataset, index: int):
        sample = dataset[index]
        if len(sample) == 5:
            clip, label_id, label_text, video_id, template = sample
        else:
            raise ValueError("Unexpected video sample shape.")
        filename = f"{video_id}.webm"
        return torch.as_tensor(clip), int(label_id), str(label_text), str(video_id), filename, template

    def _build_bank(self) -> VideoClipBank:
        if self._bank is not None:
            return self._bank
        if self.bank_cache_path.exists():
            payload = torch.load(self.bank_cache_path, map_location="cpu", weights_only=False)
            self._bank = VideoClipBank.from_cache(payload)
            return self._bank

        total = self._dataset_len(self.train_dataset)
        if total == 0:
            raise RuntimeError("Training video dataset is empty.")

        sample_count = min(self.bank_size, total)
        if sample_count <= 0:
            raise RuntimeError("Bank size must be positive.")

        sample_indices = np.linspace(0, total - 1, num=sample_count, dtype=int).tolist()
        feature_chunks: list[Tensor] = []
        index_chunks: list[int] = []
        video_ids: list[str] = []
        filenames: list[str] = []
        label_ids: list[int] = []
        label_texts: list[str] = []
        templates: list[str | None] = []

        for index in sample_indices:
            try:
                clip, label_id, label_text, video_id, filename, template = self._get_item(self.train_dataset, index)
            except Exception:
                continue
            frame_features = _encode_frames(self.encoder, clip, self.device)
            pooled = _pool_frames(frame_features)
            feature_chunks.append(pooled.squeeze(0))
            index_chunks.append(index)
            video_ids.append(video_id)
            filenames.append(filename)
            label_ids.append(label_id)
            label_texts.append(label_text)
            templates.append(template)

        if not feature_chunks:
            raise RuntimeError("Could not build a video feature bank.")

        bank = VideoClipBank(
            features=torch.stack(feature_chunks, dim=0),
            indices=torch.tensor(index_chunks, dtype=torch.long),
            video_ids=video_ids,
            filenames=filenames,
            label_ids=torch.tensor(label_ids, dtype=torch.long),
            label_texts=label_texts,
            templates=templates,
        )
        torch.save(bank.to_cache(), self.bank_cache_path)
        self._bank = bank
        return bank

    @property
    def bank(self) -> VideoClipBank:
        return self._build_bank()

    @property
    def pca(self) -> PCA:
        if self._pca is None:
            bank = self.bank
            self._pca = PCA(n_components=2, random_state=0)
            self._pca.fit(bank.features.float().numpy())
        return self._pca

    def list_examples(self, limit: int = 12) -> list[VideoExample]:
        examples: list[VideoExample] = []
        total = min(limit, self._dataset_len(self.query_dataset))
        for index in range(total):
            try:
                _, label_id, label_text, video_id, filename, template = self._get_item(self.query_dataset, index)
            except Exception:
                continue
            examples.append(
                VideoExample(
                    index=index,
                    video_id=video_id,
                    filename=filename,
                    label_id=label_id,
                    label_text=label_text,
                    template=template,
                )
            )
        return examples

    def _compute_neighbors(
        self,
        query_feature: Tensor,
        query_video_id: str,
        top_k: int = 5,
    ) -> list[NeighborItem]:
        bank = self.bank
        query = l2_normalize(query_feature.float())
        features = l2_normalize(bank.features.float())
        scores = (query @ features.T).squeeze(0)

        same_video_mask = torch.tensor([video_id == query_video_id for video_id in bank.video_ids], dtype=torch.bool)
        if same_video_mask.any():
            scores = scores.masked_fill(same_video_mask, float("-inf"))

        finite = torch.isfinite(scores)
        if finite.any():
            candidate_scores = scores[finite]
            candidate_bank = VideoClipBank(
                features=bank.features[finite],
                indices=bank.indices[finite],
                video_ids=[video_id for video_id, keep in zip(bank.video_ids, finite.tolist()) if keep],
                filenames=[filename for filename, keep in zip(bank.filenames, finite.tolist()) if keep],
                label_ids=bank.label_ids[finite],
                label_texts=[label_text for label_text, keep in zip(bank.label_texts, finite.tolist()) if keep],
                templates=[template for template, keep in zip(bank.templates, finite.tolist()) if keep],
            )
        else:
            candidate_scores = (query @ features.T).squeeze(0)
            candidate_bank = bank

        return _make_neighbor_items(scores=candidate_scores, bank=candidate_bank, top_k=top_k)

    def analyze_index(self, index: int, top_k: int = 5) -> VideoDynamicsResult:
        clip, label_id, label_text, video_id, filename, template = self._get_item(self.query_dataset, index)
        frame_features = _encode_frames(self.encoder, clip, self.device)
        pooled = _pool_frames(frame_features)
        pca = self.pca
        coords = _project_2d(pca, frame_features.float())

        trajectory: list[FramePoint] = []
        previous = None
        for frame_index, (x, y) in enumerate(coords):
            if previous is None:
                step_distance = 0.0
            else:
                step_distance = float(np.linalg.norm(np.array([x, y]) - np.array(previous)))
            trajectory.append(
                FramePoint(
                    frame_index=frame_index,
                    x=float(x),
                    y=float(y),
                    step_distance=step_distance,
                )
            )
            previous = (x, y)

        frame_neighbors: list[list[NeighborItem]] = []
        bank = self.bank
        bank_features = l2_normalize(bank.features.float())
        for frame_feature in frame_features:
            query = l2_normalize(frame_feature.unsqueeze(0).float())
            scores = (query @ bank_features.T).squeeze(0)
            same_video_mask = torch.tensor([video_id == candidate for candidate in bank.video_ids], dtype=torch.bool)
            if same_video_mask.any():
                scores = scores.masked_fill(same_video_mask, float("-inf"))
            frame_neighbors.append(_make_neighbor_items(scores=scores, bank=bank, top_k=top_k))

        global_neighbors = self._compute_neighbors(pooled, video_id, top_k=top_k)

        bounds = {
            "x_min": float(coords[:, 0].min()),
            "x_max": float(coords[:, 0].max()),
            "y_min": float(coords[:, 1].min()),
            "y_max": float(coords[:, 1].max()),
        }

        return VideoDynamicsResult(
            query={
                "index": int(index),
                "video_id": video_id,
                "filename": filename,
                "label_id": int(label_id),
                "label_text": label_text,
                "template": template,
                "num_frames": int(clip.shape[0]),
                "video_url": f"/api/video/file/{video_id}.webm",
            },
            bounds=bounds,
            trajectory=trajectory,
            global_neighbors=global_neighbors,
            frame_neighbors=frame_neighbors,
        )

    def analyze_reverse_index(self, index: int, top_k: int = 5) -> VideoDynamicsResult:
        clip, label_id, label_text, video_id, filename, template = self._get_item(self.query_dataset, index)
        reversed_clip = torch.flip(clip, dims=[0])
        frame_features = _encode_frames(self.encoder, reversed_clip, self.device)
        pooled = _pool_frames(frame_features)
        pca = self.pca
        coords = _project_2d(pca, frame_features.float())

        trajectory: list[FramePoint] = []
        previous = None
        for frame_index, (x, y) in enumerate(coords):
            if previous is None:
                step_distance = 0.0
            else:
                step_distance = float(np.linalg.norm(np.array([x, y]) - np.array(previous)))
            trajectory.append(
                FramePoint(
                    frame_index=frame_index,
                    x=float(x),
                    y=float(y),
                    step_distance=step_distance,
                )
            )
            previous = (x, y)

        frame_neighbors: list[list[NeighborItem]] = []
        bank = self.bank
        bank_features = l2_normalize(bank.features.float())
        for frame_feature in frame_features:
            query = l2_normalize(frame_feature.unsqueeze(0).float())
            scores = (query @ bank_features.T).squeeze(0)
            same_video_mask = torch.tensor([video_id == candidate for candidate in bank.video_ids], dtype=torch.bool)
            if same_video_mask.any():
                scores = scores.masked_fill(same_video_mask, float("-inf"))
            frame_neighbors.append(_make_neighbor_items(scores=scores, bank=bank, top_k=top_k))

        global_neighbors = self._compute_neighbors(pooled, video_id, top_k=top_k)
        bounds = {
            "x_min": float(coords[:, 0].min()),
            "x_max": float(coords[:, 0].max()),
            "y_min": float(coords[:, 1].min()),
            "y_max": float(coords[:, 1].max()),
        }
        return VideoDynamicsResult(
            query={
                "index": int(index),
                "video_id": video_id,
                "filename": filename,
                "label_id": int(label_id),
                "label_text": label_text,
                "template": template,
                "num_frames": int(reversed_clip.shape[0]),
                "video_url": f"/api/video/file/{video_id}.webm",
                "mode": "reversed",
            },
            bounds=bounds,
            trajectory=trajectory,
            global_neighbors=global_neighbors,
            frame_neighbors=frame_neighbors,
        )

    def write_demo_payload(self, index: int, output_path: str | Path, top_k: int = 5) -> Path:
        result = self.analyze_index(index=index, top_k=top_k)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
        return output_path


@lru_cache(maxsize=2)
def build_video_engine(
    checkpoint_path: str | Path,
    *,
    data_root: str | Path = "data",
    device: str | torch.device | None = None,
    image_size: int = 96,
    train_split: str = "train",
    query_split: str = "validation",
    bank_size: int = 512,
    cache_dir: str | Path = "logs/video_dynamics",
) -> VideoDynamicsEngine:
    return VideoDynamicsEngine(
        checkpoint_path=checkpoint_path,
        data_root=data_root,
        device=device,
        image_size=image_size,
        train_split=train_split,
        query_split=query_split,
        bank_size=bank_size,
        cache_dir=cache_dir,
    )
