from __future__ import annotations

import json
import random
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from jepa_world_models.analysis.common import load_checkpointed_models, resolve_device

FEATURE_CACHE_VERSION = "v3_temporaltransformer_diff_pos"

@dataclass(slots=True)
class TemporalProbeSample:
    clip_id: str
    filename: str
    split: str
    label_name: str
    is_reversed: int


def _data_root(data_root: str | Path) -> Path:
    return Path(data_root)


def _labels_root(data_root: str | Path) -> Path:
    return _data_root(data_root) / "20bn-something-something-download-package-labels" / "labels"


def _video_root(data_root: str | Path) -> Path:
    return _data_root(data_root) / "something_v2" / "20bn-something-something-v2"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_split_entries(labels_root: Path, split: str) -> list[dict[str, Any]]:
    payload = _load_json(labels_root / f"{split}.json")
    if isinstance(payload, dict) and "labels" in payload:
        return list(payload["labels"])
    return list(payload)


def _sample_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= num_frames:
        indices = list(range(total_frames))
        if not indices:
            return [0] * num_frames
        while len(indices) < num_frames:
            indices.append(indices[-1])
        return indices
    return np.linspace(0, total_frames - 1, num=num_frames, dtype=int).tolist()


def _decode_clip(path: Path, image_size: int, num_frames: int) -> torch.Tensor:
    container = av.open(str(path))
    frames: list[Image.Image] = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    selected = [frames[i] for i in _sample_indices(len(frames), num_frames)]
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )
    return torch.stack([transform(frame) for frame in selected], dim=0)


def _is_usable_video(path: Path) -> bool:
    try:
        container = av.open(str(path))
        try:
            for _ in container.decode(video=0):
                return True
        finally:
            container.close()
    except Exception:
        return False
    return False


def build_temporal_probe_sources(
    *,
    data_root: str | Path = "data",
    split: str = "train",
    limit: int | None = None,
    seed: int = 0,
) -> list[TemporalProbeSample]:
    labels_root = _labels_root(data_root)
    video_root = _video_root(data_root)
    split_entries = _load_split_entries(labels_root, split)
    rng = random.Random(seed)
    indices = list(range(len(split_entries)))
    rng.shuffle(indices)

    sources: list[TemporalProbeSample] = []
    if limit is None:
        limit = len(indices)

    valid_source_count = 0
    visited = 0
    while valid_source_count < limit and visited < len(indices) * 3:
        entry = split_entries[indices[visited % len(indices)]]
        visited += 1
        clip_id = str(entry.get("id") or entry.get("video_id"))
        label_name = str(entry.get("label", clip_id))
        path = video_root / f"{clip_id}.webm"
        if not path.exists() or not _is_usable_video(path):
            continue
        sources.append(
            TemporalProbeSample(
                clip_id=clip_id,
                filename=path.name,
                split=split,
                label_name=label_name,
                is_reversed=0,
            )
        )
        valid_source_count += 1

    if valid_source_count < limit:
        raise RuntimeError(
            f"Could only assemble {valid_source_count} usable videos for split={split}, requested {limit}."
        )
    return sources


def expand_forward_reverse_samples(samples: list[TemporalProbeSample]) -> list[TemporalProbeSample]:
    expanded: list[TemporalProbeSample] = []
    for sample in samples:
        expanded.append(sample)
        expanded.append(
            TemporalProbeSample(
                clip_id=sample.clip_id,
                filename=sample.filename,
                split=sample.split,
                label_name=sample.label_name,
                is_reversed=1,
            )
        )
    return expanded


def _sequence_feature(frame_features: torch.Tensor, is_reversed: int) -> torch.Tensor:
    if is_reversed:
        frame_features = torch.flip(frame_features, dims=[0])
    return frame_features


def _cache_key(
    *,
    source_split: str,
    subset_size: int,
    num_frames: int,
    image_size: int,
    seed: int,
) -> str:
    payload = f"{FEATURE_CACHE_VERSION}|{source_split}|{subset_size}|{num_frames}|{image_size}|{seed}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _cache_path(
    cache_dir: str | Path,
    *,
    source_split: str,
    subset_size: int,
    num_frames: int,
    image_size: int,
    seed: int,
) -> Path:
    cache_dir = Path(cache_dir)
    return cache_dir / f"source_features_{source_split}_n{subset_size}_t{num_frames}_s{image_size}_{seed}_{_cache_key(source_split=source_split, subset_size=subset_size, num_frames=num_frames, image_size=image_size, seed=seed)}.pt"


def _save_source_feature_cache(
    path: Path,
    *,
    features: torch.Tensor,
    samples: list[TemporalProbeSample],
    num_frames: int,
    image_size: int,
    subset_size: int,
    source_split: str,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": FEATURE_CACHE_VERSION,
        "features": features.cpu(),
        "samples": [asdict(sample) for sample in samples],
        "num_frames": num_frames,
        "image_size": image_size,
        "subset_size": subset_size,
        "source_split": source_split,
        "seed": seed,
    }
    torch.save(payload, path)


def _load_source_feature_cache(path: Path) -> tuple[torch.Tensor, list[TemporalProbeSample]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("cache_version") != FEATURE_CACHE_VERSION:
        raise ValueError("stale cache version")
    samples = [TemporalProbeSample(**item) for item in payload["samples"]]
    return payload["features"], samples


def extract_source_clip_features(
    *,
    checkpoint_path: str | Path,
    samples: list[TemporalProbeSample],
    data_root: str | Path = "data",
    image_size: int = 96,
    num_frames: int = 32,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, list[TemporalProbeSample]]:
    device_obj = resolve_device(device)
    _, encoder, _, _ = load_checkpointed_models(checkpoint_path, device=device_obj)
    encoder.eval()

    video_root = _video_root(data_root)
    features: list[torch.Tensor] = []
    kept_samples: list[TemporalProbeSample] = []

    for sample in tqdm(samples, desc="Encoding clips", leave=False):
        clip = _decode_clip(video_root / sample.filename, image_size=image_size, num_frames=num_frames)
        clip = clip.to(device_obj, non_blocking=True)
        with torch.inference_mode():
            frame_features = encoder(clip)
            if frame_features.ndim == 1:
                frame_features = frame_features.unsqueeze(0)
            clip_feature = _sequence_feature(frame_features, sample.is_reversed).detach().cpu()
        features.append(clip_feature)
        kept_samples.append(sample)

    return torch.stack(features, dim=0), kept_samples


class TemporalFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class TemporalTransformerProbe(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        max_frames: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.positional = nn.Parameter(torch.zeros(1, max_frames, embedding_dim * 2))
        self.input_proj = nn.Linear(embedding_dim * 2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        diffs = torch.zeros_like(x)
        diffs[:, 1:] = x[:, 1:] - x[:, :-1]
        x = torch.cat([x, diffs], dim=-1)  # (B, T, 2d)
        x = x + self.positional[:, : x.shape[1], :]
        x = self.input_proj(x)  # (B, T, h)
        x = self.temporal(x)  # (B, T, h)
        x = x.mean(dim=1)  # (B, h)
        return self.classifier(x)


@dataclass(slots=True)
class TemporalProbeResult:
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    confusion: list[list[int]]
    report: dict[str, Any]
    feature_shape: tuple[int, int, int]
    checkpoint_path: str
    model_dir: str

    def to_json(self) -> dict[str, Any]:
        return {
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "test_accuracy": self.test_accuracy,
            "confusion": self.confusion,
            "report": self.report,
            "feature_shape": list(self.feature_shape),
            "checkpoint_path": self.checkpoint_path,
            "model_dir": self.model_dir,
        }


def _train_epoch(
    model: TemporalTransformerProbe,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_correct = 0
    total_seen = 0
    loss_fn = nn.CrossEntropyLoss()
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_seen += labels.numel()
    return total_correct / max(1, total_seen)


@torch.inference_mode()
def _evaluate(model: TemporalTransformerProbe, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    y_pred = torch.cat(preds, dim=0).numpy()
    y_true = torch.cat(targets, dim=0).numpy()
    return float((y_pred == y_true).mean()), y_true, y_pred


def train_forward_reverse_probe(
    *,
    checkpoint_path: str | Path,
    data_root: str | Path = "data",
    source_split: str = "train",
    image_size: int = 96,
    num_frames: int = 32,
    subset_size: int = 2000,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-3,
    seed: int = 0,
    output_dir: str | Path = "logs/video_temporal_probe/temporal_transformer",
    cache_dir: str | Path | None = "logs/video_temporal_probe/cache",
) -> TemporalProbeResult:
    torch.manual_seed(seed)

    train_count = int(np.floor(subset_size * 0.7))
    val_count = int(np.floor(subset_size * 0.2))
    test_count = subset_size - train_count - val_count

    source_samples = build_temporal_probe_sources(
        data_root=data_root,
        split=source_split,
        limit=subset_size,
        seed=seed,
    )
    rng = random.Random(seed)
    rng.shuffle(source_samples)

    train_sources = source_samples[:train_count]
    val_sources = source_samples[train_count : train_count + val_count]
    test_sources = source_samples[train_count + val_count : train_count + val_count + test_count]

    cache_path = None
    if cache_dir is not None:
        cache_path = _cache_path(
            cache_dir,
            source_split=source_split,
            subset_size=subset_size,
            num_frames=num_frames,
            image_size=image_size,
            seed=seed,
        )

    if cache_path is not None and cache_path.exists():
        cached_features, cached_sources = _load_source_feature_cache(cache_path)
        if len(cached_sources) == subset_size:
            source_features = cached_features
            source_samples = cached_sources
        else:
            source_features, source_samples = extract_source_clip_features(
                checkpoint_path=checkpoint_path,
                samples=source_samples,
                data_root=data_root,
                image_size=image_size,
                num_frames=num_frames,
            )
            _save_source_feature_cache(
                cache_path,
                features=source_features,
                samples=source_samples,
                num_frames=num_frames,
                image_size=image_size,
                subset_size=subset_size,
                source_split=source_split,
                seed=seed,
            )
    else:
        source_features, source_samples = extract_source_clip_features(
            checkpoint_path=checkpoint_path,
            samples=source_samples,
            data_root=data_root,
            image_size=image_size,
            num_frames=num_frames,
        )
        if cache_path is not None:
            _save_source_feature_cache(
                cache_path,
                features=source_features,
                samples=source_samples,
                num_frames=num_frames,
                image_size=image_size,
                subset_size=subset_size,
                source_split=source_split,
                seed=seed,
            )

    def _split_features(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_f = features[:train_count]
        val_f = features[train_count : train_count + val_count]
        test_f = features[train_count + val_count : train_count + val_count + test_count]
        return train_f, val_f, test_f

    train_source_features, val_source_features, test_source_features = _split_features(source_features)

    def _expand_features(split_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        forward = split_features
        reversed_features = torch.flip(split_features, dims=[1])
        features = torch.cat([forward, reversed_features], dim=0)
        labels = torch.cat(
            [
                torch.zeros(forward.shape[0], dtype=torch.long),
                torch.ones(reversed_features.shape[0], dtype=torch.long),
            ],
            dim=0,
        )
        return features, labels

    x_train, y_train = _expand_features(train_source_features)
    x_val, y_val = _expand_features(val_source_features)
    x_test, y_test = _expand_features(test_source_features)

    train_tensor = x_train.float()
    val_tensor = x_val.float()
    test_tensor = x_test.float()
    y_train_tensor = y_train.long()
    y_val_tensor = y_val.long()
    y_test_tensor = y_test.long()

    train_loader = DataLoader(TemporalFeatureDataset(train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TemporalFeatureDataset(val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TemporalFeatureDataset(test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    device = resolve_device(None)
    model = TemporalTransformerProbe(embedding_dim=train_tensor.shape[-1], max_frames=train_tensor.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_temporal_transformer.pt"
    final_path = model_dir / "temporal_transformer.pt"
    best_val = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_acc = _train_epoch(model, train_loader, optimizer, device)
        val_acc, _, _ = _evaluate(model, val_loader, device)
        history.append({"epoch": float(epoch), "train_accuracy": train_acc, "val_accuracy": val_acc})
        if val_acc >= best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(), "embedding_dim": train_tensor.shape[-1], "epoch": epoch}, best_path)

    torch.save({"state_dict": model.state_dict(), "embedding_dim": train_tensor.shape[-1], "epoch": epochs}, final_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    train_accuracy, y_train_true, y_train_pred = _evaluate(model, train_loader, device)
    val_accuracy, y_val_true, y_val_pred = _evaluate(model, val_loader, device)
    test_accuracy, y_test_true, y_test_pred = _evaluate(model, test_loader, device)

    confusion = confusion_matrix(y_test_true, y_test_pred).tolist()
    report = classification_report(y_test_true, y_test_pred, output_dict=True, zero_division=0)

    metrics = {
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "confusion": confusion,
        "report": report,
        "feature_shape": list(train_tensor.shape),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "history": history,
        "subset_size": subset_size,
        "split_counts": {
            "train": train_count,
            "validation": val_count,
            "test": test_count,
        },
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TemporalProbeResult(
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        test_accuracy=test_accuracy,
        confusion=confusion,
        report=report,
        feature_shape=tuple(train_tensor.shape),
        checkpoint_path=str(best_path),
        model_dir=str(model_dir),
    )
