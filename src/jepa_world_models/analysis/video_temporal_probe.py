from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from jepa_world_models.analysis.common import load_checkpointed_models, resolve_device


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
    return frame_features.reshape(-1)


def extract_clip_features(
    *,
    checkpoint_path: str | Path,
    samples: list[TemporalProbeSample],
    data_root: str | Path = "data",
    image_size: int = 96,
    num_frames: int = 32,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, list[TemporalProbeSample]]:
    device_obj = resolve_device(device)
    _, encoder, _, _ = load_checkpointed_models(checkpoint_path, device=device_obj)
    encoder.eval()

    video_root = _video_root(data_root)
    features: list[np.ndarray] = []
    labels: list[int] = []
    kept_samples: list[TemporalProbeSample] = []

    for sample in tqdm(samples, desc="Encoding clips", leave=False):
        clip = _decode_clip(video_root / sample.filename, image_size=image_size, num_frames=num_frames)
        clip = clip.to(device_obj, non_blocking=True)
        with torch.inference_mode():
            frame_features = encoder(clip)
            if frame_features.ndim == 1:
                frame_features = frame_features.unsqueeze(0)
            clip_feature = _sequence_feature(frame_features, sample.is_reversed)
        features.append(clip_feature.detach().cpu().numpy())
        labels.append(sample.is_reversed)
        kept_samples.append(sample)

    return np.stack(features, axis=0), np.asarray(labels, dtype=np.int64), kept_samples


class TemporalFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class TemporalProbeHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass(slots=True)
class TemporalProbeResult:
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    confusion: list[list[int]]
    report: dict[str, Any]
    feature_shape: tuple[int, int]
    checkpoint_path: str

    def to_json(self) -> dict[str, Any]:
        return {
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "test_accuracy": self.test_accuracy,
            "confusion": self.confusion,
            "report": self.report,
            "feature_shape": list(self.feature_shape),
            "checkpoint_path": self.checkpoint_path,
        }


def _train_epoch(
    model: TemporalProbeHead,
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
def _evaluate(model: TemporalProbeHead, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
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
    output_dir: str | Path = "logs/video_temporal_probe",
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

    train_samples = expand_forward_reverse_samples(train_sources)
    val_samples = expand_forward_reverse_samples(val_sources)
    test_samples = expand_forward_reverse_samples(test_sources)

    x_train, y_train, _ = extract_clip_features(
        checkpoint_path=checkpoint_path,
        samples=train_samples,
        data_root=data_root,
        image_size=image_size,
        num_frames=num_frames,
    )
    x_val, y_val, _ = extract_clip_features(
        checkpoint_path=checkpoint_path,
        samples=val_samples,
        data_root=data_root,
        image_size=image_size,
        num_frames=num_frames,
    )
    x_test, y_test, _ = extract_clip_features(
        checkpoint_path=checkpoint_path,
        samples=test_samples,
        data_root=data_root,
        image_size=image_size,
        num_frames=num_frames,
    )

    train_tensor = torch.from_numpy(x_train).float()
    val_tensor = torch.from_numpy(x_val).float()
    test_tensor = torch.from_numpy(x_test).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    y_val_tensor = torch.from_numpy(y_val).long()
    y_test_tensor = torch.from_numpy(y_test).long()

    train_loader = DataLoader(TemporalFeatureDataset(train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TemporalFeatureDataset(val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TemporalFeatureDataset(test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    device = resolve_device(None)
    model = TemporalProbeHead(input_dim=train_tensor.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_path = output_path / "best_probe.pt"
    best_val = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_acc = _train_epoch(model, train_loader, optimizer, device)
        val_acc, _, _ = _evaluate(model, val_loader, device)
        history.append({"epoch": float(epoch), "train_accuracy": train_acc, "val_accuracy": val_acc})
        if val_acc >= best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(), "input_dim": train_tensor.shape[1], "epoch": epoch}, best_path)

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
        "history": history,
        "subset_size": subset_size,
        "split_counts": {
            "train": train_count,
            "validation": val_count,
            "test": test_count,
        },
    }
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TemporalProbeResult(
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        test_accuracy=test_accuracy,
        confusion=confusion,
        report=report,
        feature_shape=tuple(train_tensor.shape),
        checkpoint_path=str(best_path),
    )
