from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from jepa_world_models.data.stl10 import STL10Labeled, stl10_eval_transform
from jepa_world_models.vic_reg_loss.train import load_model_from_checkpoint


FeatureSpace = Literal["encoder", "projector"]


@dataclass(slots=True)
class FeatureBank:
    features: Tensor
    labels: Tensor
    indices: Tensor


@dataclass(slots=True)
class LayerwiseFeatureBank:
    layer_features: dict[int, Tensor]
    labels: Tensor
    indices: Tensor


@dataclass(slots=True)
class LabeledSplits:
    train: STL10Labeled
    test: STL10Labeled
    class_names: tuple[str, ...]


@dataclass(slots=True)
class CachedFeatureBank:
    feature_space: FeatureSpace
    split: str
    bank: FeatureBank


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpointed_models(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
):
    device_obj = resolve_device(device)
    cfg, encoder, projector, checkpoint = load_model_from_checkpoint(
        str(checkpoint_path),
        device=str(device_obj),
    )
    return cfg, encoder, projector, checkpoint


def build_labeled_splits(
    data_root: str | Path = "data_raw",
    image_size: int = 96,
) -> LabeledSplits:
    transform = stl10_eval_transform(image_size=image_size)
    train = STL10Labeled(
        root=str(data_root),
        split="train",
        transform=transform,
        return_index=True,
    )
    test = STL10Labeled(
        root=str(data_root),
        split="test",
        transform=transform,
        return_index=True,
    )
    return LabeledSplits(train=train, test=test, class_names=train.classes)


def build_eval_loader(
    dataset: STL10Labeled,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _unpack_batch(batch):
    if len(batch) == 3:
        images, labels, indices = batch
    else:
        images, labels = batch
        indices = torch.arange(labels.shape[0])
    return images, labels, indices


@torch.inference_mode()
def extract_feature_bank(
    encoder,
    projector,
    loader: DataLoader,
    device: str | torch.device | None = None,
    feature_space: FeatureSpace = "encoder",
) -> FeatureBank:
    device_obj = resolve_device(device)
    encoder.eval()
    projector.eval()

    feature_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []
    index_chunks: list[Tensor] = []

    for batch in loader:
        images, labels, indices = _unpack_batch(batch)
        images = images.to(device_obj, non_blocking=True)

        if feature_space == "encoder":
            features = encoder(images)
        elif feature_space == "projector":
            features = projector(encoder(images))
        else:
            raise ValueError(f"Unsupported feature_space: {feature_space}")

        feature_chunks.append(features.detach().cpu())
        label_chunks.append(labels.detach().cpu())
        index_chunks.append(indices.detach().cpu())

    return FeatureBank(
        features=torch.cat(feature_chunks, dim=0),
        labels=torch.cat(label_chunks, dim=0).long(),
        indices=torch.cat(index_chunks, dim=0).long(),
    )


@torch.inference_mode()
def extract_layerwise_feature_bank(
    encoder,
    loader: DataLoader,
    device: str | torch.device | None = None,
) -> LayerwiseFeatureBank:
    device_obj = resolve_device(device)
    encoder.eval()

    layer_chunks: list[list[Tensor]] = []
    label_chunks: list[Tensor] = []
    index_chunks: list[Tensor] = []

    for batch in loader:
        images, labels, indices = _unpack_batch(batch)
        images = images.to(device_obj, non_blocking=True)
        _, hidden_states = encoder(images, return_hidden_states=True)

        if not layer_chunks:
            layer_chunks = [[] for _ in range(len(hidden_states))]

        for layer_idx, hidden in enumerate(hidden_states):
            pooled = hidden.mean(dim=1).detach().cpu()
            layer_chunks[layer_idx].append(pooled)

        label_chunks.append(labels.detach().cpu())
        index_chunks.append(indices.detach().cpu())

    layer_features = {
        layer_idx: torch.cat(chunks, dim=0)
        for layer_idx, chunks in enumerate(layer_chunks)
    }
    return LayerwiseFeatureBank(
        layer_features=layer_features,
        labels=torch.cat(label_chunks, dim=0).long(),
        indices=torch.cat(index_chunks, dim=0).long(),
    )


def l2_normalize(x: Tensor) -> Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=1)


def save_feature_bank(
    bank: FeatureBank,
    path: str | Path,
    *,
    feature_space: FeatureSpace,
    split: str,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        CachedFeatureBank(feature_space=feature_space, split=split, bank=bank),
        path,
    )
    return str(path)


def load_feature_bank(path: str | Path) -> CachedFeatureBank:
    return torch.load(path, map_location="cpu", weights_only=False)
