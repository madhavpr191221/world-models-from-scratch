from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from jepa_world_models.analysis.common import (
    FeatureBank,
    load_feature_bank,
    LayerwiseFeatureBank,
    build_eval_loader,
    build_labeled_splits,
    extract_feature_bank,
    extract_layerwise_feature_bank,
    l2_normalize,
    load_checkpointed_models,
    save_feature_bank,
    resolve_device,
)


@dataclass(slots=True)
class LinearProbeResult:
    feature_space: str
    train_fraction: float
    train_accuracy: float
    test_accuracy: float
    train_loss: float
    test_loss: float


@dataclass(slots=True)
class KNNResult:
    feature_space: str
    k: int
    accuracy: float


def balanced_subset_indices(labels: Tensor, fraction: float, seed: int = 0) -> Tensor:
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")

    labels = labels.long().cpu()
    classes = labels.unique(sorted=True)
    class_indices = []
    generator = torch.Generator().manual_seed(seed)

    for cls in classes:
        cls_positions = torch.where(labels == cls)[0]
        perm = cls_positions[torch.randperm(len(cls_positions), generator=generator)]
        take = max(1, int(round(len(cls_positions) * fraction)))
        class_indices.append(perm[:take])

    return torch.cat(class_indices, dim=0)


def make_feature_dataset(features: Tensor, labels: Tensor) -> TensorDataset:
    return TensorDataset(features.float(), labels.long())


def _accuracy(logits: Tensor, labels: Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_linear_probe(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    *,
    num_classes: int = 10,
    device: str | torch.device | None = None,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
) -> tuple[LinearProbeResult, torch.nn.Module]:
    device_obj = resolve_device(device)
    train_dataset = make_feature_dataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    head = torch.nn.Linear(train_features.shape[1], num_classes).to(device_obj)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    x_train = train_features.to(device_obj)
    y_train = train_labels.to(device_obj)
    x_test = test_features.to(device_obj)
    y_test = test_labels.to(device_obj)

    for _ in range(epochs):
        head.train()
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device_obj)
            batch_labels = batch_labels.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    head.eval()
    with torch.inference_mode():
        train_logits = head(x_train)
        test_logits = head(x_test)
        train_loss = criterion(train_logits, y_train).item()
        test_loss = criterion(test_logits, y_test).item()
        train_acc = _accuracy(train_logits, y_train)
        test_acc = _accuracy(test_logits, y_test)

    result = LinearProbeResult(
        feature_space="linear",
        train_fraction=1.0,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        train_loss=train_loss,
        test_loss=test_loss,
    )
    return result, head


def train_probe_on_subset(
    train_bank: FeatureBank,
    test_bank: FeatureBank,
    *,
    fraction: float,
    seed: int = 0,
    num_classes: int = 10,
    device: str | torch.device | None = None,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    feature_space: str = "encoder",
) -> LinearProbeResult:
    subset = balanced_subset_indices(train_bank.labels, fraction=fraction, seed=seed)
    subset_features = train_bank.features[subset]
    subset_labels = train_bank.labels[subset]
    result, _ = train_linear_probe(
        subset_features,
        subset_labels,
        test_bank.features,
        test_bank.labels,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
    )
    result.feature_space = feature_space
    result.train_fraction = fraction
    return result


def cosine_knn_predict(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    *,
    k: int = 20,
    temperature: float = 0.07,
) -> Tensor:
    train_norm = l2_normalize(train_features.float())
    test_norm = l2_normalize(test_features.float())
    sims = test_norm @ train_norm.T
    topk_sims, topk_idx = sims.topk(k=min(k, train_norm.shape[0]), dim=1)
    topk_labels = train_labels[topk_idx]
    num_classes = int(train_labels.max().item()) + 1
    weights = torch.exp(topk_sims / temperature).unsqueeze(-1)
    votes = torch.nn.functional.one_hot(topk_labels, num_classes=num_classes).float()
    class_scores = (votes * weights).sum(dim=1)
    return class_scores.argmax(dim=1)


def evaluate_knn(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    *,
    feature_space: str,
    k: int = 20,
) -> KNNResult:
    preds = cosine_knn_predict(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        k=k,
    )
    accuracy = (preds == test_labels).float().mean().item()
    return KNNResult(feature_space=feature_space, k=k, accuracy=accuracy)


def run_probe_suite(
    checkpoint_path: str | Path,
    *,
    data_root: str | Path = "data_raw",
    device: str | torch.device | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
    train_epochs: int = 100,
    probe_lr: float = 1e-2,
    probe_weight_decay: float = 0.0,
    fractions: Sequence[float] = (0.01, 0.1, 1.0),
    k: int = 20,
    cache_dir: str | Path | None = None,
    refresh_cache: bool = False,
) -> dict:
    device_obj = resolve_device(device)
    cfg, encoder, projector, _ = load_checkpointed_models(checkpoint_path, device=device_obj)
    splits = build_labeled_splits(data_root=data_root, image_size=cfg.image_size)
    train_loader = build_eval_loader(
        splits.train,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = build_eval_loader(
        splits.test,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    cache_path = Path(cache_dir) if cache_dir is not None else None

    def load_or_compute(feature_space: str, split: str, loader: DataLoader) -> FeatureBank:
        if cache_path is not None:
            file_path = cache_path / f"{feature_space}_{split}.pt"
            if file_path.exists() and not refresh_cache:
                cached = load_feature_bank(file_path)
                if cached.feature_space == feature_space and cached.split == split:
                    print(f"Loaded cached features: {file_path}")
                    return cached.bank

        print(f"Computing features: {feature_space}/{split}")
        bank = extract_feature_bank(
            encoder,
            projector,
            loader,
            device=device_obj,
            feature_space=feature_space,
        )
        if cache_path is not None:
            saved_path = save_feature_bank(
                bank,
                cache_path / f"{feature_space}_{split}.pt",
                feature_space=feature_space,
                split=split,
            )
            print(f"Saved cached features: {saved_path}")
        return bank

    results: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "device": str(device_obj),
        "class_names": splits.class_names,
        "probe_results": [],
        "knn_results": [],
    }

    feature_spaces = {
        "encoder": load_or_compute("encoder", "train", train_loader),
        "projector": load_or_compute("projector", "train", train_loader),
    }
    test_spaces = {
        "encoder": load_or_compute("encoder", "test", test_loader),
        "projector": load_or_compute("projector", "test", test_loader),
    }

    for feature_space in ("encoder", "projector"):
        train_bank = feature_spaces[feature_space]
        test_bank = test_spaces[feature_space]
        results["knn_results"].append(
            evaluate_knn(
                train_bank.features,
                train_bank.labels,
                test_bank.features,
                test_bank.labels,
                feature_space=feature_space,
                k=k,
            )
        )
        for fraction in fractions:
            results["probe_results"].append(
                train_probe_on_subset(
                    train_bank,
                    test_bank,
                    fraction=fraction,
                    seed=0,
                    device=device_obj,
                    epochs=train_epochs,
                    batch_size=batch_size,
                    lr=probe_lr,
                    weight_decay=probe_weight_decay,
                    feature_space=feature_space,
                )
            )

    layerwise = extract_layerwise_feature_bank(
        encoder,
        train_loader,
        device=device_obj,
    )
    layerwise_test = extract_layerwise_feature_bank(
        encoder,
        test_loader,
        device=device_obj,
    )
    layer_results = []
    for layer_idx in sorted(layerwise.layer_features):
        result, _ = train_linear_probe(
            layerwise.layer_features[layer_idx],
            layerwise.labels,
            layerwise_test.layer_features[layer_idx],
            layerwise_test.labels,
            device=device_obj,
            epochs=train_epochs,
            batch_size=batch_size,
            lr=probe_lr,
            weight_decay=probe_weight_decay,
        )
        result.feature_space = f"layer_{layer_idx}"
        layer_results.append(result)
    results["layerwise_results"] = layer_results
    results["train_features"] = feature_spaces
    results["test_features"] = test_spaces
    return results
