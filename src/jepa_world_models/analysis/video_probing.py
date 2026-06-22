from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from jepa_world_models.analysis.common import load_checkpointed_models, resolve_device
from jepa_world_models.analysis.probing import train_linear_probe
from jepa_world_models.data.video import SomethingSomethingV2Dataset


@dataclass(slots=True)
class DirectionFeatureSplit:
    split_name: str
    selected_clip_ids: list[str]
    selected_templates: list[str]
    labels: Tensor
    features: dict[str, Tensor]


@dataclass(slots=True)
class TemporalDirectionResult:
    task: str
    feature_view: str
    train_clips: int
    test_clips: int
    train_examples: int
    test_examples: int
    train_accuracy: float
    test_accuracy: float
    train_loss: float
    test_loss: float


def sample_clip_indices(total: int, max_clips: int, *, seed: int = 0) -> list[int]:
    if total <= 0:
        return []
    if max_clips <= 0 or max_clips >= total:
        return list(range(total))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:max_clips]
    return sorted(int(index) for index in indices.tolist())


def _encode_frames(encoder: torch.nn.Module, clip: Tensor, device: torch.device) -> Tensor:
    frames = clip.to(device)
    with torch.inference_mode():
        embeddings = encoder(frames)
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
    return embeddings.detach().cpu()


def _build_direction_examples(frame_embeddings: Tensor) -> dict[str, Tensor]:
    sequence_forward = frame_embeddings.flatten()
    sequence_reverse = torch.flip(frame_embeddings, dims=(0,)).flatten()
    pooled = frame_embeddings.mean(dim=0)
    return {
        "sequence_forward": sequence_forward,
        "sequence_reverse": sequence_reverse,
        "pooled_forward": pooled,
        "pooled_reverse": pooled,
    }


def build_direction_feature_split(
    dataset: SomethingSomethingV2Dataset,
    encoder: torch.nn.Module,
    device: str | torch.device | None = None,
    *,
    max_clips: int = 256,
    seed: int = 0,
    split_name: str,
) -> DirectionFeatureSplit:
    device_obj = resolve_device(device)
    encoder = encoder.to(device_obj)
    encoder.eval()

    indices = sample_clip_indices(len(dataset), max_clips, seed=seed)
    selected_clip_ids: list[str] = []
    selected_templates: list[str] = []

    sequence_features: list[Tensor] = []
    pooled_features: list[Tensor] = []
    labels: list[int] = []

    for index in indices:
        clip, _, template_text, video_id = dataset[index]
        frame_embeddings = _encode_frames(encoder, clip, device_obj)
        example_features = _build_direction_examples(frame_embeddings)

        selected_clip_ids.append(video_id)
        selected_templates.append(template_text)

        sequence_features.append(example_features["sequence_forward"])
        pooled_features.append(example_features["pooled_forward"])
        labels.append(1)

        sequence_features.append(example_features["sequence_reverse"])
        pooled_features.append(example_features["pooled_reverse"])
        labels.append(0)

    if not sequence_features:
        raise RuntimeError(f"No clips selected for split: {split_name}")

    return DirectionFeatureSplit(
        split_name=split_name,
        selected_clip_ids=selected_clip_ids,
        selected_templates=selected_templates,
        labels=torch.tensor(labels, dtype=torch.long),
        features={
            "sequence": torch.stack(sequence_features, dim=0),
            "pooled": torch.stack(pooled_features, dim=0),
        },
    )


def save_direction_feature_split(split: DirectionFeatureSplit, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(split, path)
    return str(path)


def load_direction_feature_split(path: str | Path) -> DirectionFeatureSplit:
    return torch.load(path, map_location="cpu", weights_only=False)


def run_temporal_direction_probe(
    checkpoint_path: str | Path,
    train_dataset: SomethingSomethingV2Dataset,
    test_dataset: SomethingSomethingV2Dataset,
    *,
    device: str | torch.device | None = None,
    max_train_clips: int = 256,
    max_test_clips: int = 64,
    seed: int = 0,
    train_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    cache_dir: str | Path | None = None,
    refresh_cache: bool = False,
) -> dict[str, object]:
    device_obj = resolve_device(device)
    cfg, encoder, _, _ = load_checkpointed_models(checkpoint_path, device=device_obj)
    encoder = encoder.to(device_obj)
    encoder.eval()

    cache_path = Path(cache_dir) if cache_dir is not None else None

    def load_or_compute(split_name: str, dataset: SomethingSomethingV2Dataset, max_clips: int, split_seed: int) -> DirectionFeatureSplit:
        if cache_path is not None:
            file_path = cache_path / f"{split_name}.pt"
            if file_path.exists() and not refresh_cache:
                cached = load_direction_feature_split(file_path)
                if cached.split_name == split_name:
                    print(f"Loaded cached video features: {file_path}")
                    return cached

        print(f"Computing video features: {split_name}")
        split = build_direction_feature_split(
            dataset,
            encoder,
            device_obj,
            max_clips=max_clips,
            seed=split_seed,
            split_name=split_name,
        )
        if cache_path is not None:
            saved_path = save_direction_feature_split(split, cache_path / f"{split_name}.pt")
            print(f"Saved cached video features: {saved_path}")
        return split

    train_split = load_or_compute("train", train_dataset, max_train_clips, seed)
    test_split = load_or_compute("validation", test_dataset, max_test_clips, seed + 1)

    results: list[TemporalDirectionResult] = []
    for feature_view in ("sequence", "pooled"):
        probe_result, _ = train_linear_probe(
            train_split.features[feature_view],
            train_split.labels,
            test_split.features[feature_view],
            test_split.labels,
            num_classes=2,
            device=device_obj,
            epochs=train_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
        )
        results.append(
            TemporalDirectionResult(
                task="forward_vs_reversed",
                feature_view=feature_view,
                train_clips=len(train_split.selected_clip_ids),
                test_clips=len(test_split.selected_clip_ids),
                train_examples=int(train_split.labels.shape[0]),
                test_examples=int(test_split.labels.shape[0]),
                train_accuracy=probe_result.train_accuracy,
                test_accuracy=probe_result.test_accuracy,
                train_loss=probe_result.train_loss,
                test_loss=probe_result.test_loss,
            )
        )

    return {
        "checkpoint": str(checkpoint_path),
        "device": str(device_obj),
        "image_size": cfg.image_size,
        "num_frames": train_dataset.num_frames,
        "train_split": train_split,
        "test_split": test_split,
        "results": results,
    }
