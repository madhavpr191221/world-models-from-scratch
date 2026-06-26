from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from jepa_world_models.analysis.common import resolve_device
from jepa_world_models.analysis.videomae_pipeline import (
    SomethingSomethingVideoDataset,
    VideoMAEModel,
)


@dataclass(slots=True)
class LatentSequenceBank:
    context_latents: torch.Tensor  # (N, C, D)
    future_latents: torch.Tensor  # (N, F, D)
    sample_indices: list[int]
    video_ids: list[str]
    source_split: str
    checkpoint_path: str
    data_root: str
    subset_size: int
    image_size: int
    total_frames: int
    context_frames: int
    future_frames: int

    def to_payload(self) -> dict:
        return {
            "context_latents": self.context_latents.cpu(),
            "future_latents": self.future_latents.cpu(),
            "sample_indices": self.sample_indices,
            "video_ids": self.video_ids,
            "source_split": self.source_split,
            "checkpoint_path": self.checkpoint_path,
            "data_root": self.data_root,
            "subset_size": self.subset_size,
            "image_size": self.image_size,
            "total_frames": self.total_frames,
            "context_frames": self.context_frames,
            "future_frames": self.future_frames,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "LatentSequenceBank":
        return cls(
            context_latents=payload["context_latents"],
            future_latents=payload["future_latents"],
            sample_indices=[int(i) for i in payload["sample_indices"]],
            video_ids=[str(video_id) for video_id in payload["video_ids"]],
            source_split=str(payload["source_split"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            data_root=str(payload["data_root"]),
            subset_size=int(payload["subset_size"]),
            image_size=int(payload["image_size"]),
            total_frames=int(payload["total_frames"]),
            context_frames=int(payload["context_frames"]),
            future_frames=int(payload["future_frames"]),
        )

    @property
    def latent_dim(self) -> int:
        return int(self.context_latents.shape[-1])

    @property
    def context_steps(self) -> int:
        return int(self.context_latents.shape[1])

    @property
    def future_steps(self) -> int:
        return int(self.future_latents.shape[1])


@dataclass(slots=True)
class LatentWorldModelBundle:
    state_dict: dict
    latent_dim: int
    hidden_dim: int
    context_steps: int
    future_steps: int
    num_layers: int
    num_heads: int
    dropout: float
    architecture_version: int
    checkpoint_path: str
    source_split: str
    subset_size: int
    image_size: int
    total_frames: int
    context_frames: int
    future_frames: int
    train_loss: float | None = None
    val_loss: float | None = None
    best_epoch: int | None = None

    def to_payload(self) -> dict:
        return {
            "state_dict": self.state_dict,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "context_steps": self.context_steps,
            "future_steps": self.future_steps,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "architecture_version": self.architecture_version,
            "checkpoint_path": self.checkpoint_path,
            "source_split": self.source_split,
            "subset_size": self.subset_size,
            "image_size": self.image_size,
            "total_frames": self.total_frames,
            "context_frames": self.context_frames,
            "future_frames": self.future_frames,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_epoch": self.best_epoch,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "LatentWorldModelBundle":
        return cls(
            state_dict=payload["state_dict"],
            latent_dim=int(payload["latent_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            context_steps=int(payload["context_steps"]),
            future_steps=int(payload["future_steps"]),
            num_layers=int(payload["num_layers"]),
            num_heads=int(payload["num_heads"]),
            dropout=float(payload["dropout"]),
            architecture_version=int(payload.get("architecture_version", 1)),
            checkpoint_path=str(payload["checkpoint_path"]),
            source_split=str(payload["source_split"]),
            subset_size=int(payload["subset_size"]),
            image_size=int(payload["image_size"]),
            total_frames=int(payload["total_frames"]),
            context_frames=int(payload["context_frames"]),
            future_frames=int(payload["future_frames"]),
            train_loss=None if payload.get("train_loss") is None else float(payload["train_loss"]),
            val_loss=None if payload.get("val_loss") is None else float(payload["val_loss"]),
            best_epoch=None if payload.get("best_epoch") is None else int(payload["best_epoch"]),
        )


@dataclass(slots=True)
class LatentWorldModelResult:
    train_loss: float
    val_loss: float
    test_loss: float
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    baseline_metrics: dict[str, dict[str, float]]
    checkpoint_path: str
    cache_path: str
    report_path: str
    predictions_path: str
    latent_shape: tuple[int, int, int]
    history: list[dict[str, float]]
    best_epoch: int
    num_samples: int
    context_frames: int
    future_frames: int
    total_frames: int
    image_size: int

    def to_json(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "test_loss": self.test_loss,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "baseline_metrics": self.baseline_metrics,
            "checkpoint_path": self.checkpoint_path,
            "cache_path": self.cache_path,
            "report_path": self.report_path,
            "predictions_path": self.predictions_path,
            "latent_shape": list(self.latent_shape),
            "history": self.history,
            "best_epoch": self.best_epoch,
            "num_samples": self.num_samples,
            "context_frames": self.context_frames,
            "future_frames": self.future_frames,
            "total_frames": self.total_frames,
            "image_size": self.image_size,
        }


class LatentWindowDataset(Dataset):
    def __init__(
        self,
        context_latents: torch.Tensor,
        future_latents: torch.Tensor,
        sample_indices: list[int],
        video_ids: list[str],
    ) -> None:
        if context_latents.shape[0] != future_latents.shape[0]:
            raise ValueError("Context and future tensors must have the same batch size.")
        if len(sample_indices) != context_latents.shape[0]:
            raise ValueError("sample_indices must align with the tensor batch size.")
        if len(video_ids) != context_latents.shape[0]:
            raise ValueError("video_ids must align with the tensor batch size.")
        self.context_latents = context_latents
        self.future_latents = future_latents
        self.sample_indices = sample_indices
        self.video_ids = video_ids

    def __len__(self) -> int:
        return self.context_latents.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        return {
            "context": self.context_latents[index],
            "future": self.future_latents[index],
            "sample_index": int(self.sample_indices[index]),
            "video_id": self.video_ids[index],
        }


class TemporalLatentPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_steps: int,
        future_steps: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if context_steps <= 0 or future_steps <= 0:
            raise ValueError("context_steps and future_steps must be positive.")
        self.latent_dim = latent_dim
        self.context_steps = context_steps
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.input_norm = nn.LayerNorm(latent_dim)
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.context_pos_embed = nn.Parameter(torch.zeros(1, context_steps, hidden_dim))
        self.future_query = nn.Parameter(torch.zeros(1, future_steps, hidden_dim))
        self.future_pos_embed = nn.Parameter(torch.zeros(1, future_steps, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, latent_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.context_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.future_query, std=0.02)
        nn.init.trunc_normal_(self.future_pos_embed, std=0.02)

    def forward(self, context_latents: torch.Tensor) -> torch.Tensor:
        if context_latents.ndim != 3:
            raise ValueError(f"Expected context_latents with shape (B, C, D), got {tuple(context_latents.shape)}")
        if context_latents.shape[1] != self.context_steps:
            raise ValueError(
                f"Expected {self.context_steps} context steps, got {context_latents.shape[1]}"
            )
        if context_latents.shape[2] != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {context_latents.shape[2]}")

        context_tokens = self.input_proj(self.input_norm(context_latents))
        context_tokens = context_tokens + self.context_pos_embed
        future_tokens = self.future_query.expand(context_latents.shape[0], -1, -1)
        future_tokens = future_tokens + self.future_pos_embed
        tokens = torch.cat([context_tokens, future_tokens], dim=1)
        encoded = self.blocks(tokens)
        future_encoded = encoded[:, -self.future_steps :, :]
        return self.output_head(self.output_norm(future_encoded))


def _checkpoint_fingerprint(checkpoint_path: str | Path) -> str:
    path = Path(checkpoint_path).resolve()
    stat = path.stat()
    digest = f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(digest).hexdigest()[:10]


def _resolve_video_root(data_root: str | Path) -> Path:
    data_root = Path(data_root)
    candidates = [
        data_root / "something_v2" / "20bn-something-something-v2",
        data_root / "20bn-something-something-v2",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find Something-Something V2 video root.")


def _load_model(checkpoint_path: str | Path, device: str | torch.device | None = None) -> VideoMAEModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = VideoMAEModel(
        image_size=int(config.get("image_size", 224)),
        num_frames=int(config.get("num_frames", 16)),
        embed_dim=int(config.get("embed_dim", 192)),
    )
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    device_obj = resolve_device(device)
    return model.to(device_obj)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def _sample_contiguous_window(frames: list[np.ndarray], num_frames: int, seed: int) -> list[np.ndarray]:
    if not frames:
        return []
    if len(frames) >= num_frames:
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, len(frames) - num_frames + 1))
        return frames[start : start + num_frames]
    window = list(frames)
    while len(window) < num_frames:
        window.append(window[-1])
    return window


def _normalize_frame(frame: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _split_sizes(n: int) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError("Need at least 3 samples to create train/validation/test splits.")
    train = max(1, int(math.floor(0.7 * n)))
    val = max(1, int(math.floor(0.2 * n)))
    test = n - train - val
    if test < 1:
        deficit = 1 - test
        train_cut = min(deficit, max(0, train - 1))
        train -= train_cut
        deficit -= train_cut
        if deficit > 0:
            val_cut = min(deficit, max(0, val - 1))
            val -= val_cut
        test = n - train - val
    return train, val, test


def _split_indices(n: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator).tolist()
    train_size, val_size, _ = _split_sizes(n)
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size : train_size + val_size]
    test_indices = permutation[train_size + val_size :]
    return train_indices, val_indices, test_indices


@torch.inference_mode()
def _encode_latent_sequences(
    model: VideoMAEModel,
    clips: torch.Tensor,
) -> torch.Tensor:
    return model.encoder.forward_sequence(clips)


@torch.inference_mode()
def build_latent_sequence_bank(
    *,
    checkpoint_path: str | Path,
    data_root: str | Path,
    source_split: str = "train",
    subset_size: int = 128,
    image_size: int = 224,
    total_frames: int = 16,
    context_frames: int = 12,
    future_frames: int = 4,
    batch_size: int = 1,
    cache_dir: str | Path | None = None,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> LatentSequenceBank:
    if context_frames + future_frames != total_frames:
        raise ValueError("context_frames + future_frames must equal total_frames.")
    if total_frames % 2 != 0:
        raise ValueError("total_frames must be divisible by the tubelet size of 2.")
    if context_frames % 2 != 0 or future_frames % 2 != 0:
        raise ValueError("context_frames and future_frames must be divisible by 2.")

    cache_path: Path | None = None
    if cache_dir is not None:
        checkpoint_stem = Path(checkpoint_path).stem
        checkpoint_id = _checkpoint_fingerprint(checkpoint_path)
        cache_name = (
            f"latent_sequence_bank_{checkpoint_stem}_{checkpoint_id}_{source_split}_{subset_size}"
            f"_{image_size}_{total_frames}_{context_frames}_{future_frames}_{seed}.pt"
        )
        cache_path = Path(cache_dir) / cache_name
        if cache_path.exists():
            return LatentSequenceBank.from_payload(torch.load(cache_path, map_location="cpu", weights_only=False))

    data_root = Path(data_root)
    video_root = _resolve_video_root(data_root)
    dataset = SomethingSomethingVideoDataset(
        data_root=data_root,
        split=source_split,
        image_size=image_size,
        num_frames=total_frames,
        limit=subset_size,
        seed=seed,
        cache_dir=cache_dir,
    )
    if len(dataset) < 3:
        raise RuntimeError("Need at least 3 videos to build a latent sequence bank.")

    model = _load_model(checkpoint_path, device=device)
    model_device = next(model.parameters()).device

    context_chunks: list[torch.Tensor] = []
    future_chunks: list[torch.Tensor] = []
    sample_indices: list[int] = []
    video_ids: list[str] = []
    pending_clips: list[torch.Tensor] = []
    pending_indices: list[int] = []
    pending_video_ids: list[str] = []

    def flush_pending() -> None:
        if not pending_clips:
            return
        batch_clips = torch.stack(pending_clips, dim=0).to(model_device)
        context_latents = _encode_latent_sequences(model, batch_clips[:, :context_frames]).detach().cpu()
        future_latents = _encode_latent_sequences(model, batch_clips[:, context_frames:]).detach().cpu()
        context_chunks.extend(context_latents.unbind(dim=0))
        future_chunks.extend(future_latents.unbind(dim=0))
        sample_indices.extend(pending_indices)
        video_ids.extend(pending_video_ids)
        pending_clips.clear()
        pending_indices.clear()
        pending_video_ids.clear()

    for dataset_index in tqdm(range(len(dataset)), desc="Encoding latent sequences", unit="sample"):
        dataset_sample = dataset.samples[dataset_index]
        video_path = Path(dataset_sample.path)
        frames = _read_video_frames(video_path)
        if not frames:
            raise RuntimeError(f"Could not decode video: {video_path}")
        window = _sample_contiguous_window(frames, total_frames, seed + dataset_index)
        clip_tensor = torch.stack([_normalize_frame(frame, image_size) for frame in window], dim=0)
        pending_clips.append(clip_tensor)
        pending_indices.append(dataset_index)
        pending_video_ids.append(dataset_sample.video_id)
        if len(pending_clips) >= batch_size:
            flush_pending()

    flush_pending()

    context_bank = torch.stack(context_chunks, dim=0)
    future_bank = torch.stack(future_chunks, dim=0)
    bank = LatentSequenceBank(
        context_latents=context_bank,
        future_latents=future_bank,
        sample_indices=sample_indices,
        video_ids=video_ids,
        source_split=source_split,
        checkpoint_path=str(checkpoint_path),
        data_root=str(data_root),
        subset_size=subset_size,
        image_size=image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bank.to_payload(), cache_path)
    return bank


def _latent_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.float()
    target = target.float()
    flat_pred = pred.reshape(-1, pred.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    normalized_pred = F.normalize(flat_pred, dim=-1)
    normalized_target = F.normalize(flat_target, dim=-1)
    return {
        "latent_mse": float(F.mse_loss(flat_pred, flat_target).item()),
        "normalized_latent_mse": float(F.mse_loss(normalized_pred, normalized_target).item()),
        "cosine_similarity": float(F.cosine_similarity(flat_pred, flat_target, dim=-1).mean().item()),
    }


def _baseline_repeat_last(context: torch.Tensor, future_steps: int) -> torch.Tensor:
    return context[:, -1:, :].expand(-1, future_steps, -1).contiguous()


def _baseline_mean(context: torch.Tensor, future_steps: int) -> torch.Tensor:
    mean = context.mean(dim=1, keepdim=True)
    return mean.expand(-1, future_steps, -1).contiguous()


def _latent_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return F.mse_loss(pred_norm, target_norm) + 0.1 * F.mse_loss(pred, target)


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    pin_memory = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=pin_memory)


def _run_epoch(
    model: TemporalLatentPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    desc: str | None = None,
    show_progress: bool = False,
) -> float:
    losses: list[float] = []
    is_train = optimizer is not None
    model.train(is_train)
    context = torch.enable_grad() if is_train else torch.no_grad()
    iterator = tqdm(loader, desc=desc, leave=False) if show_progress else loader
    with context:
        for batch in iterator:
            inputs = batch["context"].to(device)
            targets = batch["future"].to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            preds = model(inputs)
            loss = _latent_loss(preds, targets)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            losses.append(float(loss.item()))
            if show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{loss.item():.4g}")
    return float(np.mean(losses)) if losses else float("nan")


@torch.inference_mode()
def _evaluate_dataset(
    model: TemporalLatentPredictor,
    dataset: LatentWindowDataset,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float]]]:
    loader = _make_loader(dataset, batch_size=64, shuffle=False)
    model.eval()
    model_preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    repeat_preds: list[torch.Tensor] = []
    mean_preds: list[torch.Tensor] = []
    rows: list[dict[str, float]] = []

    for batch in loader:
        context = batch["context"].to(device)
        future = batch["future"].to(device)
        pred = model(context).cpu()
        target = future.cpu()
        repeat = _baseline_repeat_last(batch["context"], model.future_steps)
        mean = _baseline_mean(batch["context"], model.future_steps)
        model_preds.append(pred)
        targets.append(target)
        repeat_preds.append(repeat)
        mean_preds.append(mean)

        model_metrics = _latent_metrics(pred, target)
        repeat_metrics = _latent_metrics(repeat, target)
        mean_metrics = _latent_metrics(mean, target)
        sample_indices = batch["sample_index"].tolist()
        video_ids = list(batch["video_id"])
        for row_index, (sample_index, video_id) in enumerate(zip(sample_indices, video_ids)):
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "video_id": str(video_id),
                    "latent_mse": float(F.mse_loss(pred[row_index], target[row_index]).item()),
                    "normalized_latent_mse": float(
                        F.mse_loss(
                            F.normalize(pred[row_index].reshape(-1, pred.shape[-1]), dim=-1),
                            F.normalize(target[row_index].reshape(-1, target.shape[-1]), dim=-1),
                        ).item()
                    ),
                    "cosine_similarity": float(
                        F.cosine_similarity(
                            pred[row_index].reshape(-1, pred.shape[-1]),
                            target[row_index].reshape(-1, target.shape[-1]),
                            dim=-1,
                        ).mean().item()
                    ),
                    "repeat_last_mse": float(F.mse_loss(repeat[row_index], target[row_index]).item()),
                    "mean_context_mse": float(F.mse_loss(mean[row_index], target[row_index]).item()),
                }
            )

    model_concat = torch.cat(model_preds, dim=0)
    target_concat = torch.cat(targets, dim=0)
    repeat_concat = torch.cat(repeat_preds, dim=0)
    mean_concat = torch.cat(mean_preds, dim=0)
    metrics = _latent_metrics(model_concat, target_concat)
    baseline_metrics = {
        "repeat_last": _latent_metrics(repeat_concat, target_concat),
        "mean_context": _latent_metrics(mean_concat, target_concat),
    }
    return metrics, baseline_metrics, rows


def _write_prediction_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "video_id",
        "latent_mse",
        "normalized_latent_mse",
        "cosine_similarity",
        "repeat_last_mse",
        "mean_context_mse",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def train_video_world_model(
    *,
    checkpoint_path: str | Path,
    data_root: str | Path,
    source_split: str = "train",
    subset_size: int = 128,
    image_size: int = 224,
    total_frames: int = 16,
    context_frames: int = 12,
    future_frames: int = 4,
    feature_batch_size: int = 1,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    seed: int = 0,
    cache_dir: str | Path | None = "logs/video_world_model/cache",
    output_dir: str | Path = "logs/video_world_model",
    device: str | torch.device | None = None,
) -> LatentWorldModelResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir)

    bank = build_latent_sequence_bank(
        checkpoint_path=checkpoint_path,
        data_root=data_root,
        source_split=source_split,
        subset_size=subset_size,
        image_size=image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
        batch_size=feature_batch_size,
        cache_dir=cache_dir,
        seed=seed,
        device=device,
    )

    train_indices, val_indices, test_indices = _split_indices(len(bank.sample_indices), seed)
    train_dataset = LatentWindowDataset(
        bank.context_latents[train_indices],
        bank.future_latents[train_indices],
        [bank.sample_indices[i] for i in train_indices],
        [bank.video_ids[i] for i in train_indices],
    )
    val_dataset = LatentWindowDataset(
        bank.context_latents[val_indices],
        bank.future_latents[val_indices],
        [bank.sample_indices[i] for i in val_indices],
        [bank.video_ids[i] for i in val_indices],
    )
    test_dataset = LatentWindowDataset(
        bank.context_latents[test_indices],
        bank.future_latents[test_indices],
        [bank.sample_indices[i] for i in test_indices],
        [bank.video_ids[i] for i in test_indices],
    )

    model = TemporalLatentPredictor(
        latent_dim=bank.latent_dim,
        context_steps=bank.context_steps,
        future_steps=bank.future_steps,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
    device_obj = resolve_device(device)
    model = model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = _make_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(val_dataset, batch_size=batch_size, shuffle=False)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_train = float("nan")
    best_epoch = 0
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    epoch_bar = tqdm(range(epochs), desc="latent world model epochs")
    for epoch in epoch_bar:
        train_loss = _run_epoch(model, train_loader, optimizer, device_obj, desc=f"train {epoch + 1}/{epochs}", show_progress=True)
        val_loss = _run_epoch(model, val_loader, None, device_obj)
        epoch_bar.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}")
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )
        if math.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_train = train_loss
            best_epoch = epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    model.load_state_dict(best_state)
    model = model.to(device_obj)

    train_metrics, train_baselines, _ = _evaluate_dataset(model, train_dataset, device_obj)
    val_metrics, val_baselines, _ = _evaluate_dataset(model, val_dataset, device_obj)
    test_metrics, test_baselines, test_rows = _evaluate_dataset(model, test_dataset, device_obj)

    baseline_metrics = {
        "train": train_baselines,
        "val": val_baselines,
        "test": test_baselines,
    }

    checkpoint_name = (
        f"latent_world_model_{Path(checkpoint_path).stem}_{_checkpoint_fingerprint(checkpoint_path)}"
        f"_{source_split}_{subset_size}_{image_size}_{total_frames}_{context_frames}_{future_frames}.pt"
    )
    checkpoint_file = output_dir / checkpoint_name
    bundle = LatentWorldModelBundle(
        state_dict={name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        latent_dim=bank.latent_dim,
        hidden_dim=hidden_dim,
        context_steps=bank.context_steps,
        future_steps=bank.future_steps,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        architecture_version=1,
        checkpoint_path=str(checkpoint_path),
        source_split=source_split,
        subset_size=subset_size,
        image_size=image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
        train_loss=best_train,
        val_loss=best_val,
        best_epoch=best_epoch,
    )
    torch.save(bundle.to_payload(), checkpoint_file)

    predictions_path = output_dir / "predictions.csv"
    _write_prediction_rows(predictions_path, test_rows)

    report_path = output_dir / "metrics.json"
    result = LatentWorldModelResult(
        train_loss=float(best_train),
        val_loss=float(best_val),
        test_loss=float(test_metrics["normalized_latent_mse"]),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        checkpoint_path=str(checkpoint_file),
        cache_path=str(cache_path) if cache_path is not None else "",
        report_path=str(report_path),
        predictions_path=str(predictions_path),
        latent_shape=(bank.context_steps, bank.future_steps, bank.latent_dim),
        history=history,
        best_epoch=best_epoch,
        num_samples=len(bank.sample_indices),
        context_frames=context_frames,
        future_frames=future_frames,
        total_frames=total_frames,
        image_size=image_size,
    )
    _write_json(report_path, result.to_json())
    return result






