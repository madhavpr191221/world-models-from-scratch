from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from jepa_world_models.analysis.common import resolve_device
from jepa_world_models.analysis.video_data import SomethingSomethingVideoDataset
from jepa_world_models.analysis.video_latent_models import (
    EncoderSpec,
    PredictorSpec,
    build_temporal_predictor,
    build_video_encoder,
)


LATENT_CACHE_FORMAT_VERSION = 2
LATENT_WORLD_MODEL_ARCHITECTURE_VERSION = 2


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
    sample_fps: float | None = None
    encoder_name: str = ""
    encoder_fingerprint: str = ""
    cache_format_version: int = LATENT_CACHE_FORMAT_VERSION
    cache_kind: str = "latent_sequence_bank"
    created_at: str = ""

    def to_payload(self) -> dict:
        return {
            "manifest": self.to_manifest(),
            "payload": {
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
                "sample_fps": self.sample_fps,
                "encoder_name": self.encoder_name,
                "encoder_fingerprint": self.encoder_fingerprint,
                "cache_format_version": self.cache_format_version,
                "cache_kind": self.cache_kind,
                "created_at": self.created_at,
            },
        }

    def to_manifest(self) -> dict:
        return {
            "cache_format_version": self.cache_format_version,
            "cache_kind": self.cache_kind,
            "encoder_fingerprint": self.encoder_fingerprint,
            "checkpoint_path": self.checkpoint_path,
            "data_root": self.data_root,
            "source_split": self.source_split,
            "subset_size": self.subset_size,
            "image_size": self.image_size,
            "sample_fps": self.sample_fps,
            "encoder_name": self.encoder_name,
            "total_frames": self.total_frames,
            "context_frames": self.context_frames,
            "future_frames": self.future_frames,
            "latent_dim": self.latent_dim,
            "num_samples": len(self.sample_indices),
            "video_ids": self.video_ids,
            "sample_indices": self.sample_indices,
            "encoder_name": self.encoder_name,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }

    @property
    def content_hash(self) -> str:
        payload = {
            "checkpoint_path": self.checkpoint_path,
            "data_root": self.data_root,
            "source_split": self.source_split,
            "subset_size": self.subset_size,
            "image_size": self.image_size,
            "sample_fps": self.sample_fps,
            "encoder_name": self.encoder_name,
            "total_frames": self.total_frames,
            "context_frames": self.context_frames,
            "future_frames": self.future_frames,
            "sample_indices": self.sample_indices,
            "video_ids": self.video_ids,
            "encoder_name": self.encoder_name,
            "encoder_fingerprint": self.encoder_fingerprint,
        }
        digest = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(digest).hexdigest()

    @classmethod
    def from_payload(cls, payload: dict) -> "LatentSequenceBank":
        if "payload" in payload and isinstance(payload["payload"], dict):
            payload = payload["payload"]
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
            sample_fps=None if payload.get("sample_fps") is None else float(payload["sample_fps"]),
            encoder_name=str(payload.get("encoder_name", "")),
            encoder_fingerprint=str(payload.get("encoder_fingerprint", "")),
            cache_format_version=int(payload.get("cache_format_version", LATENT_CACHE_FORMAT_VERSION)),
            cache_kind=str(payload.get("cache_kind", "latent_sequence_bank")),
            created_at=str(payload.get("created_at", "")),
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
    context_lag_steps: int
    future_steps: int
    num_layers: int
    num_heads: int
    dropout: float
    objective_name: str
    rollout_decay: float
    architecture_version: int
    checkpoint_path: str
    source_split: str
    encoder_name: str
    predictor_mode: str
    predictor_name: str
    subset_size: int
    image_size: int
    total_frames: int
    context_frames: int
    future_frames: int
    train_loss: float | None = None
    val_loss: float | None = None
    best_epoch: int | None = None
    epoch: int | None = None

    def to_payload(self) -> dict:
        return {
            "state_dict": self.state_dict,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "context_steps": self.context_steps,
            "context_lag_steps": self.context_lag_steps,
            "future_steps": self.future_steps,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "objective_name": self.objective_name,
            "rollout_decay": self.rollout_decay,
            "architecture_version": self.architecture_version,
            "checkpoint_path": self.checkpoint_path,
            "source_split": self.source_split,
            "encoder_name": self.encoder_name,
            "predictor_mode": self.predictor_mode,
            "predictor_name": self.predictor_name,
            "subset_size": self.subset_size,
            "image_size": self.image_size,
            "total_frames": self.total_frames,
            "context_frames": self.context_frames,
            "future_frames": self.future_frames,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_epoch": self.best_epoch,
            "epoch": self.epoch,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "LatentWorldModelBundle":
        return cls(
            state_dict=payload["state_dict"],
            latent_dim=int(payload["latent_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            context_steps=int(payload["context_steps"]),
            context_lag_steps=int(payload.get("context_lag_steps", payload["context_steps"])),
            future_steps=int(payload["future_steps"]),
            num_layers=int(payload["num_layers"]),
            num_heads=int(payload["num_heads"]),
            dropout=float(payload["dropout"]),
            objective_name=str(payload.get("objective_name", "balanced")),
            rollout_decay=float(payload.get("rollout_decay", 1.0)),
            architecture_version=int(payload.get("architecture_version", 1)),
            checkpoint_path=str(payload["checkpoint_path"]),
            source_split=str(payload["source_split"]),
            encoder_name=str(payload.get("encoder_name", "")),
            predictor_mode=str(payload.get("predictor_mode", "context")),
            predictor_name=str(payload.get("predictor_name", "causal_transformer")),
            subset_size=int(payload["subset_size"]),
            image_size=int(payload["image_size"]),
            total_frames=int(payload["total_frames"]),
            context_frames=int(payload["context_frames"]),
            future_frames=int(payload["future_frames"]),
            train_loss=None if payload.get("train_loss") is None else float(payload["train_loss"]),
            val_loss=None if payload.get("val_loss") is None else float(payload["val_loss"]),
            best_epoch=None if payload.get("best_epoch") is None else int(payload["best_epoch"]),
            epoch=None if payload.get("epoch") is None else int(payload["epoch"]),
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
    baseline_verdict: dict[str, dict[str, object]]
    checkpoint_path: str
    checkpoint_dir: str
    encoder_name: str
    predictor_name: str
    objective_name: str
    rollout_decay: float
    cache_path: str
    cache_manifest_path: str
    report_path: str
    predictions_path: str
    latent_shape: tuple[int, int, int]
    predictor_mode: str
    context_lag_steps: int
    history: list[dict[str, float]]
    step_history: list[dict[str, float]]
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
            "baseline_verdict": self.baseline_verdict,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_dir": self.checkpoint_dir,
            "encoder_name": self.encoder_name,
            "predictor_name": self.predictor_name,
            "objective_name": self.objective_name,
            "rollout_decay": self.rollout_decay,
            "cache_path": self.cache_path,
            "cache_manifest_path": self.cache_manifest_path,
            "report_path": self.report_path,
            "predictions_path": self.predictions_path,
            "latent_shape": list(self.latent_shape),
            "predictor_mode": self.predictor_mode,
            "context_lag_steps": self.context_lag_steps,
            "history": self.history,
            "step_history": self.step_history,
            "best_epoch": self.best_epoch,
            "num_samples": self.num_samples,
            "context_frames": self.context_frames,
            "future_frames": self.future_frames,
            "total_frames": self.total_frames,
            "image_size": self.image_size,
        }


def _is_windows() -> bool:
    return os.name == "nt"


def _resolve_dataloader_settings(requested_workers: int, batch_size: int) -> dict[str, object]:
    """Pick a safer worker configuration for large video tensors.

    Windows is much more sensitive to shared-memory pressure in multi-worker
    collation, so we back off the worker count and prefetch depth there.
    """
    if not torch.cuda.is_available():
        return {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }

    if requested_workers <= 0:
        return {
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": False,
        }

    if _is_windows():
        # Large video batches can exhaust paging/shared-memory very quickly on Windows.
        # Keep some parallelism, but avoid a worker explosion.
        num_workers = max(2, min(requested_workers, 4))
        prefetch_factor = 2 if batch_size <= 16 else 1
        return {
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
            "prefetch_factor": prefetch_factor,
        }

    num_workers = max(0, requested_workers)
    settings: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        settings["prefetch_factor"] = 4 if batch_size <= 32 else 2
    return settings
@dataclass(slots=True)
class EpochRunResult:
    loss: float
    objective_loss: float
    mse_loss: float
    normalized_mse_loss: float
    cosine_loss: float
    rollout_loss: float
    delta_loss: float
    step_history: list[dict[str, float]]

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


def _checkpoint_fingerprint(checkpoint_path: str | Path) -> str:
    path = Path(checkpoint_path).resolve()
    stat = path.stat()
    digest = f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(digest).hexdigest()[:10]


def _latent_sequence_bank_cache_name(
    *,
    checkpoint_path: str | Path,
    source_split: str,
    subset_size: int,
    image_size: int,
    total_frames: int,
    context_frames: int,
    future_frames: int,
    sample_fps: float | None,
    seed: int,
) -> str:
    checkpoint_stem = Path(checkpoint_path).stem
    checkpoint_id = _checkpoint_fingerprint(checkpoint_path)
    sample_fps_tag = "na" if sample_fps is None else f"{sample_fps:g}".replace(".", "p")
    return (
        f"latent_sequence_bank_{checkpoint_stem}_{checkpoint_id}_{source_split}_{subset_size}"
        f"_{image_size}_{total_frames}_{context_frames}_{future_frames}_{sample_fps_tag}_{seed}.pt"
    )


def _latent_sequence_bank_cache_path(
    *,
    cache_dir: str | Path,
    checkpoint_path: str | Path,
    source_split: str,
    subset_size: int,
    image_size: int,
    total_frames: int,
    context_frames: int,
    future_frames: int,
    sample_fps: float | None,
    seed: int,
) -> Path:
    return Path(cache_dir) / _latent_sequence_bank_cache_name(
        checkpoint_path=checkpoint_path,
        source_split=source_split,
        subset_size=subset_size,
        image_size=image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
        sample_fps=sample_fps,
        seed=seed,
    )


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


def _load_model(checkpoint_path: str | Path, device: str | torch.device | None = None) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    encoder = build_video_encoder(
        EncoderSpec(
            name=str(config.get("encoder_name", "")),
            latent_dim=int(config.get("latent_dim", 192)),
            tubelet_size=int(config.get("tubelet_size", 2)),
            pretrained=bool(config.get("pretrained", False)),
            variant=str(config.get("variant", "t")),
        )
    )
    encoder.load_state_dict(checkpoint["model_state"], strict=False)
    encoder.eval()
    device_obj = resolve_device(device)
    return encoder.to(device_obj)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_cache_artifact(cache_path: Path, bank: LatentSequenceBank) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_path.with_suffix(".json")
    payload = bank.to_payload()
    torch.save(payload, cache_path)
    _write_json(manifest_path, payload["manifest"])
    return manifest_path


def _load_cache_artifact(cache_path: Path) -> tuple[LatentSequenceBank, bool]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "payload" in payload:
        return LatentSequenceBank.from_payload(payload), False
    return LatentSequenceBank.from_payload(payload), True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    model: nn.Module,
    clips: torch.Tensor,
) -> torch.Tensor:
    if hasattr(model, "forward_sequence"):
        return model.forward_sequence(clips)
    if hasattr(model, "encoder") and hasattr(model.encoder, "forward_sequence"):
        return model.encoder.forward_sequence(clips)
    raise TypeError("Encoder must implement forward_sequence().")


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
    sample_fps: float | None = None,
    batch_size: int = 1,
    num_workers: int = 4,
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
        cache_path = _latent_sequence_bank_cache_path(
            cache_dir=cache_dir,
            checkpoint_path=checkpoint_path,
            source_split=source_split,
            subset_size=subset_size,
            image_size=image_size,
            total_frames=total_frames,
            context_frames=context_frames,
            future_frames=future_frames,
            sample_fps=sample_fps,
            seed=seed,
        )
        if cache_path.exists():
            bank, legacy = _load_cache_artifact(cache_path)
            if not bank.sample_fps:
                bank.sample_fps = sample_fps
            if not bank.encoder_name:
                bank.encoder_name = str(torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("config", {}).get("encoder_name", ""))
            if not bank.encoder_fingerprint:
                bank.encoder_fingerprint = _checkpoint_fingerprint(checkpoint_path)
            if not bank.created_at:
                bank.created_at = _utc_now_iso()
            if legacy or not cache_path.with_suffix(".json").exists():
                _write_cache_artifact(cache_path, bank)
            return bank

    data_root = Path(data_root)
    _resolve_video_root(data_root)
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_name = str(checkpoint.get("config", {}).get("encoder_name", ""))

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        **_resolve_dataloader_settings(num_workers, batch_size),
    }
    loader = DataLoader(dataset, **loader_kwargs)

    context_chunks: list[torch.Tensor] = []
    future_chunks: list[torch.Tensor] = []
    sample_indices: list[int] = []
    video_ids: list[str] = []
    sample_cursor = 0

    for batch in tqdm(loader, desc="Encoding latent sequences", unit="sample"):
        clips = batch["clip"]
        video_batch = batch["video_id"]
        if isinstance(video_batch, str):
            video_batch = [video_batch]
        batch_size_actual = clips.shape[0]
        batch_clips = clips.to(model_device, non_blocking=True)
        context_latents = _encode_latent_sequences(model, batch_clips[:, :context_frames]).detach().cpu()
        future_latents = _encode_latent_sequences(model, batch_clips[:, context_frames:]).detach().cpu()
        context_chunks.extend(context_latents.unbind(dim=0))
        future_chunks.extend(future_latents.unbind(dim=0))
        sample_indices.extend(range(sample_cursor, sample_cursor + batch_size_actual))
        video_ids.extend([str(video_id) for video_id in video_batch])
        sample_cursor += batch_size_actual

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
        sample_fps=sample_fps,
        encoder_name=encoder_name,
        encoder_fingerprint=_checkpoint_fingerprint(checkpoint_path),
        created_at=_utc_now_iso(),
    )
    if cache_path is not None:
        _write_cache_artifact(cache_path, bank)
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


def _rollout_weights(future_steps: int, decay: float, device: torch.device) -> torch.Tensor:
    if future_steps <= 0:
        raise ValueError("future_steps must be positive.")
    if decay <= 0:
        raise ValueError("rollout_decay must be positive.")
    weights = torch.tensor([decay**index for index in range(future_steps)], dtype=torch.float32, device=device)
    return weights / weights.sum().clamp_min(1e-12)


def _direct_loss_components(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    flat_pred = pred.float().reshape(-1, pred.shape[-1])
    flat_target = target.float().reshape(-1, target.shape[-1])
    normalized_pred = F.normalize(flat_pred, dim=-1)
    normalized_target = F.normalize(flat_target, dim=-1)
    mse_loss = F.mse_loss(flat_pred, flat_target)
    normalized_mse_loss = F.mse_loss(normalized_pred, normalized_target)
    cosine_loss = 1.0 - F.cosine_similarity(flat_pred, flat_target, dim=-1).mean()
    return {
        "mse_loss": mse_loss,
        "normalized_mse_loss": normalized_mse_loss,
        "cosine_loss": cosine_loss,
    }


def _objective_loss_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    context: torch.Tensor,
    objective_name: str,
    rollout_decay: float,
) -> dict[str, torch.Tensor]:
    objective = objective_name.strip().lower()
    direct = _direct_loss_components(pred, target)
    device = pred.device
    future_steps = int(pred.shape[1])
    weights = _rollout_weights(future_steps, rollout_decay, device)

    if objective in {"balanced", "combined", "default"}:
        objective_loss = direct["normalized_mse_loss"] + 0.1 * direct["mse_loss"] + 0.1 * direct["cosine_loss"]
        rollout_loss = objective_loss
        delta_loss = objective_loss
    elif objective == "mse":
        objective_loss = direct["mse_loss"]
        rollout_loss = objective_loss
        delta_loss = objective_loss
    elif objective == "normalized_mse":
        objective_loss = direct["normalized_mse_loss"]
        rollout_loss = objective_loss
        delta_loss = objective_loss
    elif objective == "cosine":
        objective_loss = direct["cosine_loss"]
        rollout_loss = objective_loss
        delta_loss = objective_loss
    else:
        rollout_pred = pred
        rollout_target = target
        context_anchor = context[:, -1:, :].expand_as(pred)
        if objective in {"delta_balanced", "delta_rollout_balanced"}:
            rollout_pred = pred - context_anchor
            rollout_target = target - context_anchor
        per_step_mse = ((rollout_pred.float() - rollout_target.float()) ** 2).mean(dim=-1)
        per_step_norm = (
            F.mse_loss(
                F.normalize(rollout_pred.float(), dim=-1),
                F.normalize(rollout_target.float(), dim=-1),
                reduction="none",
            )
            .mean(dim=-1)
        )
        per_step_cosine = 1.0 - F.cosine_similarity(
            rollout_pred.float().reshape(-1, rollout_pred.shape[-1]),
            rollout_target.float().reshape(-1, rollout_target.shape[-1]),
            dim=-1,
        ).reshape(rollout_pred.shape[0], rollout_pred.shape[1])
        rollout_mse = (per_step_mse * weights).sum()
        rollout_norm = (per_step_norm * weights).sum()
        rollout_cosine = (per_step_cosine * weights).sum()
        rollout_loss = rollout_norm + 0.1 * rollout_mse + 0.1 * rollout_cosine
        delta_loss = rollout_loss
        if objective == "delta_balanced":
            objective_loss = rollout_loss
        elif objective == "delta_rollout_balanced":
            objective_loss = rollout_loss
        elif objective == "rollout_balanced":
            objective_loss = rollout_loss
        else:
            raise ValueError(
                f"Unsupported objective_name={objective_name}. "
                "Use balanced, mse, normalized_mse, cosine, rollout_balanced, delta_balanced, or delta_rollout_balanced."
            )

    return {
        "combined_loss": objective_loss,
        "objective_loss": objective_loss,
        "mse_loss": direct["mse_loss"],
        "normalized_mse_loss": direct["normalized_mse_loss"],
        "cosine_loss": direct["cosine_loss"],
        "rollout_loss": rollout_loss,
        "delta_loss": delta_loss,
    }


def _resolve_context_lag_steps(
    context_steps: int,
    *,
    predictor_mode: str,
    context_lag_steps: int | None,
) -> int:
    if context_lag_steps is None:
        mode = predictor_mode.strip().lower()
        if mode == "one-lag":
            return 1
        if mode in {"context", "multi-context", "multi"}:
            return context_steps
        raise ValueError(f"Unsupported predictor_mode: {predictor_mode}")
    if context_lag_steps <= 0:
        raise ValueError("context_lag_steps must be positive.")
    if context_lag_steps > context_steps:
        raise ValueError(
            f"context_lag_steps={context_lag_steps} cannot exceed available context_steps={context_steps}."
        )
    return context_lag_steps


def _select_context_window(context_latents: torch.Tensor, context_lag_steps: int) -> torch.Tensor:
    if context_lag_steps <= 0:
        raise ValueError("context_lag_steps must be positive.")
    if context_lag_steps > context_latents.shape[1]:
        raise ValueError(
            f"context_lag_steps={context_lag_steps} cannot exceed the available context window {context_latents.shape[1]}."
        )
    return context_latents[:, -context_lag_steps:, :].contiguous()


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 4) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        **_resolve_dataloader_settings(num_workers, batch_size),
    }
    return DataLoader(dataset, **kwargs)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    objective_name: str,
    rollout_decay: float,
    desc: str | None = None,
    show_progress: bool = False,
    epoch_index: int | None = None,
    collect_step_history: bool = False,
) -> EpochRunResult:
    is_train = optimizer is not None
    model.train(is_train)
    context = torch.enable_grad() if is_train else torch.no_grad()
    iterator = tqdm(loader, desc=desc, leave=False) if show_progress else loader
    totals = {
        "combined_loss": 0.0,
        "objective_loss": 0.0,
        "mse_loss": 0.0,
        "normalized_mse_loss": 0.0,
        "cosine_loss": 0.0,
        "rollout_loss": 0.0,
        "delta_loss": 0.0,
    }
    step_history: list[dict[str, float]] = []
    num_batches = 0
    with context:
        for batch_index, batch in enumerate(iterator):
            inputs = batch["context"].to(device)
            targets = batch["future"].to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            preds = model(inputs)
            loss_components = _objective_loss_components(
                preds,
                targets,
                context=inputs,
                objective_name=objective_name,
                rollout_decay=rollout_decay,
            )
            loss = loss_components["combined_loss"]
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            num_batches += 1
            for key in totals:
                totals[key] += float(loss_components[key].item())
            if collect_step_history and is_train:
                record = {
                    "batch_index": float(batch_index),
                    "combined_loss": float(loss.item()),
                    "objective_loss": float(loss_components["objective_loss"].item()),
                    "mse_loss": float(loss_components["mse_loss"].item()),
                    "normalized_mse_loss": float(loss_components["normalized_mse_loss"].item()),
                    "cosine_loss": float(loss_components["cosine_loss"].item()),
                    "rollout_loss": float(loss_components["rollout_loss"].item()),
                    "delta_loss": float(loss_components["delta_loss"].item()),
                    "latent_mse": float(loss_components["mse_loss"].item()),
                    "normalized_latent_mse": float(loss_components["normalized_mse_loss"].item()),
                    "cosine_similarity": float(1.0 - loss_components["cosine_loss"].item()),
                    "objective_name": objective_name,
                }
                if epoch_index is not None:
                    record["epoch"] = float(epoch_index)
                step_history.append(record)
            if show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    loss=f"{loss.item():.4g}",
                    mse=f"{loss_components['mse_loss'].item():.4g}",
                    norm=f"{loss_components['normalized_mse_loss'].item():.4g}",
                )
    if num_batches == 0:
        return EpochRunResult(
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            step_history,
        )
    scale = 1.0 / num_batches
    return EpochRunResult(
        loss=totals["combined_loss"] * scale,
        objective_loss=totals["objective_loss"] * scale,
        mse_loss=totals["mse_loss"] * scale,
        normalized_mse_loss=totals["normalized_mse_loss"] * scale,
        cosine_loss=totals["cosine_loss"] * scale,
        rollout_loss=totals["rollout_loss"] * scale,
        delta_loss=totals["delta_loss"] * scale,
        step_history=step_history,
    )


@torch.inference_mode()
def _evaluate_dataset(
    model: nn.Module,
    dataset: LatentWindowDataset,
    device: torch.device,
    *,
    objective_name: str,
    rollout_decay: float,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float]], float]:
    loader = _make_loader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    model_preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    repeat_preds: list[torch.Tensor] = []
    mean_preds: list[torch.Tensor] = []
    rows: list[dict[str, float]] = []
    objective_losses: list[torch.Tensor] = []

    for batch in loader:
        context = batch["context"].to(device)
        future = batch["future"].to(device)
        pred = model(context).cpu()
        target = future.cpu()
        objective_components = _objective_loss_components(
            pred,
            target,
            context=batch["context"],
            objective_name=objective_name,
            rollout_decay=rollout_decay,
        )
        objective_losses.append(objective_components["objective_loss"].detach().cpu())
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
    objective_loss = float(torch.stack(objective_losses).mean().item()) if objective_losses else float("nan")
    return metrics, baseline_metrics, rows, objective_loss


def _compare_against_baselines(
    model_metrics: dict[str, float], baseline_metrics: dict[str, dict[str, float]]
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    for metric_name in ("latent_mse", "normalized_latent_mse", "cosine_similarity"):
        repeat_value = float(baseline_metrics["repeat_last"][metric_name])
        mean_value = float(baseline_metrics["mean_context"][metric_name])
        model_value = float(model_metrics[metric_name])
        if metric_name == "cosine_similarity":
            best_baseline = max(repeat_value, mean_value)
            better = model_value > best_baseline
            direction = "higher_is_better"
        else:
            best_baseline = min(repeat_value, mean_value)
            better = model_value < best_baseline
            direction = "lower_is_better"
        comparisons[metric_name] = {
            "model": model_value,
            "repeat_last": repeat_value,
            "mean_context": mean_value,
            "best_baseline": best_baseline,
            "better_than_best_baseline": better,
            "direction": direction,
        }
    comparisons["overall"] = all(
        bool(comparisons[metric_name]["better_than_best_baseline"]) for metric_name in ("latent_mse", "normalized_latent_mse", "cosine_similarity")
    )
    comparisons["verdict"] = (
        "beat_baselines" if comparisons["overall"] else "did_not_beat_baselines"
    )
    return comparisons

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
    validation_source_split: str | None = None,
    test_source_split: str | None = None,
    subset_size: int = 128,
    image_size: int = 224,
    total_frames: int = 16,
    context_frames: int = 12,
    future_frames: int = 4,
    sample_fps: float | None = None,
    feature_batch_size: int = 1,
    num_workers: int = 4,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    predictor_name: str = "causal_transformer",
    predictor_mode: str = "context",
    context_lag_steps: int | None = None,
    objective_name: str = "balanced",
    rollout_decay: float = 1.0,
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
        cache_path = _latent_sequence_bank_cache_path(
            cache_dir=cache_dir,
            checkpoint_path=checkpoint_path,
            source_split=source_split,
            subset_size=subset_size,
            image_size=image_size,
            total_frames=total_frames,
            context_frames=context_frames,
            future_frames=future_frames,
            sample_fps=sample_fps,
            seed=seed,
        )

    bank = build_latent_sequence_bank(
        checkpoint_path=checkpoint_path,
        data_root=data_root,
        source_split=source_split,
        subset_size=subset_size,
        image_size=image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
        sample_fps=sample_fps,
        batch_size=feature_batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        seed=seed,
        device=device,
    )

    validation_bank = None
    if validation_source_split is not None:
        validation_bank = build_latent_sequence_bank(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            source_split=validation_source_split,
            subset_size=subset_size,
            image_size=image_size,
            total_frames=total_frames,
            context_frames=context_frames,
            future_frames=future_frames,
            sample_fps=sample_fps,
            batch_size=feature_batch_size,
            num_workers=num_workers,
            cache_dir=cache_dir,
            seed=seed,
            device=device,
        )

    test_bank = None
    if test_source_split is not None:
        test_bank = build_latent_sequence_bank(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            source_split=test_source_split,
            subset_size=subset_size,
            image_size=image_size,
            total_frames=total_frames,
            context_frames=context_frames,
            future_frames=future_frames,
            sample_fps=sample_fps,
            batch_size=feature_batch_size,
            num_workers=num_workers,
            cache_dir=cache_dir,
            seed=seed,
            device=device,
        )

    resolved_context_lag_steps = _resolve_context_lag_steps(
        bank.context_steps,
        predictor_mode=predictor_mode,
        context_lag_steps=context_lag_steps,
    )
    objective_name = objective_name.strip().lower()

    train_context_latents = _select_context_window(bank.context_latents, resolved_context_lag_steps)
    train_dataset = LatentWindowDataset(
        train_context_latents,
        bank.future_latents,
        list(bank.sample_indices),
        list(bank.video_ids),
    )

    if validation_bank is not None:
        val_context_latents = _select_context_window(validation_bank.context_latents, resolved_context_lag_steps)
        val_dataset = LatentWindowDataset(
            val_context_latents,
            validation_bank.future_latents,
            list(validation_bank.sample_indices),
            list(validation_bank.video_ids),
        )
    else:
        train_indices, val_indices, _ = _split_indices(len(bank.sample_indices), seed)
        val_context_latents = train_context_latents[val_indices]
        val_dataset = LatentWindowDataset(
            val_context_latents,
            bank.future_latents[val_indices],
            [bank.sample_indices[i] for i in val_indices],
            [bank.video_ids[i] for i in val_indices],
        )

    if test_bank is not None:
        test_context_latents = _select_context_window(test_bank.context_latents, resolved_context_lag_steps)
        test_dataset = LatentWindowDataset(
            test_context_latents,
            test_bank.future_latents,
            list(test_bank.sample_indices),
            list(test_bank.video_ids),
        )
    else:
        train_indices, val_indices, test_indices = _split_indices(len(bank.sample_indices), seed)
        test_context_latents = train_context_latents[test_indices]
        test_dataset = LatentWindowDataset(
            test_context_latents,
            bank.future_latents[test_indices],
            [bank.sample_indices[i] for i in test_indices],
            [bank.video_ids[i] for i in test_indices],
        )

    context_latents = train_context_latents

    encoder_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_config = encoder_checkpoint.get("config", {})
    encoder_name = str(encoder_config.get("encoder_name", ""))

    model = build_temporal_predictor(
        PredictorSpec(
            name=predictor_name,
            latent_dim=bank.latent_dim,
            context_steps=context_latents.shape[1],
            future_steps=bank.future_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    )
    device_obj = resolve_device(device)
    model = model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = _make_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(val_dataset, batch_size=batch_size, shuffle=False)

    history: list[dict[str, float]] = []
    step_history: list[dict[str, float]] = []
    best_val = float("inf")
    best_train = float("nan")
    best_epoch = 0
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epoch_bar = tqdm(range(epochs), desc="latent world model epochs")
    for epoch in epoch_bar:
        train_result = _run_epoch(
            model,
            train_loader,
            optimizer,
            device_obj,
            objective_name=objective_name,
            rollout_decay=rollout_decay,
            desc=f"train {epoch + 1}/{epochs}",
            show_progress=True,
            epoch_index=epoch + 1,
            collect_step_history=True,
        )
        val_result = _run_epoch(
            model,
            val_loader,
            None,
            device_obj,
            objective_name=objective_name,
            rollout_decay=rollout_decay,
        )
        epoch_bar.set_postfix(train=f"{train_result.loss:.6f}", val=f"{val_result.loss:.6f}")
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_result.loss),
                "train_objective_loss": float(train_result.objective_loss),
                "train_mse_loss": float(train_result.mse_loss),
                "train_normalized_mse_loss": float(train_result.normalized_mse_loss),
                "train_cosine_loss": float(train_result.cosine_loss),
                "train_rollout_loss": float(train_result.rollout_loss),
                "train_delta_loss": float(train_result.delta_loss),
                "train_latent_mse": float(train_result.mse_loss),
                "train_normalized_latent_mse": float(train_result.normalized_mse_loss),
                "train_cosine_similarity": float(1.0 - train_result.cosine_loss),
                "val_loss": float(val_result.loss),
                "val_objective_loss": float(val_result.objective_loss),
                "val_mse_loss": float(val_result.mse_loss),
                "val_normalized_mse_loss": float(val_result.normalized_mse_loss),
                "val_cosine_loss": float(val_result.cosine_loss),
                "val_rollout_loss": float(val_result.rollout_loss),
                "val_delta_loss": float(val_result.delta_loss),
                "val_latent_mse": float(val_result.mse_loss),
                "val_normalized_latent_mse": float(val_result.normalized_mse_loss),
                "val_cosine_similarity": float(1.0 - val_result.cosine_loss),
            }
        )
        step_history.extend(train_result.step_history)
        if math.isfinite(val_result.loss) and val_result.loss < best_val:
            best_val = val_result.loss
            best_train = train_result.loss
            best_epoch = epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        epoch_bundle = LatentWorldModelBundle(
            state_dict={name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            latent_dim=bank.latent_dim,
            hidden_dim=hidden_dim,
            context_steps=context_latents.shape[1],
            context_lag_steps=resolved_context_lag_steps,
            future_steps=bank.future_steps,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            objective_name=objective_name,
            rollout_decay=rollout_decay,
            architecture_version=LATENT_WORLD_MODEL_ARCHITECTURE_VERSION,
            checkpoint_path=str(checkpoint_path),
            source_split=source_split,
            encoder_name=encoder_name,
            predictor_mode=predictor_mode,
            predictor_name=predictor_name,
            subset_size=subset_size,
            image_size=image_size,
            total_frames=total_frames,
            context_frames=context_frames,
            future_frames=future_frames,
            train_loss=float(train_result.loss),
            val_loss=float(val_result.loss),
            best_epoch=best_epoch if best_epoch > 0 else None,
            epoch=epoch + 1,
        )
        torch.save(epoch_bundle.to_payload(), checkpoint_dir / f"decoder_{predictor_name}_epoch_{epoch + 1:03d}.pt")

    model.load_state_dict(best_state)
    model = model.to(device_obj)

    train_metrics, train_baselines, _, train_objective_loss = _evaluate_dataset(
        model,
        train_dataset,
        device_obj,
        objective_name=objective_name,
        rollout_decay=rollout_decay,
    )
    val_metrics, val_baselines, _, val_objective_loss = _evaluate_dataset(
        model,
        val_dataset,
        device_obj,
        objective_name=objective_name,
        rollout_decay=rollout_decay,
    )
    test_metrics, test_baselines, test_rows, test_objective_loss = _evaluate_dataset(
        model,
        test_dataset,
        device_obj,
        objective_name=objective_name,
        rollout_decay=rollout_decay,
    )

    baseline_metrics = {
        "train": train_baselines,
        "val": val_baselines,
        "test": test_baselines,
    }
    baseline_verdict = {
        "train": _compare_against_baselines(train_metrics, train_baselines),
        "val": _compare_against_baselines(val_metrics, val_baselines),
        "test": _compare_against_baselines(test_metrics, test_baselines),
    }

    checkpoint_name = (
        f"decoder_{predictor_name}.pt"
    )
    checkpoint_file = output_dir / checkpoint_name
    bundle = LatentWorldModelBundle(
        state_dict={name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        latent_dim=bank.latent_dim,
        hidden_dim=hidden_dim,
        context_steps=context_latents.shape[1],
        context_lag_steps=resolved_context_lag_steps,
        future_steps=bank.future_steps,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        objective_name=objective_name,
        rollout_decay=rollout_decay,
        architecture_version=LATENT_WORLD_MODEL_ARCHITECTURE_VERSION,
        checkpoint_path=str(checkpoint_path),
        source_split=source_split,
        encoder_name=encoder_name,
        predictor_mode=predictor_mode,
        predictor_name=predictor_name,
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
        test_loss=float(test_objective_loss),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        baseline_verdict=baseline_verdict,
        checkpoint_path=str(checkpoint_file),
        checkpoint_dir=str(checkpoint_dir),
        encoder_name=encoder_name,
        predictor_name=predictor_name,
        objective_name=objective_name,
        rollout_decay=rollout_decay,
        cache_path=str(cache_path) if cache_path is not None else "",
        cache_manifest_path=str(cache_path.with_suffix(".json")) if cache_path is not None else "",
        report_path=str(report_path),
        predictions_path=str(predictions_path),
        latent_shape=(context_latents.shape[1], bank.future_steps, bank.latent_dim),
        predictor_mode=predictor_mode,
        context_lag_steps=resolved_context_lag_steps,
        history=history,
        step_history=step_history,
        best_epoch=best_epoch,
        num_samples=len(bank.sample_indices),
        context_frames=context_frames,
        future_frames=future_frames,
        total_frames=total_frames,
        image_size=image_size,
    )
    _write_json(report_path, result.to_json())
    return result












