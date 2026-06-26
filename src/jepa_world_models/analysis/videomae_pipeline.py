"""VideoMAE pretraining and feature extraction for video clips.

This module implements a compact VideoMAE-style pipeline:
- sample short clips from Something-Something V2 videos
- mask most spatiotemporal tokens
- reconstruct the masked tokens
- save a pretrained encoder for downstream probing

The goal is clarity and debuggability, not a reproduction of every VideoMAE
paper detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import json
import math
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


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


def _resolve_video_root(data_root: Path) -> Path:
    candidates = [
        data_root / "something_v2" / "20bn-something-something-v2",
        data_root / "20bn-something-something-v2",
    ]
    resolved = _first_existing(candidates)
    if resolved is None:
        raise FileNotFoundError("Could not find Something-Something V2 video root.")
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
        self.video_root = _resolve_video_root(self.data_root)
        self.labels_path = _resolve_labels_path(self.data_root, split)
        self.image_size = image_size
        self.num_frames = num_frames
        self.rng = random.Random(seed)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.samples = self._load_samples(limit=limit)

    def _sample_cache_path(self, limit: int | None) -> Path | None:
        if self.cache_dir is None:
            return None
        key = f"{self.labels_path.stem}_{self.image_size}_{self.num_frames}_{limit or 'all'}"
        return self.cache_dir / "videomae_samples" / f"{key}.json"

    def _load_samples(self, limit: int | None) -> list[VideoSample]:
        cache_path = self._sample_cache_path(limit)
        if cache_path is not None and cache_path.exists():
            payload = _load_json_file(cache_path)
            return [VideoSample(**item) for item in payload]

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
            "video_id": sample.video_id,
            "label": sample.label,
            "clip": clip,
        }


class TubePatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 192, patch_size: int = 16, tubelet_size: int = 2) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class VideoMAEEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        num_frames: int = 16,
        patch_size: int = 16,
        tubelet_size: int = 2,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0
        assert num_frames % tubelet_size == 0
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.base_temporal_steps = num_frames // tubelet_size
        self.spatial_tokens = (image_size // patch_size) ** 2
        self.patch_embed = TubePatchEmbed(3, embed_dim, patch_size, tubelet_size)
        num_tokens = self.base_temporal_steps * self.spatial_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pos_embed_for_frames(self, num_frames: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        temporal_steps = num_frames // self.tubelet_size
        if temporal_steps <= 0:
            raise ValueError("num_frames is too small for the configured tubelet size.")
        if temporal_steps == self.base_temporal_steps:
            return self.pos_embed.to(device=device, dtype=dtype)
        spatial_side = self.image_size // self.patch_size
        pos = self.pos_embed.reshape(1, self.base_temporal_steps, spatial_side, spatial_side, self.embed_dim)
        pos = pos.permute(0, 4, 1, 2, 3)
        pos = F.interpolate(
            pos,
            size=(temporal_steps, spatial_side, spatial_side),
            mode="trilinear",
            align_corners=False,
        )
        pos = pos.permute(0, 2, 3, 4, 1).contiguous()
        return pos.reshape(1, temporal_steps * self.spatial_tokens, self.embed_dim).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        pos = self._pos_embed_for_frames(x.shape[1], device=tokens.device, dtype=tokens.dtype)
        tokens = tokens + pos[:, : tokens.shape[1]]
        tokens = self.blocks(tokens)
        return self.norm(tokens)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return mean spatial tokens per temporal slice: (B, T', D)."""
        tokens = self.patch_embed(x)
        pos = self._pos_embed_for_frames(x.shape[1], device=tokens.device, dtype=tokens.dtype)
        tokens = tokens + pos[:, : tokens.shape[1]]
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        batch, seq_len, dim = tokens.shape
        temporal_steps = x.shape[1] // self.tubelet_size
        spatial_tokens = seq_len // temporal_steps
        return tokens.view(batch, temporal_steps, spatial_tokens, dim).mean(dim=2)
class VideoMAEDecoder(nn.Module):
    def __init__(self, embed_dim: int = 192, decoder_dim: int = 128, depth: int = 2, num_heads: int = 4) -> None:
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, decoder_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)


class VideoMAEModel(nn.Module):
    def __init__(self, image_size: int = 224, num_frames: int = 16, embed_dim: int = 192) -> None:
        super().__init__()
        self.encoder = VideoMAEEncoder(image_size=image_size, num_frames=num_frames, embed_dim=embed_dim)
        self.decoder = VideoMAEDecoder(embed_dim=embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_frames = num_frames
        self.image_size = image_size
        self.embed_dim = embed_dim
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @staticmethod
    def _mask_tokens(tokens: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, dim = tokens.shape
        num_visible = max(1, int(seq_len * (1.0 - mask_ratio)))
        noise = torch.rand(batch, seq_len, device=tokens.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :num_visible]
        visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, dim))
        mask = torch.ones(batch, seq_len, device=tokens.device)
        mask[:, :num_visible] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return visible, mask

    def forward(self, clips: torch.Tensor, mask_ratio: float = 0.75) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder.patch_embed(clips)
        tokens = tokens + self.encoder._pos_embed_for_frames(clips.shape[1], device=tokens.device, dtype=tokens.dtype)[:, : tokens.shape[1]]
        visible, mask = self._mask_tokens(tokens, mask_ratio)
        latent = self.encoder.blocks(visible)
        latent = self.encoder.norm(latent)

        decoded = self.decoder(latent)
        if decoded.shape[1] < tokens.shape[1]:
            pad = self.mask_token.expand(decoded.shape[0], tokens.shape[1] - decoded.shape[1], -1)
            decoded = torch.cat([decoded, pad], dim=1)
        recon = decoded[:, : tokens.shape[1]]
        return recon, tokens


@dataclass
class VideoMAEResult:
    train_loss: float
    val_loss: float
    checkpoint_path: str
    final_checkpoint_path: str
    history_path: str
    feature_shape: tuple[int, ...] | None = None


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def pretrain_videomae(
    data_root: str | Path,
    source_split: str = "train",
    subset_size: int = 256,
    image_size: int = 224,
    num_frames: int = 16,
    batch_size: int = 4,
    epochs: int = 1,
    lr: float = 1e-4,
    mask_ratio: float = 0.75,
    seed: int = 0,
    output_dir: str | Path = "logs/videomae",
    cache_dir: str | Path | None = None,
    num_workers: int = 0,
    device: str | None = None,
) -> VideoMAEResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SomethingSomethingVideoDataset(
        data_root=data_root,
        split=source_split,
        image_size=image_size,
        num_frames=num_frames,
        limit=subset_size,
        seed=seed,
        cache_dir=cache_dir,
    )
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 videos to pretrain VideoMAE.")

    split = max(1, int(len(dataset) * 0.9))
    train_ds = torch.utils.data.Subset(dataset, list(range(split)))
    val_ds = torch.utils.data.Subset(dataset, list(range(split, len(dataset))))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoMAEModel(image_size=image_size, num_frames=num_frames).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    best_val = float("inf")
    best_path = output_dir / "best_videomae.pt"
    final_path = output_dir / "videomae_final.pt"
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs} train"):
            clips = batch["clip"].to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, target = model(clips, mask_ratio=mask_ratio)
            loss = _masked_mse(recon, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch+1}/{epochs} val"):
                clips = batch["clip"].to(device)
                recon, target = model(clips, mask_ratio=mask_ratio)
                loss = _masked_mse(recon, target)
                val_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        state = {
            "model_state": model.state_dict(),
            "config": {
                "image_size": image_size,
                "num_frames": num_frames,
                "mask_ratio": mask_ratio,
                "embed_dim": model.embed_dim,
            },
        }
        torch.save(state, final_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, best_path)

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return VideoMAEResult(
        train_loss=train_loss,
        val_loss=val_loss,
        checkpoint_path=str(best_path),
        final_checkpoint_path=str(final_path),
        history_path=str(history_path),
    )


@torch.no_grad()
def extract_videomae_features(
    checkpoint_path: str | Path,
    clips: torch.Tensor,
    device: str | None = None,
) -> torch.Tensor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = VideoMAEModel(
        image_size=int(config.get("image_size", 224)),
        num_frames=int(config.get("num_frames", 16)),
        embed_dim=int(config.get("embed_dim", 192)),
    )
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    clips = clips.to(device)
    tokens = model.encoder.forward_sequence(clips)
    return tokens.detach().cpu()



