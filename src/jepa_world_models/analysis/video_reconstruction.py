"""Masked video reconstruction utilities for the VideoMAE track.

This module builds a small tubelet bank from the dataset and uses the
pretrained VideoMAE encoder/decoder pair to produce masked reconstruction
demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import json
import math
import uuid

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image

from jepa_world_models.analysis.videomae_pipeline import (
    SomethingSomethingVideoDataset,
    VideoMAEModel,
)


MaskMode = Literal["middle", "random"]


@dataclass
class TubeletBank:
    embeddings: torch.Tensor  # (M, D)
    tubelets: torch.Tensor  # (M, tau, 3, P, P) in uint8
    patch_size: int
    tubelet_size: int
    image_size: int
    num_frames: int
    source_split: str
    checkpoint_path: str

    def to_payload(self) -> dict:
        return {
            "embeddings": self.embeddings.cpu(),
            "tubelets": self.tubelets.cpu(),
            "patch_size": self.patch_size,
            "tubelet_size": self.tubelet_size,
            "image_size": self.image_size,
            "num_frames": self.num_frames,
            "source_split": self.source_split,
            "checkpoint_path": self.checkpoint_path,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "TubeletBank":
        return cls(
            embeddings=payload["embeddings"],
            tubelets=payload["tubelets"],
            patch_size=int(payload["patch_size"]),
            tubelet_size=int(payload["tubelet_size"]),
            image_size=int(payload["image_size"]),
            num_frames=int(payload["num_frames"]),
            source_split=str(payload["source_split"]),
            checkpoint_path=str(payload["checkpoint_path"]),
        )


@dataclass
class TubeletDecoderHeadBundle:
    state_dict: dict
    embed_dim: int
    tubelet_dim: int
    hidden_dim: int
    patch_size: int
    tubelet_size: int
    image_size: int
    num_frames: int
    source_split: str
    checkpoint_path: str
    mask_ratio: float
    train_loss: float | None = None
    val_loss: float | None = None

    def to_payload(self) -> dict:
        return {
            "state_dict": self.state_dict,
            "embed_dim": self.embed_dim,
            "tubelet_dim": self.tubelet_dim,
            "hidden_dim": self.hidden_dim,
            "patch_size": self.patch_size,
            "tubelet_size": self.tubelet_size,
            "image_size": self.image_size,
            "num_frames": self.num_frames,
            "source_split": self.source_split,
            "checkpoint_path": self.checkpoint_path,
            "mask_ratio": self.mask_ratio,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "TubeletDecoderHeadBundle":
        return cls(
            state_dict=payload["state_dict"],
            embed_dim=int(payload["embed_dim"]),
            tubelet_dim=int(payload["tubelet_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            patch_size=int(payload["patch_size"]),
            tubelet_size=int(payload["tubelet_size"]),
            image_size=int(payload["image_size"]),
            num_frames=int(payload["num_frames"]),
            source_split=str(payload["source_split"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            mask_ratio=float(payload["mask_ratio"]),
            train_loss=None if payload.get("train_loss") is None else float(payload["train_loss"]),
            val_loss=None if payload.get("val_loss") is None else float(payload["val_loss"]),
        )


class TubeletDecoderHead(nn.Module):
    def __init__(self, embed_dim: int, tubelet_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(embed_dim * 2, tubelet_dim // 2)
        self.embed_dim = embed_dim
        self.tubelet_dim = tubelet_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, tubelet_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _normalize_clip(clip: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=clip.dtype, device=clip.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=clip.dtype, device=clip.device).view(1, 3, 1, 1)
    return (clip - mean) / std


def _denormalize_clip(clip: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=clip.dtype, device=clip.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=clip.dtype, device=clip.device).view(1, 3, 1, 1)
    return clip * std + mean


def _to_uint8(clip: torch.Tensor) -> torch.Tensor:
    clip = _denormalize_clip(clip).clamp(0.0, 1.0)
    return (clip * 255.0).round().to(torch.uint8)


def _from_uint8(clip: torch.Tensor) -> torch.Tensor:
    clip = clip.float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=clip.dtype, device=clip.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=clip.dtype, device=clip.device).view(1, 3, 1, 1)
    return (clip - mean) / std


def _tubelets_to_uint8(tubelets: torch.Tensor) -> torch.Tensor:
    if tubelets.ndim != 5:
        raise ValueError("Expected tubelets shape (N, tau, C, P, P).")
    n, tau, c, p1, p2 = tubelets.shape
    flat = tubelets.reshape(n * tau, c, p1, p2)
    flat_u8 = _to_uint8(flat)
    return flat_u8.reshape(n, tau, c, p1, p2)


def _tubelets_from_uint8(tubelets: torch.Tensor) -> torch.Tensor:
    if tubelets.ndim != 5:
        raise ValueError("Expected tubelets shape (N, tau, C, P, P).")
    n, tau, c, p1, p2 = tubelets.shape
    flat = tubelets.reshape(n * tau, c, p1, p2)
    flat_norm = _from_uint8(flat)
    return flat_norm.reshape(n, tau, c, p1, p2)


def patchify_clip(clip: torch.Tensor, patch_size: int = 16, tubelet_size: int = 2) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Convert a clip (T, C, H, W) into tubelets (N, tau, C, P, P)."""
    if clip.ndim != 4:
        raise ValueError(f"Expected clip shape (T, C, H, W), got {tuple(clip.shape)}")
    t, c, h, w = clip.shape
    t_blocks = t // tubelet_size
    h_blocks = h // patch_size
    w_blocks = w // patch_size
    if t_blocks == 0 or h_blocks == 0 or w_blocks == 0:
        raise ValueError("Clip too small for the requested tubelet and patch sizes.")
    clip = clip[: t_blocks * tubelet_size, :, : h_blocks * patch_size, : w_blocks * patch_size]
    x = clip.view(t_blocks, tubelet_size, c, h_blocks, patch_size, w_blocks, patch_size)
    x = x.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
    tubelets = x.view(t_blocks * h_blocks * w_blocks, tubelet_size, c, patch_size, patch_size)
    return tubelets, (t_blocks, h_blocks, w_blocks)


def batch_patchify_clips(
    clips: torch.Tensor,
    patch_size: int = 16,
    tubelet_size: int = 2,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Patchify a batch of clips into tubelets.

    Returns:
        tubelets: (B, N, tau, C, P, P)
    """
    if clips.ndim != 5:
        raise ValueError(f"Expected clips shape (B, T, C, H, W), got {tuple(clips.shape)}")
    tubelet_batches = []
    shape: tuple[int, int, int] | None = None
    for clip in clips:
        tubelets, shape = patchify_clip(clip, patch_size=patch_size, tubelet_size=tubelet_size)
        tubelet_batches.append(tubelets)
    assert shape is not None
    return torch.stack(tubelet_batches, dim=0), shape


def _tubelets_to_flat(tubelets: torch.Tensor) -> torch.Tensor:
    if tubelets.ndim != 5:
        raise ValueError("Expected tubelets shape (N, tau, C, P, P).")
    return tubelets.reshape(tubelets.shape[0], -1)


def _flat_to_tubelets(
    flat: torch.Tensor,
    tubelet_size: int,
    patch_size: int,
) -> torch.Tensor:
    if flat.ndim != 2:
        raise ValueError("Expected flat tubelet tensor shape (N, D).")
    expected = tubelet_size * 3 * patch_size * patch_size
    if flat.shape[1] != expected:
        raise ValueError(f"Expected feature dimension {expected}, got {flat.shape[1]}.")
    return flat.view(flat.shape[0], tubelet_size, 3, patch_size, patch_size)


def unpatchify_tubelets(
    tubelets: torch.Tensor,
    t_blocks: int,
    h_blocks: int,
    w_blocks: int,
    patch_size: int = 16,
    tubelet_size: int = 2,
) -> torch.Tensor:
    """Convert tubelets back into a clip (T, C, H, W)."""
    if tubelets.ndim != 5:
        raise ValueError(f"Expected tubelets shape (N, tau, C, P, P), got {tuple(tubelets.shape)}")
    n, tau, c, p1, p2 = tubelets.shape
    if tau != tubelet_size or p1 != patch_size or p2 != patch_size:
        raise ValueError("Tubelet dimensions do not match the requested geometry.")
    expected = t_blocks * h_blocks * w_blocks
    if n != expected:
        raise ValueError(f"Expected {expected} tubelets, got {n}")
    x = tubelets.view(t_blocks, h_blocks, w_blocks, tubelet_size, c, patch_size, patch_size)
    x = x.permute(0, 3, 4, 1, 5, 2, 6).contiguous()
    clip = x.view(t_blocks * tubelet_size, c, h_blocks * patch_size, w_blocks * patch_size)
    return clip


def _mask_tubelets(num_tokens: int, t_blocks: int, h_blocks: int, w_blocks: int, mask_ratio: float, mode: MaskMode, seed: int = 0) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    if mode == "middle":
        masked_t_blocks = max(1, int(round(t_blocks * mask_ratio)))
        start = max(0, (t_blocks - masked_t_blocks) // 2)
        stop = min(t_blocks, start + masked_t_blocks)
        for tb in range(start, stop):
            begin = tb * h_blocks * w_blocks
            end = begin + h_blocks * w_blocks
            mask[begin:end] = True
        return mask
    num_mask = max(1, int(round(num_tokens * mask_ratio)))
    perm = torch.randperm(num_tokens, generator=rng)
    mask[perm[:num_mask]] = True
    return mask


def _sample_mask_indices(num_tokens: int, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ids_keep = torch.nonzero(~mask, as_tuple=False).flatten()
    ids_mask = torch.nonzero(mask, as_tuple=False).flatten()
    ids_shuffle = torch.cat([ids_keep, ids_mask], dim=0)
    ids_restore = torch.argsort(ids_shuffle)
    return ids_keep, ids_restore


def _load_model(checkpoint_path: str | Path, device: str | None = None) -> VideoMAEModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config", {})
    model = VideoMAEModel(
        image_size=int(cfg.get("image_size", 224)),
        num_frames=int(cfg.get("num_frames", 16)),
        embed_dim=int(cfg.get("embed_dim", 192)),
    )
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


@torch.no_grad()
def encode_tokens(model: VideoMAEModel, clips: torch.Tensor) -> torch.Tensor:
    tokens = model.encoder.patch_embed(clips)
    tokens = tokens + model.encoder.pos_embed[:, : tokens.shape[1]]
    tokens = model.encoder.blocks(tokens)
    tokens = model.encoder.norm(tokens)
    return tokens


@torch.no_grad()
def reconstruct_tokens(
    model: VideoMAEModel,
    clips: torch.Tensor,
    mask_ratio: float = 0.5,
    mask_mode: MaskMode = "middle",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reconstructed token embeddings and mask."""
    tokens = model.encoder.patch_embed(clips)
    tokens = tokens + model.encoder.pos_embed[:, : tokens.shape[1]]
    batch, num_tokens, dim = tokens.shape
    t_blocks = clips.shape[1] // model.encoder.patch_embed.tubelet_size
    h_blocks = clips.shape[3] // model.encoder.patch_embed.patch_size
    w_blocks = clips.shape[4] // model.encoder.patch_embed.patch_size
    mask = _mask_tubelets(num_tokens, t_blocks, h_blocks, w_blocks, mask_ratio, mask_mode, seed=seed).to(clips.device)
    ids_keep, ids_restore = _sample_mask_indices(num_tokens, mask)
    visible = tokens[:, ids_keep]
    latent = model.encoder.blocks(visible)
    latent = model.encoder.norm(latent)
    x = model.decoder.proj_in(latent)
    mask_tokens = model.decoder.proj_in(model.mask_token).expand(batch, num_tokens - visible.shape[1], -1)
    x = torch.cat([x, mask_tokens], dim=1)
    x = torch.gather(x, 1, ids_restore.view(1, -1, 1).expand(batch, -1, x.shape[-1]))
    x = model.decoder.blocks(x)
    x = model.decoder.norm(x)
    recon = model.decoder.head(x)
    return recon, mask


def clip_to_frames_uint8(clip: torch.Tensor) -> list[np.ndarray]:
    if clip.ndim != 4:
        raise ValueError("Expected clip tensor with shape (T, C, H, W).")
    clip = clip.detach().cpu()
    if clip.dtype != torch.uint8:
        clip = _to_uint8(clip.float())
    frames = []
    for frame in clip:
        frames.append(frame.permute(1, 2, 0).numpy())
    return frames


def frames_to_mp4(frames: list[np.ndarray], path: str | Path, fps: int = 8) -> None:
    if not frames:
        raise ValueError("No frames to write.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def frames_to_gif(frames: list[np.ndarray], path: str | Path, fps: int = 8) -> None:
    if not frames:
        raise ValueError("No frames to write.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = max(1, int(round(1000.0 / max(fps, 1))))
    images[0].save(
        str(path),
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def contact_sheet(frames: list[np.ndarray], max_frames: int = 16) -> np.ndarray:
    if not frames:
        raise ValueError("No frames to compose.")
    frames = frames[:max_frames]
    return np.concatenate(frames, axis=1)


def mask_to_sheet(
    mask: torch.Tensor,
    t_blocks: int,
    h_blocks: int,
    w_blocks: int,
    patch_size: int,
    tubelet_size: int,
) -> np.ndarray:
    if mask.ndim != 1:
        raise ValueError("Expected mask shape (N,).")
    expected = t_blocks * h_blocks * w_blocks
    if mask.numel() != expected:
        raise ValueError(f"Expected {expected} mask entries, got {mask.numel()}.")
    _ = tubelet_size
    mask_3d = mask.view(t_blocks, h_blocks, w_blocks).float().numpy()
    frames: list[np.ndarray] = []
    for tb in range(t_blocks):
        frame = (mask_3d[tb] * 255.0).astype(np.uint8)
        frame = np.repeat(np.repeat(frame, patch_size, axis=0), patch_size, axis=1)
        frame = np.stack([frame, frame, frame], axis=-1)
        frames.append(frame)
    return contact_sheet(frames, max_frames=t_blocks)


def load_clip_from_video_path(
    video_path: str | Path,
    num_frames: int = 16,
    image_size: int = 224,
) -> torch.Tensor:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
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
    if not frames:
        raise RuntimeError(f"Could not decode video: {video_path}")
    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int).tolist()
    selected = []
    for idx in indices:
        frame = cv2.resize(frames[idx], (image_size, image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(frame).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor = _normalize_clip(tensor.unsqueeze(0)).squeeze(0)
        selected.append(tensor)
    return torch.stack(selected, dim=0)


def build_tubelet_bank(
    checkpoint_path: str | Path,
    data_root: str | Path,
    source_split: str = "train",
    subset_size: int = 128,
    image_size: int = 224,
    num_frames: int = 16,
    batch_size: int = 4,
    cache_dir: str | Path | None = None,
    device: str | None = None,
) -> TubeletBank:
    cache_path = None
    if cache_dir is not None:
        checkpoint_stem = Path(checkpoint_path).stem
        cache_name = f"tubelet_bank_{checkpoint_stem}_{source_split}_{subset_size}_{image_size}_{num_frames}.pt"
        cache_path = Path(cache_dir) / cache_name
        if cache_path.exists():
            return TubeletBank.from_payload(torch.load(cache_path, map_location="cpu"))

    model = _load_model(checkpoint_path, device=device)
    dataset = SomethingSomethingVideoDataset(
        data_root=data_root,
        split=source_split,
        image_size=image_size,
        num_frames=num_frames,
        limit=subset_size,
        seed=0,
        cache_dir=cache_dir,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embedding_batches = []
    tubelet_batches = []
    for batch in tqdm(loader, desc="Building tubelet bank", unit="batch"):
        clips = batch["clip"].to(next(model.parameters()).device)
        tokens = encode_tokens(model, clips)
        for i in range(clips.shape[0]):
            tubelets, _ = patchify_clip(batch["clip"][i], patch_size=model.encoder.patch_embed.patch_size, tubelet_size=model.encoder.patch_embed.tubelet_size)
            embedding_batches.append(tokens[i].detach().cpu())
            tubelet_batches.append(_tubelets_to_uint8(tubelets.detach().cpu()))
    embeddings = torch.cat(embedding_batches, dim=0)
    tubelets = torch.cat(tubelet_batches, dim=0)
    bank = TubeletBank(
        embeddings=embeddings.float(),
        tubelets=tubelets.to(torch.uint8),
        patch_size=model.encoder.patch_embed.patch_size,
        tubelet_size=model.encoder.patch_embed.tubelet_size,
        image_size=image_size,
        num_frames=num_frames,
        source_split=source_split,
        checkpoint_path=str(checkpoint_path),
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bank.to_payload(), cache_path)
    return bank


def _fit_head_epoch(
    model: VideoMAEModel,
    head: TubeletDecoderHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    mask_ratio: float,
    mask_mode: MaskMode,
    patch_size: int,
    tubelet_size: int,
    train: bool,
    seed_offset: int,
) -> float:
    losses: list[float] = []
    head.train(mode=train)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(tqdm(loader, desc="decoder head train" if train else "decoder head val", leave=False)):
            clips = batch["clip"].to(device)
            for sample_idx in range(clips.shape[0]):
                clip = clips[sample_idx : sample_idx + 1]
                recon_tokens, mask = reconstruct_tokens(
                    model,
                    clip,
                    mask_ratio=mask_ratio,
                    mask_mode=mask_mode,
                    seed=seed_offset + batch_idx * 1000 + sample_idx,
                )
                tubelets, _ = patchify_clip(
                    clip[0].detach().cpu(),
                    patch_size=patch_size,
                    tubelet_size=tubelet_size,
                )
                target = tubelets.reshape(tubelets.shape[0], -1).to(device)
                features = recon_tokens[0]
                features = features[mask]
                target = target[mask]
                if features.numel() == 0:
                    continue
                if train:
                    optimizer.zero_grad(set_to_none=True)
                pred = head(features)
                loss = F.mse_loss(pred, target)
                if train:
                    loss.backward()
                    optimizer.step()
                losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def build_reconstruction_head(
    checkpoint_path: str | Path,
    data_root: str | Path,
    source_split: str = "train",
    subset_size: int = 128,
    image_size: int = 224,
    num_frames: int = 16,
    batch_size: int = 2,
    epochs: int = 3,
    lr: float = 1e-3,
    mask_ratio: float = 0.5,
    mask_mode: MaskMode = "random",
    seed: int = 0,
    cache_dir: str | Path | None = None,
    device: str | None = None,
) -> tuple[TubeletDecoderHead, TubeletDecoderHeadBundle]:
    cache_path = None
    if cache_dir is not None:
        checkpoint_stem = Path(checkpoint_path).stem
        cache_name = (
            f"reconstruction_head_{checkpoint_stem}_{source_split}_{subset_size}_{image_size}_{num_frames}"
            f"_{int(mask_ratio * 100)}_{mask_mode}.pt"
        )
        cache_path = Path(cache_dir) / cache_name
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            bundle = TubeletDecoderHeadBundle.from_payload(payload)
            head = TubeletDecoderHead(
                embed_dim=bundle.embed_dim,
                tubelet_dim=bundle.tubelet_dim,
                hidden_dim=bundle.hidden_dim,
            )
            head.load_state_dict(bundle.state_dict)
            return head.eval(), bundle

    model = _load_model(checkpoint_path, device=device)
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
        raise RuntimeError("Need at least 2 videos to fit the reconstruction head.")

    split = max(1, int(len(dataset) * 0.9))
    train_ds = torch.utils.data.Subset(dataset, list(range(split)))
    val_ds = torch.utils.data.Subset(dataset, list(range(split, len(dataset))))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    patch_size = model.encoder.patch_embed.patch_size
    tubelet_size = model.encoder.patch_embed.tubelet_size
    tubelet_dim = tubelet_size * 3 * patch_size * patch_size
    head = TubeletDecoderHead(embed_dim=model.embed_dim, tubelet_dim=tubelet_dim).to(next(model.parameters()).device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    train_loss = float("nan")
    val_loss = float("nan")
    for epoch in range(epochs):
        train_loss = _fit_head_epoch(
            model,
            head,
            train_loader,
            optimizer,
            device=next(model.parameters()).device,
            mask_ratio=mask_ratio,
            mask_mode=mask_mode,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            train=True,
            seed_offset=seed + epoch * 100,
        )
        val_loss = _fit_head_epoch(
            model,
            head,
            val_loader,
            optimizer,
            device=next(model.parameters()).device,
            mask_ratio=mask_ratio,
            mask_mode=mask_mode,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            train=False,
            seed_offset=seed + 10_000 + epoch * 100,
        )

    bundle = TubeletDecoderHeadBundle(
        state_dict=head.state_dict(),
        embed_dim=model.embed_dim,
        tubelet_dim=tubelet_dim,
        hidden_dim=head.hidden_dim,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        image_size=image_size,
        num_frames=num_frames,
        source_split=source_split,
        checkpoint_path=str(checkpoint_path),
        mask_ratio=mask_ratio,
        train_loss=train_loss,
        val_loss=val_loss,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle.to_payload(), cache_path)
    return head.eval(), bundle


@torch.no_grad()
def reconstruct_clip_with_decoder(
    checkpoint_path: str | Path,
    head: TubeletDecoderHead,
    clip: torch.Tensor,
    mask_ratio: float = 0.5,
    mask_mode: MaskMode = "middle",
    seed: int = 0,
    device: str | None = None,
) -> dict:
    model = _load_model(checkpoint_path, device=device)
    clip = clip.detach().cpu()
    raw_tubelets, (t_blocks, h_blocks, w_blocks) = patchify_clip(
        clip,
        patch_size=model.encoder.patch_embed.patch_size,
        tubelet_size=model.encoder.patch_embed.tubelet_size,
    )
    clips = clip.unsqueeze(0).to(next(model.parameters()).device)
    recon_tokens, mask = reconstruct_tokens(
        model,
        clips,
        mask_ratio=mask_ratio,
        mask_mode=mask_mode,
        seed=seed,
    )
    recon_tokens = recon_tokens[0].detach().cpu()
    head_device = next(head.parameters()).device
    pred_flat = head(recon_tokens.to(head_device)).detach().cpu()
    pred_tubelets = _flat_to_tubelets(
        pred_flat,
        tubelet_size=model.encoder.patch_embed.tubelet_size,
        patch_size=model.encoder.patch_embed.patch_size,
    )
    reconstructed = raw_tubelets.clone()
    reconstructed[mask.cpu()] = pred_tubelets[mask.cpu()]
    masked = raw_tubelets.clone()
    masked[mask.cpu()] = torch.zeros_like(masked[mask.cpu()])
    mse = 0.0
    if mask.any():
        mse = float(F.mse_loss(pred_flat[mask.cpu()], _tubelets_to_flat(raw_tubelets)[mask.cpu()]).item())
    return {
        "mask": mask.cpu(),
        "original_clip": clip,
        "masked_clip": unpatchify_tubelets(
            masked,
            t_blocks,
            h_blocks,
            w_blocks,
            patch_size=model.encoder.patch_embed.patch_size,
            tubelet_size=model.encoder.patch_embed.tubelet_size,
        ),
        "reconstructed_clip": unpatchify_tubelets(
            reconstructed,
            t_blocks,
            h_blocks,
            w_blocks,
            patch_size=model.encoder.patch_embed.patch_size,
            tubelet_size=model.encoder.patch_embed.tubelet_size,
        ),
        "t_blocks": t_blocks,
        "h_blocks": h_blocks,
        "w_blocks": w_blocks,
        "patch_size": model.encoder.patch_embed.patch_size,
        "tubelet_size": model.encoder.patch_embed.tubelet_size,
        "reconstruction_loss": mse,
        "mode": "decoder",
    }


@torch.no_grad()
def reconstruct_clip_with_decoder(
    checkpoint_path: str | Path,
    head: TubeletDecoderHead,
    clip: torch.Tensor,
    mask_ratio: float = 0.5,
    mask_mode: MaskMode = "middle",
    seed: int = 0,
    device: str | None = None,
) -> dict:
    model = _load_model(checkpoint_path, device=device)
    clip = clip.detach().cpu()
    raw_tubelets, (t_blocks, h_blocks, w_blocks) = patchify_clip(
        clip,
        patch_size=model.encoder.patch_embed.patch_size,
        tubelet_size=model.encoder.patch_embed.tubelet_size,
    )
    clips = clip.unsqueeze(0).to(next(model.parameters()).device)
    recon_tokens, mask = reconstruct_tokens(
        model,
        clips,
        mask_ratio=mask_ratio,
        mask_mode=mask_mode,
        seed=seed,
    )
    recon_tokens = recon_tokens[0].detach().cpu()
    pred_flat = head(recon_tokens.to(next(head.parameters()).device)).detach().cpu()
    pred_tubelets = _flat_to_tubelets(
        pred_flat,
        tubelet_size=model.encoder.patch_embed.tubelet_size,
        patch_size=model.encoder.patch_embed.patch_size,
    )
    reconstructed = raw_tubelets.clone()
    reconstructed[mask.cpu()] = pred_tubelets[mask.cpu()]
    masked = raw_tubelets.clone()
    masked[mask.cpu()] = torch.zeros_like(masked[mask.cpu()])
    mse = float(F.mse_loss(pred_flat[mask.cpu()], _tubelets_to_flat(raw_tubelets)[mask.cpu()]).item()) if mask.any() else 0.0
    return {
        "mask": mask.cpu(),
        "original_clip": clip,
        "masked_clip": unpatchify_tubelets(
            masked,
            t_blocks,
            h_blocks,
            w_blocks,
            patch_size=model.encoder.patch_embed.patch_size,
            tubelet_size=model.encoder.patch_embed.tubelet_size,
        ),
        "reconstructed_clip": unpatchify_tubelets(
            reconstructed,
            t_blocks,
            h_blocks,
            w_blocks,
            patch_size=model.encoder.patch_embed.patch_size,
            tubelet_size=model.encoder.patch_embed.tubelet_size,
        ),
        "t_blocks": t_blocks,
        "h_blocks": h_blocks,
        "w_blocks": w_blocks,
        "reconstruction_loss": mse,
        "mode": "decoder",
    }


def reconstruct_clip_with_bank(
    checkpoint_path: str | Path,
    bank: TubeletBank,
    clip: torch.Tensor,
    mask_ratio: float = 0.5,
    mask_mode: MaskMode = "middle",
    seed: int = 0,
    device: str | None = None,
) -> dict:
    model = _load_model(checkpoint_path, device=device)
    clip = clip.detach().cpu()
    raw_tubelets, (t_blocks, h_blocks, w_blocks) = patchify_clip(clip, patch_size=bank.patch_size, tubelet_size=bank.tubelet_size)
    raw_tubelets_u8 = _tubelets_to_uint8(raw_tubelets)
    clips = clip.unsqueeze(0).to(next(model.parameters()).device)
    recon_tokens, mask = reconstruct_tokens(model, clips, mask_ratio=mask_ratio, mask_mode=mask_mode, seed=seed)
    recon_tokens = recon_tokens[0].detach().cpu()
    bank_embeddings = F.normalize(bank.embeddings.float(), dim=-1)
    recon_embeddings = F.normalize(recon_tokens.float(), dim=-1)
    best_scores = torch.full((recon_embeddings.shape[0],), float("-inf"))
    best = torch.zeros(recon_embeddings.shape[0], dtype=torch.long)
    chunk_size = 8192
    for start in range(0, bank_embeddings.shape[0], chunk_size):
        end = min(start + chunk_size, bank_embeddings.shape[0])
        sim = recon_embeddings @ bank_embeddings[start:end].T
        scores, idx = sim.max(dim=1)
        update = scores > best_scores
        best_scores[update] = scores[update]
        best[update] = idx[update] + start
    selected = bank.tubelets[best].clone()
    selected[~mask.cpu()] = raw_tubelets_u8[~mask.cpu()].to(selected.dtype)
    masked = raw_tubelets_u8.clone()
    masked[mask.cpu()] = torch.zeros_like(masked[mask.cpu()])
    reconstructed = unpatchify_tubelets(selected, t_blocks, h_blocks, w_blocks, patch_size=bank.patch_size, tubelet_size=bank.tubelet_size)
    masked_clip = unpatchify_tubelets(masked, t_blocks, h_blocks, w_blocks, patch_size=bank.patch_size, tubelet_size=bank.tubelet_size)
    return {
        "mask": mask.cpu(),
        "original_clip": clip,
        "masked_clip": masked_clip,
        "reconstructed_clip": reconstructed,
        "t_blocks": t_blocks,
        "h_blocks": h_blocks,
        "w_blocks": w_blocks,
        "patch_size": bank.patch_size,
        "tubelet_size": bank.tubelet_size,
        "mode": "retrieval",
    }


def save_reconstruction_artifacts(result: dict, output_dir: str | Path, fps: int = 8) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original = clip_to_frames_uint8(result["original_clip"])
    masked = clip_to_frames_uint8(result["masked_clip"])
    recon = clip_to_frames_uint8(result["reconstructed_clip"])
    original_video = output_dir / "original.mp4"
    masked_video = output_dir / "masked.mp4"
    recon_video = output_dir / "reconstructed.mp4"
    original_gif = output_dir / "original.gif"
    masked_gif = output_dir / "masked.gif"
    recon_gif = output_dir / "reconstructed.gif"
    frames_video = output_dir / "original_sheet.png"
    frames_video2 = output_dir / "masked_sheet.png"
    frames_video3 = output_dir / "reconstructed_sheet.png"
    mask_sheet = output_dir / "mask_sheet.png"
    frames_to_mp4(original, original_video, fps=fps)
    frames_to_mp4(masked, masked_video, fps=fps)
    frames_to_mp4(recon, recon_video, fps=fps)
    frames_to_gif(original, original_gif, fps=fps)
    frames_to_gif(masked, masked_gif, fps=fps)
    frames_to_gif(recon, recon_gif, fps=fps)
    cv2.imwrite(str(frames_video), cv2.cvtColor(contact_sheet(original), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(frames_video2), cv2.cvtColor(contact_sheet(masked), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(frames_video3), cv2.cvtColor(contact_sheet(recon), cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(mask_sheet),
        cv2.cvtColor(
            mask_to_sheet(
                result["mask"],
                result["t_blocks"],
                result["h_blocks"],
                result["w_blocks"],
                patch_size=int(result.get("patch_size", 16)),
                tubelet_size=int(result.get("tubelet_size", 2)),
            ),
            cv2.COLOR_RGB2BGR,
        ),
    )
    payload = {
        "original_video": str(original_video),
        "masked_video": str(masked_video),
        "reconstructed_video": str(recon_video),
        "original_gif": str(original_gif),
        "masked_gif": str(masked_gif),
        "reconstructed_gif": str(recon_gif),
        "original_sheet": str(frames_video),
        "masked_sheet": str(frames_video2),
        "reconstructed_sheet": str(frames_video3),
        "mask_sheet": str(mask_sheet),
    }
    if "reconstruction_loss" in result:
        payload["reconstruction_loss"] = float(result["reconstruction_loss"])
    if "mode" in result:
        payload["mode"] = str(result["mode"])
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
