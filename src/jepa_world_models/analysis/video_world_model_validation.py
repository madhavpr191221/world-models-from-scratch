from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from jepa_world_models.analysis.common import resolve_device
from jepa_world_models.analysis.video_world_model import (
    LatentSequenceBank,
    LatentWindowDataset,
    LatentWorldModelBundle,
    build_latent_sequence_bank,
    _split_indices,
)
from jepa_world_models.analysis.video_latent_models import PredictorSpec, build_temporal_predictor


@dataclass(slots=True)
class RolloutValidationResult:
    checkpoint_path: str
    cache_path: str
    source_split: str
    predictor_mode: str
    subset_size: int
    image_size: int
    context_frames: int
    future_frames: int
    total_frames: int
    model_context_steps: int
    latent_dim: int
    num_samples: int
    device: str
    horizon_metrics: list[dict[str, float]]
    alignment_cosine_samples: list[list[float]]
    alignment_statistics: list[dict[str, float]]
    singular_spectrum: list[dict[str, float]]
    gradient_norms: list[dict[str, float]]

    def to_json(self) -> dict:
        return {
            "checkpoint_path": self.checkpoint_path,
            "cache_path": self.cache_path,
            "source_split": self.source_split,
            "predictor_mode": self.predictor_mode,
            "subset_size": self.subset_size,
            "image_size": self.image_size,
            "context_frames": self.context_frames,
            "future_frames": self.future_frames,
            "total_frames": self.total_frames,
            "model_context_steps": self.model_context_steps,
            "latent_dim": self.latent_dim,
            "num_samples": self.num_samples,
            "device": self.device,
            "horizon_metrics": self.horizon_metrics,
            "alignment_cosine_samples": self.alignment_cosine_samples,
            "alignment_statistics": self.alignment_statistics,
            "singular_spectrum": self.singular_spectrum,
            "gradient_norms": self.gradient_norms,
        }


def _load_predictor(checkpoint_path: str | Path, device: str | torch.device | None = None) -> tuple[LatentWorldModelBundle, nn.Module]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle = LatentWorldModelBundle.from_payload(payload)
    model = build_temporal_predictor(
        PredictorSpec(
            name=bundle.predictor_name or "causal_transformer",
            latent_dim=bundle.latent_dim,
            context_steps=bundle.context_steps,
            future_steps=bundle.future_steps,
            hidden_dim=bundle.hidden_dim,
            num_layers=bundle.num_layers,
            num_heads=bundle.num_heads,
            dropout=bundle.dropout,
        )
    )
    model.load_state_dict(bundle.state_dict, strict=True)
    device_obj = resolve_device(device)
    model = model.to(device_obj)
    model.eval()
    return bundle, model


def _make_test_dataset(bank: LatentSequenceBank, seed: int) -> LatentWindowDataset:
    train_indices, val_indices, test_indices = _split_indices(len(bank.sample_indices), seed)
    return LatentWindowDataset(
        bank.context_latents[test_indices],
        bank.future_latents[test_indices],
        [bank.sample_indices[i] for i in test_indices],
        [bank.video_ids[i] for i in test_indices],
    )


def _one_step_prediction(model: TemporalLatentPredictor, context: torch.Tensor) -> torch.Tensor:
    return model(context)[:, 0, :]


def _summary_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "p10": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=values.device))
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "min": float(values.min().item()),
        "p10": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
        "max": float(values.max().item()),
    }


@torch.inference_mode()
def _rollout_batch_metrics(
    model: TemporalLatentPredictor,
    context: torch.Tensor,
    future: torch.Tensor,
) -> tuple[list[dict[str, float]], dict[str, list[torch.Tensor]]]:
    full = torch.cat([context, future], dim=1)
    generated = context.clone()
    batch_size, context_steps, latent_dim = context.shape
    future_steps = future.shape[1]
    horizon_metrics: list[dict[str, float]] = []
    raw: dict[str, list[torch.Tensor]] = {
        "alignment_cosine": [],
        "teacher_forced_error": [],
        "rollout_error": [],
        "drift": [],
    }

    for horizon_idx in range(future_steps):
        tf_context = full[:, horizon_idx : horizon_idx + context_steps, :]
        rollout_context = generated[:, -context_steps:, :]
        target = full[:, context_steps + horizon_idx, :]

        tf_pred = _one_step_prediction(model, tf_context)
        rollout_pred = _one_step_prediction(model, rollout_context)
        drift = tf_pred - rollout_pred
        epsilon_tf = target - tf_pred
        epsilon_ro = target - rollout_pred

        tf_norm = torch.linalg.norm(epsilon_tf, dim=-1)
        ro_norm = torch.linalg.norm(epsilon_ro, dim=-1)
        drift_norm = torch.linalg.norm(drift, dim=-1)
        alignment = F.cosine_similarity(epsilon_tf, drift, dim=-1, eps=1e-8)
        denom = torch.linalg.norm(tf_context.reshape(batch_size, -1) - rollout_context.reshape(batch_size, -1), dim=-1)
        local_ratio = torch.linalg.norm(tf_pred - rollout_pred, dim=-1) / (denom + 1e-8)
        norm_identity_error = (
            ro_norm.square()
            - tf_norm.square()
            - drift_norm.square()
            - 2.0 * torch.sum(epsilon_tf * drift, dim=-1)
        ).abs()
        decomposition_error = (epsilon_ro - epsilon_tf - drift).abs().amax(dim=-1)

        horizon_metrics.append(
            {
                "horizon": float(horizon_idx + 1),
                "teacher_forced_mse": float(F.mse_loss(tf_pred, target).item()),
                "rollout_mse": float(F.mse_loss(rollout_pred, target).item()),
                "teacher_forced_error_norm": float(tf_norm.mean().item()),
                "rollout_error_norm": float(ro_norm.mean().item()),
                "drift_norm": float(drift_norm.mean().item()),
                "alignment_cosine": float(alignment.mean().item()),
                "alignment_cosine_min": float(alignment.min().item()),
                "alignment_cosine_max": float(alignment.max().item()),
                "decomposition_max_abs_error": float(decomposition_error.max().item()),
                "norm_identity_max_abs_error": float(norm_identity_error.max().item()),
                "local_lipschitz_ratio_mean": float(local_ratio.mean().item()),
                "local_lipschitz_ratio_max": float(local_ratio.max().item()),
            }
        )
        raw["alignment_cosine"].append(alignment.detach().cpu())
        raw["teacher_forced_error"].append(epsilon_tf.detach().cpu())
        raw["rollout_error"].append(epsilon_ro.detach().cpu())
        raw["drift"].append(drift.detach().cpu())

        generated = torch.cat([generated, rollout_pred.unsqueeze(1)], dim=1)

    return horizon_metrics, raw


def _gradient_norms_for_batch(model: TemporalLatentPredictor, context: torch.Tensor, future: torch.Tensor) -> list[dict[str, float]]:
    full = torch.cat([context, future], dim=1)
    batch_size, context_steps, _ = context.shape
    future_steps = future.shape[1]
    gradient_norms: list[dict[str, float]] = []

    for horizon_idx in range(future_steps):
        model.zero_grad(set_to_none=True)
        tf_context = full[:, horizon_idx : horizon_idx + context_steps, :]
        target = full[:, context_steps + horizon_idx, :]
        pred = _one_step_prediction(model, tf_context)
        loss = F.mse_loss(pred, target)
        loss.backward()
        total_sq = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            total_sq += float(param.grad.detach().pow(2).sum().item())
        gradient_norms.append(
            {
                "horizon": float(horizon_idx + 1),
                "loss": float(loss.item()),
                "gradient_norm": float(math.sqrt(total_sq)),
            }
        )
    model.zero_grad(set_to_none=True)
    return gradient_norms


def build_rollout_validation_report(
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
    feature_batch_size: int = 1,
    seed: int = 0,
    batch_limit: int | None = 1,
    cache_dir: str | Path | None = "logs/video_world_model/cache",
    device: str | torch.device | None = None,
) -> RolloutValidationResult:
    bundle, model = _load_predictor(checkpoint_path, device=device)
    bank = build_latent_sequence_bank(
        checkpoint_path=bundle.checkpoint_path,
        data_root=data_root,
        source_split=source_split,
        subset_size=subset_size,
        image_size=image_size,
        total_frames=total_frames,
        context_frames=context_frames,
        future_frames=future_frames,
        sample_fps=sample_fps,
        batch_size=feature_batch_size,
        cache_dir=cache_dir,
        seed=seed,
        device=device,
    )
    train_indices, val_indices, test_indices = _split_indices(len(bank.sample_indices), seed)
    context_latents = bank.context_latents
    if bundle.predictor_mode.strip().lower() == "one-lag":
        context_latents = context_latents[:, -1:, :].contiguous()
    dataset = LatentWindowDataset(
        context_latents=context_latents[test_indices],
        future_latents=bank.future_latents[test_indices],
        sample_indices=[bank.sample_indices[i] for i in test_indices],
        video_ids=[bank.video_ids[i] for i in test_indices],
    )
    pin_memory = torch.cuda.is_available()
    loader = DataLoader(dataset, batch_size=max(1, feature_batch_size), shuffle=False, num_workers=0, pin_memory=pin_memory)

    horizon_metrics_accum: list[list[dict[str, float]]] = []
    alignment_values_accum: list[list[torch.Tensor]] | None = None
    tf_error_accum: list[list[torch.Tensor]] | None = None
    ro_error_accum: list[list[torch.Tensor]] | None = None
    drift_accum: list[list[torch.Tensor]] | None = None
    gradient_norms: list[dict[str, float]] = []
    device_obj = next(model.parameters()).device

    for batch_index, batch in enumerate(loader):
        if batch_limit is not None and batch_index >= batch_limit:
            break
        context = batch["context"].to(device_obj)
        future = batch["future"].to(device_obj)
        batch_metrics, raw = _rollout_batch_metrics(model, context, future)
        horizon_metrics_accum.append(batch_metrics)
        if alignment_values_accum is None:
            horizon_count = len(batch_metrics)
            alignment_values_accum = [[] for _ in range(horizon_count)]
            tf_error_accum = [[] for _ in range(horizon_count)]
            ro_error_accum = [[] for _ in range(horizon_count)]
            drift_accum = [[] for _ in range(horizon_count)]
        for horizon_idx in range(len(batch_metrics)):
            alignment_values_accum[horizon_idx].append(raw["alignment_cosine"][horizon_idx])
            tf_error_accum[horizon_idx].append(raw["teacher_forced_error"][horizon_idx])
            ro_error_accum[horizon_idx].append(raw["rollout_error"][horizon_idx])
            drift_accum[horizon_idx].append(raw["drift"][horizon_idx])
        if batch_index == 0:
            gradient_norms = _gradient_norms_for_batch(model, context, future)

    if not horizon_metrics_accum:
        raise RuntimeError("Validation dataset produced no batches.")

    horizon_metrics: list[dict[str, float]] = []
    horizon_count = len(horizon_metrics_accum[0])
    for horizon_idx in range(horizon_count):
        collected = [batch_metrics[horizon_idx] for batch_metrics in horizon_metrics_accum]
        combined: dict[str, float] = {"horizon": float(horizon_idx + 1)}
        keys = [key for key in collected[0].keys() if key != "horizon"]
        for key in keys:
            values = torch.tensor([entry[key] for entry in collected], dtype=torch.float32)
            combined[key] = float(values.mean().item())
            combined[f"{key}_max"] = float(values.max().item())
        assert alignment_values_accum is not None
        assert tf_error_accum is not None
        assert ro_error_accum is not None
        assert drift_accum is not None
        alignment_tensor = torch.cat(alignment_values_accum[horizon_idx], dim=0)
        combined["alignment_cosine_std"] = float(alignment_tensor.std(unbiased=False).item()) if alignment_tensor.numel() > 1 else 0.0
        combined["alignment_cosine_p10"] = float(torch.quantile(alignment_tensor, torch.tensor(0.1, device=alignment_tensor.device)).item())
        combined["alignment_cosine_median"] = float(torch.quantile(alignment_tensor, torch.tensor(0.5, device=alignment_tensor.device)).item())
        combined["alignment_cosine_p90"] = float(torch.quantile(alignment_tensor, torch.tensor(0.9, device=alignment_tensor.device)).item())
        horizon_metrics.append(combined)

    singular_spectrum: list[dict[str, float]] = []
    assert tf_error_accum is not None
    assert ro_error_accum is not None
    for horizon_idx in range(horizon_count):
        tf_matrix = torch.cat(tf_error_accum[horizon_idx], dim=0).float()
        ro_matrix = torch.cat(ro_error_accum[horizon_idx], dim=0).float()
        drift_matrix = torch.cat(drift_accum[horizon_idx], dim=0).float()
        tf_svals = torch.linalg.svdvals(tf_matrix)
        ro_svals = torch.linalg.svdvals(ro_matrix)
        drift_svals = torch.linalg.svdvals(drift_matrix)
        tf_energy = float(tf_svals.square().sum().item()) if tf_svals.numel() else float("nan")
        ro_energy = float(ro_svals.square().sum().item()) if ro_svals.numel() else float("nan")
        drift_energy = float(drift_svals.square().sum().item()) if drift_svals.numel() else float("nan")
        singular_spectrum.append(
            {
                "horizon": float(horizon_idx + 1),
                "teacher_forced_top_singular_value": float(tf_svals[0].item()) if tf_svals.numel() else float("nan"),
                "rollout_top_singular_value": float(ro_svals[0].item()) if ro_svals.numel() else float("nan"),
                "drift_top_singular_value": float(drift_svals[0].item()) if drift_svals.numel() else float("nan"),
                "teacher_forced_energy": tf_energy,
                "rollout_energy": ro_energy,
                "drift_energy": drift_energy,
                "teacher_forced_energy_ratio_top1": float(tf_svals[0].square().item() / tf_energy) if tf_svals.numel() and tf_energy > 0 else float("nan"),
                "rollout_energy_ratio_top1": float(ro_svals[0].square().item() / ro_energy) if ro_svals.numel() and ro_energy > 0 else float("nan"),
                "drift_energy_ratio_top1": float(drift_svals[0].square().item() / drift_energy) if drift_svals.numel() and drift_energy > 0 else float("nan"),
            }
        )

    alignment_statistics = []
    alignment_cosine_samples = []
    assert alignment_values_accum is not None
    for horizon_idx in range(horizon_count):
        alignment_tensor = torch.cat(alignment_values_accum[horizon_idx], dim=0).float()
        alignment_cosine_samples.append([float(value) for value in alignment_tensor.tolist()])
        alignment_statistics.append(
            {
                "horizon": float(horizon_idx + 1),
                "mean": float(alignment_tensor.mean().item()),
                "std": float(alignment_tensor.std(unbiased=False).item()) if alignment_tensor.numel() > 1 else 0.0,
                "min": float(alignment_tensor.min().item()),
                "p10": float(torch.quantile(alignment_tensor, torch.tensor(0.1, device=alignment_tensor.device)).item()),
                "median": float(torch.quantile(alignment_tensor, torch.tensor(0.5, device=alignment_tensor.device)).item()),
                "p90": float(torch.quantile(alignment_tensor, torch.tensor(0.9, device=alignment_tensor.device)).item()),
                "max": float(alignment_tensor.max().item()),
            }
        )

    return RolloutValidationResult(
        checkpoint_path=str(checkpoint_path),
        cache_path=str(cache_dir) if cache_dir is not None else "",
        source_split=source_split,
        predictor_mode=bundle.predictor_mode,
        subset_size=subset_size,
        image_size=image_size,
        context_frames=context_frames,
        future_frames=future_frames,
        total_frames=total_frames,
        model_context_steps=bundle.context_steps,
        latent_dim=bundle.latent_dim,
        num_samples=len(dataset),
        device=str(device_obj),
        horizon_metrics=horizon_metrics,
        alignment_cosine_samples=alignment_cosine_samples,
        alignment_statistics=alignment_statistics,
        singular_spectrum=singular_spectrum,
        gradient_norms=gradient_norms,
    )


def write_rollout_validation_report(path: str | Path, report: RolloutValidationResult) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    return output







