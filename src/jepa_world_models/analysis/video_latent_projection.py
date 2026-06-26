from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from jepa_world_models.analysis.common import resolve_device
from jepa_world_models.analysis.video_reconstruction import load_clip_from_video_path
from jepa_world_models.analysis.video_world_model import (
    LatentSequenceBank,
    LatentWorldModelBundle,
    TemporalLatentPredictor,
    _baseline_mean,
    _baseline_repeat_last,
    _latent_metrics,
    _load_model,
    build_latent_sequence_bank,
)


@dataclass(slots=True)
class LatentProjectionExample:
    index: int
    sample_index: int
    video_id: str
    filename: str
    video_url: str
    context_frames: int
    future_frames: int
    total_frames: int


@dataclass(slots=True)
class LatentProjectionPoint:
    step_index: int
    frame_index: int
    x: float
    y: float
    step_distance: float
    phase: str


@dataclass(slots=True)
class LatentProjectionResult:
    query: dict[str, Any]
    projection_method: str
    bounds: dict[str, float]
    background_points: list[dict[str, Any]]
    context_trajectory: list[LatentProjectionPoint]
    future_true_trajectory: list[LatentProjectionPoint]
    future_pred_trajectory: list[LatentProjectionPoint]
    metrics: dict[str, float]
    baseline_metrics: dict[str, dict[str, float]]

    def to_json(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "projection_method": self.projection_method,
            "bounds": self.bounds,
            "background_points": self.background_points,
            "context_trajectory": [asdict(point) for point in self.context_trajectory],
            "future_true_trajectory": [asdict(point) for point in self.future_true_trajectory],
            "future_pred_trajectory": [asdict(point) for point in self.future_pred_trajectory],
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
        }


@dataclass(slots=True)
class LatentProjectionEngine:
    bank: LatentSequenceBank
    predictor: TemporalLatentPredictor
    encoder: Any
    bundle: LatentWorldModelBundle
    device: torch.device
    projection_seed: int
    background_features: np.ndarray
    background_projection: np.ndarray
    pca: PCA

    @property
    def context_steps(self) -> int:
        return self.bank.context_steps

    @property
    def future_steps(self) -> int:
        return self.bank.future_steps

    @property
    def total_frames(self) -> int:
        return self.bank.total_frames

    @property
    def latent_dim(self) -> int:
        return self.bank.latent_dim

    @property
    def background_size(self) -> int:
        return int(self.background_features.shape[0])

    def list_examples(self, limit: int = 12) -> list[LatentProjectionExample]:
        count = min(limit, len(self.bank.sample_indices))
        examples: list[LatentProjectionExample] = []
        for index in range(count):
            video_id = self.bank.video_ids[index]
            examples.append(
                LatentProjectionExample(
                    index=index,
                    sample_index=self.bank.sample_indices[index],
                    video_id=video_id,
                    filename=f"{video_id}.webm",
                    video_url=f"/api/latent/file/{video_id}.webm",
                    context_frames=self.bank.context_frames,
                    future_frames=self.bank.future_frames,
                    total_frames=self.bank.total_frames,
                )
            )
        return examples

    def _sample_background_indices(self, size: int, *, seed: int) -> np.ndarray:
        size = min(size, self.background_size)
        if size <= 0:
            return np.empty((0,), dtype=np.int64)
        rng = np.random.default_rng(seed)
        if size == self.background_size:
            return np.arange(self.background_size, dtype=np.int64)
        return np.sort(rng.choice(self.background_size, size=size, replace=False))

    def _project_with_tsne(self, background_indices: np.ndarray, query_features: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        background = self.background_features[background_indices]
        combined = np.concatenate([background, query_features], axis=0)
        if combined.shape[0] < 4:
            combined_proj = self.pca.transform(combined)
            return combined_proj[: background.shape[0]], combined_proj[background.shape[0] :]
        perplexity = min(30.0, max(5.0, (combined.shape[0] - 1) / 3.0))
        perplexity = min(perplexity, float(combined.shape[0] - 1))
        projector = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=seed,
            max_iter=750,
            method="barnes_hut",
        )
        combined_proj = projector.fit_transform(combined)
        return combined_proj[: background.shape[0]], combined_proj[background.shape[0] :]

    def _project_with_pca(self, background_indices: np.ndarray, query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        background_proj = self.background_projection[background_indices]
        query_proj = self.pca.transform(query_features)
        return background_proj, query_proj

    def _result_from_latents(
        self,
        *,
        index: int,
        sample_index: int,
        video_id: str,
        filename: str,
        video_url: str,
        context: torch.Tensor,
        future_true: torch.Tensor,
        future_pred: torch.Tensor,
        projection_method: str,
        background_sample_size: int,
        seed: int,
        source_split: str,
    ) -> LatentProjectionResult:
        query_features = torch.cat([context, future_true, future_pred], dim=0).cpu().numpy()
        background_indices = self._sample_background_indices(background_sample_size, seed=seed + index)

        method = projection_method.lower().strip()
        if method == "tsne":
            background_proj, query_proj = self._project_with_tsne(background_indices, query_features, seed=seed + index)
        else:
            method = "pca"
            background_proj, query_proj = self._project_with_pca(background_indices, query_features)

        context_proj = query_proj[: self.context_steps]
        future_true_proj = query_proj[self.context_steps : self.context_steps + self.future_steps]
        future_pred_proj = query_proj[self.context_steps + self.future_steps :]

        all_coords = np.concatenate([background_proj, query_proj], axis=0)
        x_min = float(np.min(all_coords[:, 0]))
        x_max = float(np.max(all_coords[:, 0]))
        y_min = float(np.min(all_coords[:, 1]))
        y_max = float(np.max(all_coords[:, 1]))
        bounds = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

        background_points = [
            {
                "index": int(background_idx),
                "video_id": self.bank.video_ids[background_idx],
                "sample_index": int(self.bank.sample_indices[background_idx]),
                "x": float(point[0]),
                "y": float(point[1]),
            }
            for background_idx, point in zip(background_indices.tolist(), background_proj, strict=False)
        ]

        def make_points(coords: np.ndarray, *, phase: str, frame_offset: int) -> list[LatentProjectionPoint]:
            points: list[LatentProjectionPoint] = []
            previous: np.ndarray | None = None
            for step_index, point in enumerate(coords):
                current = np.asarray(point, dtype=np.float32)
                if previous is None:
                    step_distance = 0.0
                else:
                    step_distance = float(np.linalg.norm(current - previous))
                points.append(
                    LatentProjectionPoint(
                        step_index=step_index,
                        frame_index=frame_offset + step_index,
                        x=float(current[0]),
                        y=float(current[1]),
                        step_distance=step_distance,
                        phase=phase,
                    )
                )
                previous = current
            return points

        context_trajectory = make_points(context_proj, phase="context", frame_offset=0)
        future_true_trajectory = make_points(future_true_proj, phase="future_true", frame_offset=self.context_steps)
        future_pred_trajectory = make_points(future_pred_proj, phase="future_pred", frame_offset=self.context_steps)

        metrics = _latent_metrics(future_pred.unsqueeze(0), future_true.unsqueeze(0))
        repeat = _baseline_repeat_last(context.unsqueeze(0), self.future_steps)
        mean = _baseline_mean(context.unsqueeze(0), self.future_steps)
        baseline_metrics = {
            "repeat_last": _latent_metrics(repeat, future_true.unsqueeze(0)),
            "mean_context": _latent_metrics(mean, future_true.unsqueeze(0)),
        }

        query = {
            "index": int(index),
            "sample_index": int(sample_index),
            "video_id": video_id,
            "filename": filename,
            "video_url": video_url,
            "source_split": source_split,
            "context_frames": self.bank.context_frames,
            "future_frames": self.bank.future_frames,
            "total_frames": self.bank.total_frames,
            "context_steps": self.bank.context_steps,
            "future_steps": self.bank.future_steps,
            "latent_dim": self.bank.latent_dim,
            "projection_method": method,
            "source_kind": "upload" if source_split == "upload" else "dataset",
        }

        return LatentProjectionResult(
            query=query,
            projection_method=method,
            bounds=bounds,
            background_points=background_points,
            context_trajectory=context_trajectory,
            future_true_trajectory=future_true_trajectory,
            future_pred_trajectory=future_pred_trajectory,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
        )

    def analyze_index(
        self,
        index: int,
        *,
        projection_method: str = "pca",
        background_sample_size: int = 512,
        seed: int | None = None,
    ) -> LatentProjectionResult:
        if not self.bank.sample_indices:
            raise RuntimeError("Latent sequence bank is empty.")
        index = max(0, min(index, len(self.bank.sample_indices) - 1))
        seed = self.projection_seed if seed is None else seed

        context = self.bank.context_latents[index].float()
        future_true = self.bank.future_latents[index].float()
        with torch.inference_mode():
            future_pred = self.predictor(context.unsqueeze(0).to(self.device)).detach().cpu()[0].float()

        return self._result_from_latents(
            index=index,
            sample_index=int(self.bank.sample_indices[index]),
            video_id=self.bank.video_ids[index],
            filename=f"{self.bank.video_ids[index]}.webm",
            video_url=f"/api/latent/file/{self.bank.video_ids[index]}.webm",
            context=context,
            future_true=future_true,
            future_pred=future_pred,
            projection_method=projection_method,
            background_sample_size=background_sample_size,
            seed=seed,
            source_split=self.bank.source_split,
        )

    def analyze_uploaded_path(
        self,
        video_path: str | Path,
        *,
        filename: str,
        projection_method: str = "pca",
        background_sample_size: int = 512,
        seed: int | None = None,
        video_id: str = "upload",
    ) -> LatentProjectionResult:
        seed = self.projection_seed if seed is None else seed
        clip = load_clip_from_video_path(video_path, num_frames=self.total_frames, image_size=self.bundle.image_size)
        context_frames = self.bundle.context_frames
        context_clip = clip[:context_frames].unsqueeze(0).to(self.device)
        future_clip = clip[context_frames:].unsqueeze(0).to(self.device)
        with torch.inference_mode():
            context = self.encoder.encoder.forward_sequence(context_clip).detach().cpu()[0].float()
            future_true = self.encoder.encoder.forward_sequence(future_clip).detach().cpu()[0].float()
            future_pred = self.predictor(context.unsqueeze(0).to(self.device)).detach().cpu()[0].float()

        return self._result_from_latents(
            index=0,
            sample_index=-1,
            video_id=video_id,
            filename=filename,
            video_url=f"/api/latent/upload/file/{video_id}/{Path(filename).name}",
            context=context,
            future_true=future_true,
            future_pred=future_pred,
            projection_method=projection_method,
            background_sample_size=background_sample_size,
            seed=seed,
            source_split="upload",
        )


def load_video_world_model(checkpoint_path: str | Path, device: str | torch.device | None = None) -> tuple[TemporalLatentPredictor, LatentWorldModelBundle]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle = LatentWorldModelBundle.from_payload(payload)
    model = TemporalLatentPredictor(
        latent_dim=bundle.latent_dim,
        context_steps=bundle.context_steps,
        future_steps=bundle.future_steps,
        hidden_dim=bundle.hidden_dim,
        num_layers=bundle.num_layers,
        num_heads=bundle.num_heads,
        dropout=bundle.dropout,
    )
    model.load_state_dict(bundle.state_dict)
    device_obj = resolve_device(device)
    model = model.to(device_obj)
    model.eval()
    return model, bundle


def build_latent_projection_engine(
    *,
    world_model_checkpoint: str | Path,
    data_root: str | Path,
    source_split: str = "train",
    subset_size: int = 256,
    image_size: int = 224,
    total_frames: int = 26,
    context_frames: int = 20,
    future_frames: int = 6,
    feature_batch_size: int = 1,
    cache_dir: str | Path | None = "logs/video_world_model/cache",
    seed: int = 0,
    device: str | torch.device | None = None,
) -> LatentProjectionEngine:
    predictor, bundle = load_video_world_model(world_model_checkpoint, device=device)
    encoder = _load_model(bundle.checkpoint_path, device=device)
    if (bundle.context_frames, bundle.future_frames, bundle.total_frames) != (context_frames, future_frames, total_frames):
        raise ValueError(
            "The projection view must use the same context/future frame counts as the world-model checkpoint."
        )
    if bundle.image_size != image_size:
        raise ValueError("The projection view must use the same image size as the world-model checkpoint.")

    bank = build_latent_sequence_bank(
        checkpoint_path=bundle.checkpoint_path,
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
    background_features = bank.context_latents.float().mean(dim=1).cpu().numpy()
    pca = PCA(n_components=2, random_state=seed)
    background_projection = pca.fit_transform(background_features)
    return LatentProjectionEngine(
        bank=bank,
        predictor=predictor,
        encoder=encoder,
        bundle=bundle,
        device=resolve_device(device),
        projection_seed=seed,
        background_features=background_features,
        background_projection=background_projection,
        pca=pca,
    )


def save_latent_projection_report(path: str | Path, result: LatentProjectionResult) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return str(path)









