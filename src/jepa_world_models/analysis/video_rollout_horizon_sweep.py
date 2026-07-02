from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from jepa_world_models.analysis.video_world_model_reporting import write_rollout_validation_plots
from jepa_world_models.analysis.video_world_model_validation import (
    build_rollout_validation_report,
    write_rollout_validation_report,
)


@dataclass(slots=True)
class RolloutHorizonSweepEntry:
    future_seconds: float
    future_frames: int
    report_path: str
    plot_dir: str
    final_horizon_metrics: dict[str, float]

    def to_json(self) -> dict:
        return {
            "future_seconds": self.future_seconds,
            "future_frames": self.future_frames,
            "report_path": self.report_path,
            "plot_dir": self.plot_dir,
            "final_horizon_metrics": self.final_horizon_metrics,
        }


@dataclass(slots=True)
class RolloutHorizonSweepResult:
    checkpoint_path: str
    data_root: str
    source_split: str
    context_seconds: float
    context_frames: int
    sample_fps: float
    subset_size: int
    image_size: int
    feature_batch_size: int
    seed: int
    device: str
    entries: list[RolloutHorizonSweepEntry]
    output_dir: str
    summary_plot_path: str = ""

    def to_json(self) -> dict:
        return {
            "checkpoint_path": self.checkpoint_path,
            "data_root": self.data_root,
            "source_split": self.source_split,
            "context_seconds": self.context_seconds,
            "context_frames": self.context_frames,
            "sample_fps": self.sample_fps,
            "subset_size": self.subset_size,
            "image_size": self.image_size,
            "feature_batch_size": self.feature_batch_size,
            "seed": self.seed,
            "device": self.device,
            "output_dir": self.output_dir,
            "entries": [entry.to_json() for entry in self.entries],
            "summary_plot_path": self.summary_plot_path,
        }


def _seconds_to_frame_count(seconds: float, sample_fps: float) -> int:
    frames = max(2, int(round(seconds * sample_fps)))
    return frames + (frames % 2)


def _slug_seconds(seconds: float) -> str:
    if float(seconds).is_integer():
        return f"{int(seconds)}s"
    return f"{str(seconds).replace('.', 'p')}s"


def _final_horizon_metrics(report) -> dict[str, float]:
    final_row = report.horizon_metrics[-1]
    return {
        "horizon": float(final_row["horizon"]),
        "teacher_forced_mse": float(final_row["teacher_forced_mse"]),
        "rollout_mse": float(final_row["rollout_mse"]),
        "drift_norm": float(final_row["drift_norm"]),
        "alignment_cosine": float(final_row["alignment_cosine"]),
        "local_lipschitz_ratio_mean": float(final_row["local_lipschitz_ratio_mean"]),
        "decomposition_max_abs_error": float(final_row["decomposition_max_abs_error"]),
        "norm_identity_max_abs_error": float(final_row["norm_identity_max_abs_error"]),
    }


def _plot_horizon_sweep(result: RolloutHorizonSweepResult, output_dir: Path) -> Path:
    if not result.entries:
        raise ValueError("No sweep entries to plot.")

    future_seconds = [entry.future_seconds for entry in result.entries]
    future_frames = [entry.future_frames for entry in result.entries]
    metrics = [entry.final_horizon_metrics for entry in result.entries]

    tf_mse = [row["teacher_forced_mse"] for row in metrics]
    rollout_mse = [row["rollout_mse"] for row in metrics]
    drift_norm = [row["drift_norm"] for row in metrics]
    alignment = [row["alignment_cosine"] for row in metrics]
    local_ratio = [row["local_lipschitz_ratio_mean"] for row in metrics]
    decomposition_error = [row["decomposition_max_abs_error"] for row in metrics]

    x = np.asarray(future_frames, dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.reshape(2, 3)

    axes[0, 0].plot(x, tf_mse, marker="o", label="teacher-forced")
    axes[0, 0].plot(x, rollout_mse, marker="o", label="rollout")
    axes[0, 0].set_title("Final-horizon MSE")
    axes[0, 0].set_xlabel("future frames")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.2)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(x, drift_norm, marker="o", color="tab:purple")
    axes[0, 1].set_title("Final-horizon drift norm")
    axes[0, 1].set_xlabel("future frames")
    axes[0, 1].grid(True, alpha=0.2)

    axes[0, 2].plot(x, alignment, marker="o", color="tab:red")
    axes[0, 2].set_title("Final-horizon alignment cosine")
    axes[0, 2].set_xlabel("future frames")
    axes[0, 2].set_ylim(-1.0, 1.0)
    axes[0, 2].grid(True, alpha=0.2)

    axes[1, 0].plot(x, local_ratio, marker="o", color="tab:cyan")
    axes[1, 0].set_title("Final-horizon local amplification")
    axes[1, 0].set_xlabel("future frames")
    axes[1, 0].grid(True, alpha=0.2)

    axes[1, 1].plot(x, decomposition_error, marker="o", color="tab:green")
    axes[1, 1].set_title("Final-horizon decomposition residual")
    axes[1, 1].set_xlabel("future frames")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.2)

    axes[1, 2].plot(future_seconds, rollout_mse, marker="o", color="tab:orange")
    axes[1, 2].set_title("Rollout MSE vs seconds")
    axes[1, 2].set_xlabel("future seconds")
    axes[1, 2].grid(True, alpha=0.2)

    fig.suptitle(
        f"Horizon sweep for {Path(result.checkpoint_path).name} "
        f"(context={result.context_frames} frames, sample_fps={result.sample_fps:g})"
    )
    fig.tight_layout()

    output_path = output_dir / "horizon_sweep_plot.png"
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_rollout_horizon_sweep(
    *,
    checkpoint_path: str | Path,
    data_root: str | Path,
    source_split: str = "train",
    subset_size: int = 128,
    image_size: int = 224,
    context_seconds: float = 4.0,
    future_seconds_grid: Sequence[float],
    sample_fps: float = 4.0,
    feature_batch_size: int = 1,
    seed: int = 0,
    cache_dir: str | Path | None = "logs/video_world_model/cache",
    output_dir: str | Path = "logs/video_world_model/horizon_sweep",
    device: str | None = None,
    batch_limit: int | None = 1,
) -> RolloutHorizonSweepResult:
    checkpoint_path = str(checkpoint_path)
    data_root = str(data_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    context_frames = _seconds_to_frame_count(context_seconds, sample_fps)
    entries: list[RolloutHorizonSweepEntry] = []

    for future_seconds in future_seconds_grid:
        future_frames = _seconds_to_frame_count(float(future_seconds), sample_fps)
        total_frames = context_frames + future_frames
        run_dir = output_path / f"horizon_{_slug_seconds(float(future_seconds))}"
        run_dir.mkdir(parents=True, exist_ok=True)

        report = build_rollout_validation_report(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            source_split=source_split,
            subset_size=subset_size,
            image_size=image_size,
            total_frames=total_frames,
            context_frames=context_frames,
            future_frames=future_frames,
            sample_fps=sample_fps,
            feature_batch_size=feature_batch_size,
            seed=seed,
            batch_limit=batch_limit,
            cache_dir=cache_dir,
            device=device,
        )
        report_path = write_rollout_validation_report(run_dir / "rollout_validation.json", report)
        plot_paths = write_rollout_validation_plots(report, run_dir / "plots")
        entry = RolloutHorizonSweepEntry(
            future_seconds=float(future_seconds),
            future_frames=future_frames,
            report_path=str(report_path),
            plot_dir=str(Path(plot_paths["rollout_validation"]).parent),
            final_horizon_metrics=_final_horizon_metrics(report),
        )
        entries.append(entry)

    result = RolloutHorizonSweepResult(
        checkpoint_path=checkpoint_path,
        data_root=data_root,
        source_split=source_split,
        context_seconds=context_seconds,
        context_frames=context_frames,
        sample_fps=sample_fps,
        subset_size=subset_size,
        image_size=image_size,
        feature_batch_size=feature_batch_size,
        seed=seed,
        device=str(device or ""),
        entries=entries,
        output_dir=str(output_path),
    )
    summary_path = output_path / "horizon_sweep_summary.json"
    plot_path = _plot_horizon_sweep(result, output_path)
    result.summary_plot_path = str(plot_path)
    summary_path.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return result
