from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from jepa_world_models.analysis.video_world_model import LatentWorldModelResult
from jepa_world_models.analysis.video_world_model_validation import RolloutValidationResult


_METRIC_LABELS = {
    "latent_mse": "Latent MSE",
    "normalized_latent_mse": "Normalized latent MSE",
    "cosine_similarity": "Cosine similarity",
}


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _plot_step_history(result: LatentWorldModelResult, output_dir: Path) -> Path:
    if not result.step_history:
        raise ValueError("step_history is empty; cannot plot training steps.")

    steps = np.arange(len(result.step_history))
    combined = [row["combined_loss"] for row in result.step_history]
    objective = [row.get("objective_loss", row["combined_loss"]) for row in result.step_history]
    mse = [row["mse_loss"] for row in result.step_history]
    norm_mse = [row["normalized_mse_loss"] for row in result.step_history]
    cosine = [row["cosine_loss"] for row in result.step_history]
    rollout = [row.get("rollout_loss", row["combined_loss"]) for row in result.step_history]
    delta = [row.get("delta_loss", row["combined_loss"]) for row in result.step_history]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, combined, linewidth=1.2, label="combined")
    ax.plot(steps, objective, linewidth=1.0, label="objective")
    ax.plot(steps, mse, linewidth=1.0, label="mse")
    ax.plot(steps, norm_mse, linewidth=1.0, label="normalized mse")
    ax.plot(steps, cosine, linewidth=1.0, label="cosine loss")
    ax.plot(steps, rollout, linewidth=1.0, label="rollout")
    ax.plot(steps, delta, linewidth=1.0, label="delta")
    ax.set_title("Training step loss components")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = output_dir / "training_steps.png"
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_epoch_history(result: LatentWorldModelResult, output_dir: Path) -> Path:
    if not result.history:
        raise ValueError("history is empty; cannot plot epochs.")

    epochs = [row["epoch"] for row in result.history]
    train_loss = [row["train_loss"] for row in result.history]
    val_loss = [row["val_loss"] for row in result.history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", linewidth=1.5, label="train combined")
    ax.plot(epochs, val_loss, marker="o", linewidth=1.5, label="val combined")
    ax.set_title("Epoch-level combined loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    if result.best_epoch:
        ax.axvline(result.best_epoch, color="tab:green", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path = output_dir / "training_history.png"
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_epoch_components(result: LatentWorldModelResult, output_dir: Path) -> Path:
    if not result.history:
        raise ValueError("history is empty; cannot plot epoch components.")

    epochs = [row["epoch"] for row in result.history]
    metrics = [
        ("combined_loss", "Combined loss"),
        ("objective_loss", "Objective loss"),
        ("mse_loss", "MSE loss"),
        ("normalized_mse_loss", "Normalized MSE loss"),
        ("cosine_loss", "Cosine loss"),
        ("rollout_loss", "Rollout loss"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.reshape(2, 3)
    for axis, (metric_key, metric_title) in zip(axes.flat, metrics):
        if metric_key == "combined_loss":
            train_values = [row["train_loss"] for row in result.history]
            val_values = [row["val_loss"] for row in result.history]
        else:
            train_values = [row.get(f"train_{metric_key}", row["train_loss"]) for row in result.history]
            val_values = [row.get(f"val_{metric_key}", row["val_loss"]) for row in result.history]
        axis.plot(epochs, train_values, marker="o", linewidth=1.5, label="train")
        axis.plot(epochs, val_values, marker="o", linewidth=1.5, label="val")
        axis.set_title(metric_title)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.2)
        axis.set_yscale("log")
    axes[0, 0].legend(frameon=False)
    for axis in axes.flat[len(metrics):]:
        axis.axis("off")
    fig.suptitle("Epoch-level loss components")
    fig.tight_layout()

    output_path = output_dir / "training_components.png"
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_metric_comparison(result: LatentWorldModelResult, output_dir: Path) -> Path:
    splits = ["train", "val", "test"]
    model_metrics = [result.train_metrics, result.val_metrics, result.test_metrics]
    methods = ["model", "repeat_last", "mean_context"]
    colors = {"model": "tab:blue", "repeat_last": "tab:orange", "mean_context": "tab:green"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    bar_width = 0.24
    x = np.arange(len(splits))

    for axis, metric_name in zip(axes, ["latent_mse", "normalized_latent_mse", "cosine_similarity"]):
        values = [
            [split_metrics[metric_name] for split_metrics in model_metrics],
            [result.baseline_metrics[split]["repeat_last"][metric_name] for split in splits],
            [result.baseline_metrics[split]["mean_context"][metric_name] for split in splits],
        ]
        for offset, method in enumerate(methods):
            axis.bar(x + (offset - 1) * bar_width, values[offset], width=bar_width, color=colors[method], label=method if metric_name == "latent_mse" else None)
        axis.set_title(_METRIC_LABELS[metric_name])
        axis.set_xticks(x)
        axis.set_xticklabels(splits)
        axis.grid(True, axis="y", alpha=0.2)
        if metric_name == "cosine_similarity":
            axis.set_ylim(0.95, 1.0)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Model vs trivial baselines")
    fig.tight_layout()

    output_path = output_dir / "metric_comparison.png"
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_rollout_validation(report: RolloutValidationResult, output_dir: Path) -> Path:
    horizon = [row["horizon"] for row in report.horizon_metrics]
    tf_mse = [row["teacher_forced_mse"] for row in report.horizon_metrics]
    ro_mse = [row["rollout_mse"] for row in report.horizon_metrics]
    drift = [row["drift_norm"] for row in report.horizon_metrics]
    alignment = [row["alignment_cosine"] for row in report.horizon_metrics]
    alignment_std = [row.get("alignment_cosine_std", 0.0) for row in report.horizon_metrics]
    local_ratio = [row["local_lipschitz_ratio_mean"] for row in report.horizon_metrics]
    grad_norm = [row["gradient_norm"] for row in report.gradient_norms]
    grad_horizon = [row["horizon"] for row in report.gradient_norms]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(2, 2)

    axes[0, 0].plot(horizon, tf_mse, marker="o", label="teacher-forced")
    axes[0, 0].plot(horizon, ro_mse, marker="o", label="rollout")
    axes[0, 0].set_title("Horizon MSE")
    axes[0, 0].set_xlabel("horizon step")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.2)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(horizon, drift, marker="o", color="tab:purple", label="drift norm")
    axes[0, 1].plot(horizon, local_ratio, marker="o", color="tab:cyan", label="local ratio")
    axes[0, 1].set_title("Drift and amplification")
    axes[0, 1].set_xlabel("horizon step")
    axes[0, 1].grid(True, alpha=0.2)
    axes[0, 1].legend(frameon=False)

    if report.alignment_cosine_samples:
        axes[1, 0].boxplot(report.alignment_cosine_samples, positions=horizon, widths=0.45, showfliers=False)
        axes[1, 0].plot(horizon, alignment, marker="o", color="tab:red", linewidth=1.2, label="mean")
        axes[1, 0].fill_between(
            horizon,
            np.array(alignment) - np.array(alignment_std),
            np.array(alignment) + np.array(alignment_std),
            color="tab:red",
            alpha=0.15,
            label="std",
        )
        axes[1, 0].legend(frameon=False)
    else:
        axes[1, 0].plot(horizon, alignment, marker="o", color="tab:red")
    axes[1, 0].set_title("Alignment cosine distribution")
    axes[1, 0].set_xlabel("horizon step")
    axes[1, 0].set_ylim(-1.0, 1.0)
    axes[1, 0].grid(True, alpha=0.2)

    axes[1, 1].plot(grad_horizon, grad_norm, marker="o", color="tab:green")
    axes[1, 1].set_title("Per-horizon gradient norm")
    axes[1, 1].set_xlabel("horizon step")
    axes[1, 1].grid(True, alpha=0.2)

    fig.suptitle(
        f"Rollout validation: max decomposition error={max(r['decomposition_max_abs_error'] for r in report.horizon_metrics):.2e}, "
        f"max norm identity error={max(r['norm_identity_max_abs_error'] for r in report.horizon_metrics):.2e}"
    )
    fig.tight_layout()

    output_path = output_dir / "rollout_validation.png"
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_rollout_spectrum(report: RolloutValidationResult, output_dir: Path) -> Path:
    horizon = [row["horizon"] for row in report.singular_spectrum]
    tf_top = [row["teacher_forced_top_singular_value"] for row in report.singular_spectrum]
    ro_top = [row["rollout_top_singular_value"] for row in report.singular_spectrum]
    drift_top = [row["drift_top_singular_value"] for row in report.singular_spectrum]
    tf_ratio = [row["teacher_forced_energy_ratio_top1"] for row in report.singular_spectrum]
    ro_ratio = [row["rollout_energy_ratio_top1"] for row in report.singular_spectrum]
    drift_ratio = [row["drift_energy_ratio_top1"] for row in report.singular_spectrum]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(horizon, tf_top, marker="o", label="teacher-forced")
    axes[0].plot(horizon, ro_top, marker="o", label="rollout")
    axes[0].plot(horizon, drift_top, marker="o", label="drift")
    axes[0].set_title("Top singular value by horizon")
    axes[0].set_xlabel("horizon step")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].plot(horizon, tf_ratio, marker="o", label="teacher-forced")
    axes[1].plot(horizon, ro_ratio, marker="o", label="rollout")
    axes[1].plot(horizon, drift_ratio, marker="o", label="drift")
    axes[1].set_title("Top singular energy ratio")
    axes[1].set_xlabel("horizon step")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(frameon=False)

    fig.suptitle("Rollout singular-spectrum diagnostics")
    fig.tight_layout()

    output_path = output_dir / "rollout_spectrum.png"
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_video_world_model_plots(
    result: LatentWorldModelResult,
    output_dir: str | Path,
    validation_report: RolloutValidationResult | None = None,
) -> dict[str, str]:
    output = _ensure_output_dir(output_dir)
    paths = {
        "training_steps": str(_plot_step_history(result, output)),
        "training_history": str(_plot_epoch_history(result, output)),
        "training_components": str(_plot_epoch_components(result, output)),
        "metric_comparison": str(_plot_metric_comparison(result, output)),
    }
    if validation_report is not None:
        paths["rollout_validation"] = str(_plot_rollout_validation(validation_report, output))
        paths["rollout_spectrum"] = str(_plot_rollout_spectrum(validation_report, output))
    return paths


def write_rollout_validation_plots(
    report: RolloutValidationResult,
    output_dir: str | Path,
) -> dict[str, str]:
    output = _ensure_output_dir(output_dir)
    return {
        "rollout_validation": str(_plot_rollout_validation(report, output)),
        "rollout_spectrum": str(_plot_rollout_spectrum(report, output)),
    }

