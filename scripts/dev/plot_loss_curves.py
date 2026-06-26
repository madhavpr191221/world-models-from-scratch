"""
Plot VICReg loss curves from the CSV log produced by train_one_epoch.

Usage (from project root, run any time during or after training):
    uv run python scripts/dev/plot_loss_curves.py

Reads logs/vicreg/loss_log.csv and produces a 4-panel plot showing
L_total, L_inv, L_var, L_cov over training steps. The diagnostic to
watch: L_var trending UP toward gamma=1 while L_cov trends toward 0
SIMULTANEOUSLY is the collapse signature, even if L_total looks fine.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

LOG_PATH = "logs/vicreg/loss_log.csv"
OUTPUT_PATH = "logs/vicreg/loss_curves.png"
GAMMA = 1.0  # target std, for reference line on the L_var plot


def main() -> None:
    log_path = Path(LOG_PATH)
    if not log_path.exists():
        raise FileNotFoundError(
            f"No log file found at {log_path}. Run training first "
            f"(scripts/dev/run_few_epochs.py) to generate it."
        )

    steps, l_total, l_inv, l_var, l_cov = [], [], [], [], []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["global_step"]))
            l_total.append(float(row["L_total"]))
            l_inv.append(float(row["L_inv"]))
            l_var.append(float(row["L_var"]))
            l_cov.append(float(row["L_cov"]))

    print(f"Loaded {len(steps)} logged steps from {log_path}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("VICReg training loss components", fontsize=14)

    # L_total
    axes[0, 0].plot(steps, l_total, color="black", linewidth=0.8)
    axes[0, 0].set_title("L_total")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_yscale("log")  # log scale: total can span orders of magnitude

    # L_inv
    axes[0, 1].plot(steps, l_inv, color="tab:blue", linewidth=0.8)
    axes[0, 1].set_title("L_inv (invariance)")
    axes[0, 1].set_xlabel("step")

    # L_var -- THE key collapse indicator. Reference line at gamma.
    axes[1, 0].plot(steps, l_var, color="tab:red", linewidth=0.8)
    axes[1, 0].axhline(GAMMA, color="red", linestyle="--", alpha=0.5,
                        label=f"gamma={GAMMA} (collapse ceiling)")
    axes[1, 0].set_title("L_var (variance) -- watch for upward drift toward gamma")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylim(0, GAMMA * 1.1)
    axes[1, 0].legend(fontsize=8)

    # L_cov -- secondary collapse indicator (crashing to exactly 0 = bad sign)
    axes[1, 1].plot(steps, l_cov, color="tab:green", linewidth=0.8)
    axes[1, 1].set_title("L_cov (covariance) -- watch for collapse toward exactly 0")
    axes[1, 1].set_xlabel("step")

    plt.tight_layout()

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    print(f"Saved plot to {output_path}")

    # Print a quick textual diagnostic alongside the plot
    if len(l_var) > 10:
        early_var = sum(l_var[:10]) / 10
        late_var = sum(l_var[-10:]) / 10
        early_cov = sum(l_cov[:10]) / 10
        late_cov = sum(l_cov[-10:]) / 10
        print(f"\nDiagnostic summary:")
        print(f"  L_var: early avg={early_var:.4f} -> late avg={late_var:.4f} "
              f"({'INCREASING toward gamma -- possible collapse!' if late_var > early_var else 'decreasing or stable -- healthy'})")
        print(f"  L_cov: early avg={early_cov:.4f} -> late avg={late_cov:.4f} "
              f"({'crashing toward 0 -- possible collapse!' if late_cov < early_cov * 0.1 else 'stable -- healthy'})")


if __name__ == "__main__":
    main()
