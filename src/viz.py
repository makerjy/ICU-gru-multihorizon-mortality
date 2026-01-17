from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
        }
    )


def plot_training_history(history: dict, out_dir: Path) -> None:
    if not history.get("epoch"):
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    epochs = history["epoch"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(epochs, history["train_loss"], label="train_loss", color="#1f77b4")
    ax.plot(epochs, history["valid_loss"], label="valid_loss", color="#ff7f0e")
    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(epochs, history.get("row_auc", []), label="valid_row_auc", color="#2ca02c")
    ax.plot(epochs, history.get("row_ap", []), label="valid_row_ap", color="#d62728")
    ax.set_title("Validation Row-level AUC/AP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "valid_row_auc_ap.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(epochs, history.get("stay_auc", []), label="valid_stay_auc", color="#1f77b4")
    ax.plot(epochs, history.get("stay_ap", []), label="valid_stay_ap", color="#ff7f0e")
    ax.set_title("Validation Stay-level AUC/AP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "valid_stay_auc_ap.png", dpi=200)
    plt.close(fig)


def plot_stay_metrics(stay_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    metrics = ["auc", "ap", "f1"]
    labels = stay_df["method"].to_numpy()
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for offset, split in [(-width / 2, "valid"), (width / 2, "test")]:
            vals = stay_df[f"{split}_{metric}"].to_numpy()
            ax.bar(x + offset, vals, width=width, label=split.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_ylim(0, 1)
        ax.set_title(metric.upper())
        ax.grid(axis="y")
        if idx == 0:
            ax.set_ylabel("Score")
        ax.legend()

    fig.suptitle("Stay-level Metrics")
    fig.tight_layout()
    fig.savefig(out_dir / "stay_metrics.png", dpi=200)
    plt.close(fig)


def plot_metrics_table(df: pd.DataFrame, title: str, out_path: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 0.5 + 0.35 * len(df)))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.3)

    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
