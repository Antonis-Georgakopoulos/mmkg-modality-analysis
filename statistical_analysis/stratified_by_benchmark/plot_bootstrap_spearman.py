"""
Plot standalone heatmaps of the bootstrap Spearman correlation between models.
One heatmap per metric (D and C12), showing mean ρ across 100k replicates.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent
RESULTS_PATH = OUTPUT_DIR / "bootstrap_spearman_results.json"

MODEL_ORDER = [
    "Gemma3-4B", "Gemma3-27B", "GPT-4o-mini", "Qwen3-VL-8B", "Qwen3-VL-30B",
]

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


def _build_matrix(metric_data, field="mean_rho"):
    """Reconstruct 5×5 symmetric matrix from pairwise results."""
    n = len(MODEL_ORDER)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{MODEL_ORDER[i]} vs {MODEL_ORDER[j]}"
            val = metric_data[key][field]
            mat[i, j] = val
            mat[j, i] = val
    return mat


for metric in ("D", "C12"):
    mean_mat = _build_matrix(data[metric], "mean_rho")
    std_mat  = _build_matrix(data[metric], "std_rho")
    ci_lo_mat = _build_matrix(data[metric], "ci_lo")
    ci_hi_mat = _build_matrix(data[metric], "ci_hi")
    n = len(MODEL_ORDER)

    # ── Mean ρ heatmap ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean_mat, cmap="YlOrRd", vmin=0.3, vmax=1.0)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mean_mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(MODEL_ORDER, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.set_title(
    r"Spearman $\rho$",
    fontsize=12,
    fontweight="bold",
    pad=10
)
    fig.tight_layout()
    path = OUTPUT_DIR / f"bootstrap_spearman_{metric}_mean.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Std heatmap ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(std_mat, cmap="Blues", vmin=0.0, vmax=0.30)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{std_mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=11)

    ax.set_xticks(range(n))
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(MODEL_ORDER, fontsize=10)
    ax.set_title(f"{metric} — Std of Spearman ρ\n"
                 f"(uncertainty across 100,000 bootstrap replicates)",
                 fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = OUTPUT_DIR / f"bootstrap_spearman_{metric}_std.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── CI width heatmap (shows precision) ───────────────────────────
    ci_width = ci_hi_mat - ci_lo_mat
    np.fill_diagonal(ci_width, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ci_width, cmap="Purples", vmin=0.0, vmax=1.0)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11)
            else:
                ax.text(j, i, f"{ci_width[i, j]:.2f}", ha="center",
                        va="center", fontsize=11)

    ax.set_xticks(range(n))
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(MODEL_ORDER, fontsize=10)
    ax.set_title(f"{metric} — 95% CI Width of Spearman ρ\n"
                 f"(narrower = more stable agreement between models)",
                 fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = OUTPUT_DIR / f"bootstrap_spearman_{metric}_ci_width.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

print("\nDone.")
