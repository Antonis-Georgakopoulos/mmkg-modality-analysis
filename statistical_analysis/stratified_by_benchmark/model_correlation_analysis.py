"""
Model correlation analysis for stratified-by-benchmark bootstrap results.

Displays:
  1. Mean and Std of D and C12 per pair × model
  2. Spearman correlation across models (do models behave similarly across pairs?)

Produces plots:
  - Heatmap of bootstrap mean  (per pair × model) for D and C12
  - Grouped bar chart of bootstrap mean ± std for D and C12
  - Heatmap of Spearman correlation matrix between models for D and C12
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

# ── Style constants (consistent with other plots in this directory) ──────
MODEL_ORDER = [
    "gemma3_4b", "gemma3_27b", "gpt-4o-mini", "qwen3-vl_8b", "qwen3-vl_30b",
]
MODEL_DISPLAY = {
    "gemma3_4b":    "Gemma3-4B",
    "gemma3_27b":   "Gemma3-27B",
    "gpt-4o-mini":  "GPT-4o-mini",
    "qwen3-vl_8b":  "Qwen3-VL-8B",
    "qwen3-vl_30b": "Qwen3-VL-30B",
}
MODEL_COLORS = {
    "gemma3_4b":    "#2962FF",
    "gemma3_27b":   "#7B1FA2",
    "gpt-4o-mini":  "#00897B",
    "qwen3-vl_8b":  "#D32F2F",
    "qwen3-vl_30b": "#FF6F00",
}
PAIR_ORDER = [
    "image + layout", "image + plain_text", "layout + plain_text",
    "plain_text + table", "image + table", "layout + table",
]
PAIR_DISPLAY = {
    "image + layout":      "Image vs Layout",
    "image + plain_text":  "Image vs Text",
    "layout + plain_text": "Layout vs Text",
    "plain_text + table":  "Text vs Table",
    "image + table":       "Image vs Table",
    "layout + table":      "Layout vs Table",
}

RESULTS_PATH = Path(__file__).parent / "stratified_by_benchmark_results.json"
OUTPUT_DIR   = Path(__file__).parent


# ── Helpers ──────────────────────────────────────────────────────────────

def _build_tables(per_pair, metric):
    """Return (df_mean, df_std) with pairs as rows and models as columns."""
    key = f"per_model_{metric}"
    mean_data, std_data = {}, {}
    for pair in PAIR_ORDER:
        if pair not in per_pair:
            continue
        for model in MODEL_ORDER:
            vals = per_pair[pair][key].get(model)
            if vals is None:
                continue
            mean_data.setdefault(model, []).append(vals["bootstrap_mean"])
            std_data.setdefault(model, []).append(vals["bootstrap_std"])

    pair_labels = [PAIR_DISPLAY.get(p, p) for p in PAIR_ORDER if p in per_pair]
    model_labels = [MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER]

    df_mean = pd.DataFrame(mean_data, index=pair_labels)
    df_mean.columns = model_labels
    df_std = pd.DataFrame(std_data, index=pair_labels)
    df_std.columns = model_labels
    return df_mean, df_std


def _plot_heatmap(df, title, cmap, out_path, fmt=".4f", center=None,
                  vmin=None, vmax=None, annot_kw=None):
    """Generic annotated heatmap."""
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(df) + 1.6))

    norm = None
    if center is not None:
        lo = vmin if vmin is not None else df.min().min()
        hi = vmax if vmax is not None else df.max().max()
        if lo < center < hi:
            norm = TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)

    data = df.values
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                   vmin=vmin, vmax=vmax)

    # Annotate cells
    akw = annot_kw or {}
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:{fmt}}", ha="center", va="center",
                    fontsize=akw.get("fontsize", 10),
                    color=akw.get("color", "black"))

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=10)
    ax.set_title(title, fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_grouped_bar(df_mean, df_std, metric, out_path):
    """Grouped bar chart: bootstrap mean ± std per pair, grouped by model."""
    n_pairs = len(df_mean)
    n_models = len(df_mean.columns)
    x = np.arange(n_pairs)
    bar_w = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [MODEL_COLORS.get(m, "#555") for m in MODEL_ORDER]

    for j, (col, color) in enumerate(zip(df_mean.columns, colors)):
        offset = (j - n_models / 2 + 0.5) * bar_w
        ax.bar(x + offset, df_mean[col], bar_w, yerr=df_std[col],
               color=color, edgecolor="white", linewidth=0.5,
               capsize=2, label=col, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(df_mean.index, rotation=25, ha="right", fontsize=10)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"Bootstrap Mean {metric}", fontsize=11)
    ax.set_title(f"{metric} — Bootstrap Mean ± Std  (per pair × model)",
                 fontsize=13)
    ax.legend(fontsize=9, ncol=n_models, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_pair = data["per_pair"]

    for metric in ("D", "C12"):
        df_mean, df_std = _build_tables(per_pair, metric)
        corr = df_mean.corr(method="spearman")

        # ── Console output ───────────────────────────────────────────────
        print(f"\n{'='*80}")
        print(f"  {metric} — Bootstrap Mean (per pair × model)")
        print(f"{'='*80}")
        print(df_mean.to_string(float_format="{:.4f}".format))

        print(f"\n{'-'*80}")
        print(f"  {metric} — Bootstrap Std (per pair × model)")
        print(f"{'-'*80}")
        print(df_std.to_string(float_format="{:.4f}".format))

        print(f"\n{'-'*80}")
        print(f"  {metric} — Spearman Rank Correlation between Models")
        print(f"  (based on bootstrap_mean across {len(df_mean)} pairs)")
        print(f"{'-'*80}")
        print(corr.to_string(float_format="{:.4f}".format))

        # ── Plots ────────────────────────────────────────────────────────
        # 1. Heatmap of bootstrap mean
        _plot_heatmap(
            df_mean,
            title=f"{metric} — Bootstrap Mean (per pair × model)",
            cmap="RdBu_r", center=0.0,
            out_path=OUTPUT_DIR / f"model_corr_{metric}_mean_heatmap.png",
        )

        # 2. Grouped bar chart (mean ± std)
        _plot_grouped_bar(
            df_mean, df_std, metric,
            out_path=OUTPUT_DIR / f"model_corr_{metric}_bar.png",
        )

        # 3. Spearman correlation heatmap
        _plot_heatmap(
            corr,
            title=f"{metric} — Spearman Correlation between Models\n"
                  f"(across {len(df_mean)} modality pairs)",
            cmap="YlOrRd", vmin=0.5, vmax=1.0,
            out_path=OUTPUT_DIR / f"model_corr_{metric}_spearman_heatmap.png",
            fmt=".3f",
            annot_kw={"fontsize": 11},
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
