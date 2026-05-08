#!/usr/bin/env python3
"""
Line plots for SHAPE scores (D and C12) by question intent × modality pair.

Produces figures in two subfolders:
  - no_min_filter/       (all intent groups)
  - samples_gt_10/       (only groups with n > 10)

Configuration:
  Set PLOT_MODE at the top to control what is drawn:
    "per_model"   – one line per model
    "aggregated"  – single aggregated line (mean across models) with min/max band
    "both"        – per-model lines + bold aggregated mean line
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────
# Options: "per_model", "aggregated", "both"
PLOT_MODE = "aggregated"

RESULTS_JSON = Path(__file__).parent / "intent_shape_results.json"
OUTPUT_BASE = Path(__file__).parent / "intent_line_figures"

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
    "gemma3_4b":    "#4A7DC5",
    "gemma3_27b":   "#6AB08A",
    "gpt-4o-mini":  "#D4726A",
    "qwen3-vl_8b":  "#E8A854",
    "qwen3-vl_30b": "#9B8ABF",
}
MODEL_MARKERS = {
    "gemma3_4b":    "o",
    "gemma3_27b":   "s",
    "gpt-4o-mini":  "D",
    "qwen3-vl_8b":  "^",
    "qwen3-vl_30b": "v",
}

PAIR_ORDER = [
    "image + layout", "image + plain_text", "layout + plain_text",
    "plain_text + table", "image + table", "layout + table",
]
PAIR_DISPLAY = {
    "image + layout":      "Image + Layout",
    "image + plain_text":  "Image + Text",
    "layout + plain_text": "Layout + Text",
    "plain_text + table":  "Text + Table",
    "image + table":       "Image + Table",
    "layout + table":      "Layout + Table",
}
MOD_FULL = {
    "image": "Image", "layout": "Layout", "plain_text": "Text", "table": "Table",
}
MOD_ABBREV = {
    "image": "I", "layout": "L", "plain_text": "T", "table": "Ta",
}


def load_data():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_intents(pair_data, min_n):
    """Return sorted intent names that pass the min_n filter."""
    groups = pair_data.get("intent_groups", {})
    return sorted(
        intent for intent, info in groups.items()
        if info["n_questions"] > min_n
    )


def get_subplot_title(pair_label, pair_data, metric):
    comp = pair_data["comparison"]
    disp = PAIR_DISPLAY.get(pair_label, pair_label)
    if metric == "D":
        t_full = MOD_FULL.get(comp["target"], comp["target"])
        r_full = MOD_FULL.get(comp["reference"], comp["reference"])
        return (f"{disp}  "
                rf"($D_{{\mathrm{{{t_full}}},\mathrm{{{r_full}}}}} "
                rf"= S_{{\mathrm{{{t_full}}}}} - S_{{\mathrm{{{r_full}}}}}$)")
    return disp


def plot_metric(results, metric, min_n, output_dir):
    """Create a multi-panel line figure for a given metric."""
    # Filter pairs that have data after applying min_n
    valid_pairs = []
    for pl in PAIR_ORDER:
        if pl not in results:
            continue
        intents = filter_intents(results[pl], min_n)
        if intents:
            valid_pairs.append(pl)

    n_panels = len(valid_pairs)
    if n_panels == 0:
        print(f"  No data for {metric} with min_n={min_n} — skipping")
        return

    ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.6 * ncols, 4.2 * nrows),
        squeeze=False,
    )

    for pi, pl in enumerate(valid_pairs):
        row, col = divmod(pi, ncols)
        ax = axes[row][col]
        pair_data = results[pl]
        intents = filter_intents(pair_data, min_n)
        groups = pair_data["intent_groups"]

        x = np.arange(len(intents))
        x_labels = [
            f"{intent}\n(n={groups[intent]['n_questions']})"
            for intent in intents
        ]

        if PLOT_MODE in ("per_model", "both"):
            for model in MODEL_ORDER:
                values = []
                for intent in intents:
                    obs = groups[intent]["observed"].get(model)
                    if obs is None:
                        values.append(np.nan)
                    else:
                        v = obs.get(metric)
                        values.append(v if v is not None else np.nan)
                ax.plot(
                    x, values,
                    marker=MODEL_MARKERS[model],
                    color=MODEL_COLORS[model],
                    label=MODEL_DISPLAY[model],
                    linewidth=1.4,
                    markersize=6,
                    alpha=0.85 if PLOT_MODE == "both" else 1.0,
                )

        if PLOT_MODE in ("aggregated", "both"):
            # Compute mean/min/max across models per intent
            means, mins, maxs = [], [], []
            ci_los, ci_his = [], []
            for intent in intents:
                vals = []
                for model in MODEL_ORDER:
                    obs = groups[intent]["observed"].get(model)
                    if obs is not None:
                        v = obs.get(metric)
                        if v is not None:
                            vals.append(v)
                if vals:
                    means.append(np.mean(vals))
                    mins.append(np.min(vals))
                    maxs.append(np.max(vals))
                else:
                    means.append(np.nan)
                    mins.append(np.nan)
                    maxs.append(np.nan)

                # Aggregated bootstrap CI (stored per intent group)
                lo = groups[intent].get(f"agg_{metric}_ci_lo")
                hi = groups[intent].get(f"agg_{metric}_ci_hi")
                ci_los.append(lo if lo is not None else np.nan)
                ci_his.append(hi if hi is not None else np.nan)

            means = np.array(means)
            mins = np.array(mins)
            maxs = np.array(maxs)
            ci_los = np.array(ci_los, dtype=float)
            ci_his = np.array(ci_his, dtype=float)

            ax.fill_between(x, mins, maxs, alpha=0.18, color="#9DC4E8",
                            label="Model range (min\u2013max)", zorder=2)
            if not np.all(np.isnan(ci_los)):
                ax.fill_between(x, ci_los, ci_his, alpha=0.30, color="#2A5FB8",
                                label="95% bootstrap CI", zorder=3)
            ax.plot(
                x, means,
                marker="o", color="#1A3A6B",
                linewidth=2.5, markersize=8,
                label="Cross-model mean",
                zorder=5,
            )

        ax.axhline(0, color="0.55", linestyle="--", linewidth=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_title(get_subplot_title(pl, pair_data, metric), fontsize=17)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        if col == 0:
            if metric == "D":
                ax.set_ylabel(r"$D_{S_{m_2},\,S_{m_1}}$", fontsize=18)
            else:
                ax.set_ylabel(r"$C_{\{m_1,m_2\}}$", fontsize=18)

    # Hide unused subplots
    for pi in range(n_panels, nrows * ncols):
        row, col = divmod(pi, ncols)
        axes[row][col].set_visible(False)

    # Shared legend at the bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(handles), 6),
        frameon=False,
        fontsize=15,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    fname = f"intent_{metric}_lines"
    fig.savefig(output_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / fname}.png/.pdf")


def run_for_filter(results, min_n, folder_name):
    output_dir = OUTPUT_BASE / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"Generating figures → {output_dir}  (min_n={min_n})")
    print(f"{'='*50}")
    plot_metric(results, "D", min_n, output_dir)
    plot_metric(results, "C12", min_n, output_dir)


def main():
    results = load_data()
    run_for_filter(results, min_n=0, folder_name="no_min_filter")
    run_for_filter(results, min_n=9, folder_name="samples_gte_10")
    run_for_filter(results, min_n=20, folder_name="samples_gt_20")
    print("\nDone.")


if __name__ == "__main__":
    main()
