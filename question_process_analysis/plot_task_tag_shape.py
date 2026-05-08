#!/usr/bin/env python3
"""
Line-plot figures for SHAPE scores by task tag × modality group (LongDocURL).

Produces figures in subfolders under task_tag_line_figures/:
  - no_min_filter/       (all task-tag groups)
  - samples_gt_10/       (only groups with n > 10)
  - samples_gt_20/       (only groups with n > 20)

Configuration:
  Set PLOT_MODE at the top to control what is drawn:
    "per_model"   – one line per model
    "aggregated"  – single aggregated line (mean across models) with min/max band
    "both"        – per-model lines + bold aggregated mean line
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Configuration ─────────────────────────────────────────────────────
# Options: "per_model", "aggregated", "both"
PLOT_MODE = "aggregated"

RESULTS_JSON = Path(__file__).parent / "task_tag_shape_results.json"
OUTPUT_BASE  = Path(__file__).parent / "task_tag_line_figures"

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
    "qwen3-vl_30b": "#9B7BB8",
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

TAG_ORDER = ["Locating", "Reasoning", "Understanding"]


def filter_tags(pair_data, min_n):
    """Return sorted tag names that pass the min_n filter."""
    groups = pair_data.get("task_tag_groups", {})
    return [t for t in TAG_ORDER
            if t in groups and groups[t]["n_questions"] > min_n]


def plot_metric(results, metric, y_label, output_stem, min_n, output_dir):
    """
    Multi-panel line-plot figure: one subplot per modality pair.
    X-axis = task tags, lines = models.
    """
    valid_pairs = []
    for pl in PAIR_ORDER:
        if pl not in results:
            continue
        tags = filter_tags(results[pl], min_n)
        if tags:
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

        pd = results[pl]
        comp = pd["comparison"]
        groups = pd["task_tag_groups"]

        tags = filter_tags(pd, min_n)
        x_pos = np.arange(len(tags))
        x_labels = [f"{t}\n(n={groups[t]['n_questions']})" for t in tags]

        if PLOT_MODE in ("per_model", "both"):
            for model in MODEL_ORDER:
                values = []
                for tag in tags:
                    obs = groups[tag]["observed"].get(model)
                    if obs is None:
                        values.append(np.nan)
                    else:
                        v = obs.get(metric)
                        values.append(v if v is not None else np.nan)

                ax.plot(
                    x_pos, values,
                    marker=MODEL_MARKERS[model],
                    color=MODEL_COLORS[model],
                    label=MODEL_DISPLAY[model],
                    linewidth=1.8,
                    markersize=7,
                    alpha=0.85 if PLOT_MODE == "both" else 1.0,
                    zorder=3,
                )

        if PLOT_MODE in ("aggregated", "both"):
            means, mins, maxs = [], [], []
            ci_los, ci_his = [], []
            for tag in tags:
                vals = []
                for model in MODEL_ORDER:
                    obs = groups[tag]["observed"].get(model)
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

                # Aggregated bootstrap CI (stored per tag group)
                lo = groups[tag].get(f"agg_{metric}_ci_lo")
                hi = groups[tag].get(f"agg_{metric}_ci_hi")
                ci_los.append(lo if lo is not None else np.nan)
                ci_his.append(hi if hi is not None else np.nan)

            means = np.array(means)
            mins = np.array(mins)
            maxs = np.array(maxs)
            ci_los = np.array(ci_los, dtype=float)
            ci_his = np.array(ci_his, dtype=float)

            ax.fill_between(x_pos, mins, maxs, alpha=0.25, color="#F5BD8A",
                            label="Model range (min–max)", zorder=2)
            if not np.all(np.isnan(ci_los)):
                ax.fill_between(x_pos, ci_los, ci_his, alpha=0.30, color="#D93636",
                                label="95% bootstrap CI", zorder=3)
            ax.plot(
                x_pos, means,
                marker="o", color="#5B1A1A",
                linewidth=2.5, markersize=8,
                label="Cross-model mean",
                zorder=5,
            )

        ax.axhline(0, color="0.6", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        if metric == "D":
            t_full = MOD_FULL.get(comp["target"], comp["target"])
            r_full = MOD_FULL.get(comp["reference"], comp["reference"])
            sub = (f"{PAIR_DISPLAY.get(pl, pl)}  "
                   rf"($D_{{\mathrm{{{t_full}}},\mathrm{{{r_full}}}}} "
                   rf"= S_{{\mathrm{{{t_full}}}}} - S_{{\mathrm{{{r_full}}}}}$)")
        else:
            sub = PAIR_DISPLAY.get(pl, pl)
        ax.set_title(sub, fontsize=17, pad=8)

        if col == 0:
            ax.set_ylabel(y_label, fontsize=18)

    # Hide empty subplots
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

    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = output_dir / f"{output_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_contributions(results, output_stem, min_n, output_dir):
    """
    Multi-panel line-plot: one subplot per modality pair.
    For each model, two lines: S_m1 (solid) and S_m2 (dashed).
    """
    valid_pairs = []
    for pl in PAIR_ORDER:
        if pl not in results:
            continue
        tags = filter_tags(results[pl], min_n)
        if tags:
            valid_pairs.append(pl)

    n_panels = len(valid_pairs)
    if n_panels == 0:
        print(f"  No data for contributions with min_n={min_n} — skipping")
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

        pd = results[pl]
        m1_name = pd["m1_name"]
        m2_name = pd["m2_name"]
        
        groups = pd["task_tag_groups"]

        tags = filter_tags(pd, min_n)
        x_pos = np.arange(len(tags))
        x_labels = [f"{t}\n(n={groups[t]['n_questions']})" for t in tags]

        for model in MODEL_ORDER:
            vals_m1, vals_m2 = [], []
            for tag in tags:
                obs = groups[tag]["observed"].get(model)
                if obs is None:
                    vals_m1.append(np.nan)
                    vals_m2.append(np.nan)
                else:
                    v1 = obs.get("S_m1")
                    v2 = obs.get("S_m2")
                    vals_m1.append(v1 if v1 is not None else np.nan)
                    vals_m2.append(v2 if v2 is not None else np.nan)

            ax.plot(
                x_pos, vals_m1,
                marker=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                label=MODEL_DISPLAY[model] if pi == 0 else None,
                linewidth=1.8, markersize=7, zorder=3,
                linestyle="-",
            )
            ax.plot(
                x_pos, vals_m2,
                marker=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                label=None,
                linewidth=1.8, markersize=7, zorder=3,
                linestyle="--",
            )

        ax.axhline(0, color="0.6", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        m1_abbr = MOD_ABBREV.get(m1_name, m1_name)
        m2_abbr = MOD_ABBREV.get(m2_name, m2_name)
        ax.set_title(
            f"{PAIR_DISPLAY.get(pl, pl)}  "
            rf"(solid = $S_{{\mathrm{{{m1_abbr}}}}}$, dashed = $S_{{\mathrm{{{m2_abbr}}}}}$)",
            fontsize=17, pad=8,
        )

        if col == 0:
            ax.set_ylabel(r"$S$ (contribution)", fontsize=18)

    # Hide empty subplots
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

    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = output_dir / f"{output_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def run_for_filter(results, min_n, folder_name):
    output_dir = OUTPUT_BASE / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"Generating figures → {output_dir}  (min_n={min_n})")
    print(f"{'='*50}")

    plot_metric(
        results,
        metric="D",
        y_label=r"$D_{S_{m_2},\,S_{m_1}}$",
        output_stem="task_tag_D_by_pair",
        min_n=min_n,
        output_dir=output_dir,
    )

    plot_metric(
        results,
        metric="C12",
        y_label=r"$C_{\{m_1,m_2\}}$",
        output_stem="task_tag_C12_by_pair",
        min_n=min_n,
        output_dir=output_dir,
    )

    plot_contributions(
        results,
        output_stem="task_tag_S_by_pair",
        min_n=min_n,
        output_dir=output_dir,
    )


def main():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)

    run_for_filter(results, min_n=0, folder_name="no_min_filter")
    run_for_filter(results, min_n=9, folder_name="samples_gte_10")
    run_for_filter(results, min_n=20, folder_name="samples_gt_20")
    print("\nDone.")


if __name__ == "__main__":
    main()
