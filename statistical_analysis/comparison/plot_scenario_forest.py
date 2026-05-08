#!/usr/bin/env python3
"""
Forest-style comparison plots across the three bootstrap scenarios.

Produces two compact figures (D + C12 side-by-side):
  1. Cross-model mean + CI for each scenario, overlaid per modality pair
  2. CI width comparison across scenarios per modality pair (per-model detail)
"""

import json
from pathlib import Path
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Paths ─────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent

SCENARIO_JSONS = OrderedDict([
    ("Strat-Pair",  BASE / "stratified" / "stratified_results.json"),
    ("Strat-Bench", BASE / "stratified_by_benchmark" / "stratified_by_benchmark_results.json"),
    ("Unstratified", BASE / "unstratified" / "unstratified_results.json"),
])

SCENARIO_COLORS = {
    "Strat-Pair":  "#4C72B0",
    "Strat-Bench": "#DD8452",
    "Unstratified": "#55A868",
}
SCENARIO_MARKERS = {
    "Strat-Pair":  "o",
    "Strat-Bench": "s",
    "Unstratified": "^",
}

# ── Shared config (same as individual plots) ──────────────────────────
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
MOD_FULL = {
    "image": "Image", "layout": "Layout", "plain_text": "Text", "table": "Table",
}
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
MODEL_MARKERS = {
    "gemma3_4b":    "o",
    "gemma3_27b":   "s",
    "gpt-4o-mini":  "^",
    "qwen3-vl_8b":  "v",
    "qwen3-vl_30b": "P",
}


# ── Helpers ───────────────────────────────────────────────────────────
def _clamp(lo, hi, obs):
    return max(0.0, obs - lo), max(0.0, hi - obs)


def load_all():
    """Load all three scenario JSONs → {scenario_label: per_pair_dict}."""
    data = OrderedDict()
    for label, path in SCENARIO_JSONS.items():
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data[label] = raw.get("per_pair", raw)
    return data


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1 — Cross-model mean comparison (D left, C12 right)
# ═══════════════════════════════════════════════════════════════════════

def plot_crossmodel_comparison(all_data, output_path):
    """
    Two-panel (D | C12). For each modality pair, show the cross-model mean
    diamond + CI for each of the 3 scenarios, offset vertically.
    """
    pair_labels = [p for p in PAIR_ORDER
                   if all(p in all_data[s] for s in all_data)]
    n_pairs = len(pair_labels)
    if n_pairs == 0:
        return

    n_sc = len(all_data)
    step = 0.14
    offsets = [(i - (n_sc - 1) / 2) * step for i in range(n_sc)]
    row_height = n_sc * step + 0.30

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, row_height * n_pairs + 2.0),
        sharey=True,
    )
    fig.patch.set_facecolor("white")

    for ax_i, metric in enumerate(("D", "C12")):
        ax = axes[ax_i]
        ax.set_facecolor("white")
        is_left = (ax_i == 0)
        cross_key = f"cross_model_{metric}"

        # alternating row shading
        for pi in range(n_pairs):
            yc = pi * row_height
            if pi % 2 == 1:
                ax.axhspan(yc - row_height / 2, yc + row_height / 2,
                           color="#F5F5F5", zorder=0)

        # per-scenario cross-model mean
        for si, (slabel, pairs) in enumerate(all_data.items()):
            for pi, pl in enumerate(pair_labels):
                cross = pairs[pl].get(cross_key, {})
                if cross.get("n_valid_replicates", 0) == 0:
                    continue
                ma = cross["mean_across_models"]
                obs = ma["mean"]
                lo_e, hi_e = _clamp(ma["ci_lo"], ma["ci_hi"], obs)
                excl = ma["ci_excludes_zero"]
                fc = SCENARIO_COLORS[slabel] if excl else "white"
                ax.errorbar(
                    obs, pi * row_height + offsets[si],
                    xerr=[[lo_e], [hi_e]],
                    fmt=SCENARIO_MARKERS[slabel],
                    color=SCENARIO_COLORS[slabel],
                    markerfacecolor=fc,
                    markeredgecolor=SCENARIO_COLORS[slabel],
                    markeredgewidth=0.8,
                    capsize=3, markersize=8, linewidth=1.8, zorder=3,
                    label=slabel if (is_left and pi == 0) else None,
                )

        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.35, zorder=1)

        xlabel = ("D (contribution difference)" if metric == "D"
                  else r"$C_{12}$ (cooperation)")
        ax.set_xlabel(xlabel, fontsize=13)
        met_label = "Contribution" if metric == "D" else "Cooperation"
        ax.set_title(met_label, fontsize=14, fontweight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.grid(axis="x", alpha=0.12)
        ax.tick_params(axis="x", labelsize=11)

    # shared y-axis labels
    y_labels = []
    first_pairs = list(all_data.values())[0]
    for pl in pair_labels:
        disp = PAIR_DISPLAY.get(pl, pl)
        comp = first_pairs[pl]["comparison"]
        t_full = MOD_FULL.get(comp["target"], comp["target"])
        r_full = MOD_FULL.get(comp["reference"], comp["reference"])
        y_labels.append(f"{disp}\nD = S({t_full}) \u2212 S({r_full})")

    y_positions = [pi * row_height for pi in range(n_pairs)]
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=11, rotation=15, ha="right")
    axes[0].set_ylim(y_positions[-1] + row_height / 2,
                     y_positions[0] - row_height / 2)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=n_sc,
        fontsize=11, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle("Cross-Model Mean: Scenario Comparison",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2 — Per-model comparison (D left, C12 right)
#          3 scenarios overlaid per modality pair, per model
# ═══════════════════════════════════════════════════════════════════════

def plot_permodel_comparison(all_data, output_path):
    """
    Two-panel (D | C12). For each modality pair show per-model observed
    values + CIs from each scenario, grouped by model with scenario offsets.
    """
    pair_labels = [p for p in PAIR_ORDER
                   if all(p in all_data[s] for s in all_data)]
    n_pairs = len(pair_labels)
    if n_pairs == 0:
        return

    n_m = len(MODEL_ORDER)
    n_sc = len(all_data)
    # Within each model band: n_sc scenario lines
    sc_step = 0.06
    model_step = n_sc * sc_step + 0.08
    row_height = n_m * model_step + 0.25

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, row_height * n_pairs + 2.2),
        sharey=True,
    )
    fig.patch.set_facecolor("white")

    scenario_labels = list(all_data.keys())

    for ax_i, metric in enumerate(("D", "C12")):
        ax = axes[ax_i]
        ax.set_facecolor("white")
        per_key = f"per_model_{metric}"
        obs_key = f"observed_{metric}"

        # alternating row shading
        for pi in range(n_pairs):
            yc = pi * row_height
            if pi % 2 == 1:
                ax.axhspan(yc - row_height / 2, yc + row_height / 2,
                           color="#F5F5F5", zorder=0)

        for mi, model in enumerate(MODEL_ORDER):
            for si, slabel in enumerate(scenario_labels):
                pairs = all_data[slabel]
                xs, ys, elo, ehi = [], [], [], []
                for pi, pl in enumerate(pair_labels):
                    pm = pairs[pl].get(per_key, {})
                    if model not in pm:
                        continue
                    r = pm[model]
                    obs = r[obs_key]
                    lo_e, hi_e = _clamp(r["ci_lo"], r["ci_hi"], obs)
                    # offset: model band center + scenario sub-offset
                    model_center = mi * model_step - (n_m - 1) * model_step / 2
                    sc_off = (si - (n_sc - 1) / 2) * sc_step
                    y = pi * row_height + model_center + sc_off
                    xs.append(obs)
                    ys.append(y)
                    elo.append(lo_e)
                    ehi.append(hi_e)

                if xs:
                    ax.errorbar(
                        xs, ys, xerr=[elo, ehi],
                        fmt=SCENARIO_MARKERS[slabel],
                        color=MODEL_COLORS.get(model, "#999"),
                        capsize=1.5, markersize=4, linewidth=0.8,
                        alpha=0.7 + 0.15 * (si == 0),
                        zorder=3,
                    )

        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.35, zorder=1)

        xlabel = ("D (contribution difference)" if metric == "D"
                  else r"$C_{12}$ (cooperation)")
        ax.set_xlabel(xlabel, fontsize=12)
        met_label = "Contribution" if metric == "D" else "Cooperation"
        ax.set_title(met_label, fontsize=14, fontweight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.grid(axis="x", alpha=0.12)
        ax.tick_params(axis="x", labelsize=10)

    # y-axis labels
    y_labels = []
    first_pairs = list(all_data.values())[0]
    for pl in pair_labels:
        disp = PAIR_DISPLAY.get(pl, pl)
        comp = first_pairs[pl]["comparison"]
        t_full = MOD_FULL.get(comp["target"], comp["target"])
        r_full = MOD_FULL.get(comp["reference"], comp["reference"])
        y_labels.append(f"{disp}\nD = S({t_full}) \u2212 S({r_full})")

    y_positions = [pi * row_height for pi in range(n_pairs)]
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=10, rotation=15, ha="right")
    axes[0].set_ylim(y_positions[-1] + row_height / 2,
                     y_positions[0] - row_height / 2)

    # Manual legend: models (by colour) + scenarios (by marker)
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], marker="o", color=MODEL_COLORS[m], linestyle="",
               markersize=6, label=MODEL_DISPLAY[m])
        for m in MODEL_ORDER
    ]
    sc_handles = [
        Line2D([0], [0], marker=SCENARIO_MARKERS[s], color="gray",
               linestyle="", markersize=6, label=s)
        for s in scenario_labels
    ]
    all_handles = model_handles + sc_handles
    fig.legend(
        handles=all_handles,
        loc="lower center", ncol=4,
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle("Per-Model Observed + CI: Scenario Comparison",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 3 — CI Width grouped dots (D left, C12 right)
#          Per model × pair, 3 scenario dots connected
# ═══════════════════════════════════════════════════════════════════════

def plot_ci_width_forest(all_data, output_path):
    """
    Two-panel (D | C12). For each modality pair, show per-model CI widths
    as connected dots across scenarios.
    """
    pair_labels = [p for p in PAIR_ORDER
                   if all(p in all_data[s] for s in all_data)]
    n_pairs = len(pair_labels)
    if n_pairs == 0:
        return

    n_m = len(MODEL_ORDER)
    n_sc = len(all_data)
    model_step = 0.12
    row_height = n_m * model_step + 0.30

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, row_height * n_pairs + 2.0),
        sharey=True,
    )
    fig.patch.set_facecolor("white")

    scenario_labels = list(all_data.keys())

    for ax_i, metric in enumerate(("D", "C12")):
        ax = axes[ax_i]
        ax.set_facecolor("white")
        per_key = f"per_model_{metric}"

        # alternating row shading
        for pi in range(n_pairs):
            yc = pi * row_height
            if pi % 2 == 1:
                ax.axhspan(yc - row_height / 2, yc + row_height / 2,
                           color="#F5F5F5", zorder=0)

        for mi, model in enumerate(MODEL_ORDER):
            model_off = (mi - (n_m - 1) / 2) * model_step
            for pi, pl in enumerate(pair_labels):
                y = pi * row_height + model_off
                widths = []
                for si, slabel in enumerate(scenario_labels):
                    pm = all_data[slabel][pl].get(per_key, {})
                    if model not in pm:
                        continue
                    r = pm[model]
                    w = r["ci_hi"] - r["ci_lo"]
                    widths.append(w)
                    ax.plot(
                        w, y,
                        marker=SCENARIO_MARKERS[slabel],
                        color=MODEL_COLORS.get(model, "#999"),
                        markersize=5, zorder=3, alpha=0.85,
                    )
                # connect the scenario dots with a thin line
                if len(widths) == n_sc:
                    ax.plot(widths, [y] * n_sc,
                            color=MODEL_COLORS.get(model, "#999"),
                            linewidth=0.7, alpha=0.4, zorder=2)

        xlabel = ("D — CI width" if metric == "D"
                  else r"$C_{12}$ — CI width")
        ax.set_xlabel(xlabel, fontsize=12)
        met_label = "Contribution" if metric == "D" else "Cooperation"
        ax.set_title(met_label, fontsize=14, fontweight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.grid(axis="x", alpha=0.12)
        ax.tick_params(axis="x", labelsize=10)

    # y-axis labels
    y_labels = [PAIR_DISPLAY.get(pl, pl) for pl in pair_labels]
    y_positions = [pi * row_height for pi in range(n_pairs)]
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=11)
    axes[0].set_ylim(y_positions[-1] + row_height / 2,
                     y_positions[0] - row_height / 2)

    # Manual legend
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], marker="o", color=MODEL_COLORS[m], linestyle="",
               markersize=6, label=MODEL_DISPLAY[m])
        for m in MODEL_ORDER
    ]
    sc_handles = [
        Line2D([0], [0], marker=SCENARIO_MARKERS[s], color="gray",
               linestyle="", markersize=6, label=s)
        for s in scenario_labels
    ]
    all_handles = model_handles + sc_handles
    fig.legend(
        handles=all_handles,
        loc="lower center", ncol=4,
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle("CI Width Comparison Across Scenarios",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    all_data = load_all()

    plot_crossmodel_comparison(
        all_data, OUTPUT_DIR / "scenario_crossmodel_forest.png")

    plot_permodel_comparison(
        all_data, OUTPUT_DIR / "scenario_permodel_forest.png")

    plot_ci_width_forest(
        all_data, OUTPUT_DIR / "scenario_ci_width_forest.png")

    print("Done.")


if __name__ == "__main__":
    main()
