#!/usr/bin/env python3
"""
Compare Bootstrap Scenarios — Visual Differences
=================================================

Produces plots that directly highlight how the three bootstrap
stratification scenarios differ:

1. Stratified by modality pair  (pair ratios fixed)
2. Stratified by pair × benchmark  (pair & benchmark ratios fixed)
3. Unstratified / global pool  (pair ratios free to fluctuate)

Plots:
  A) Grouped bar chart: CI width per metric × pair × model
  B) Dot-plot of CI midpoint shifts between scenarios
  C) Heatmap: relative CI width change (%) between pairs of scenarios
  D) Summary bar: mean CI width across all pair×model combos
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

# ── paths ────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
CSVS = OrderedDict([
    ("Stratified\n(by pair)",
     BASE / "stratified" / "stratified_per_model.csv"),
    ("Stratified\n(pair × bench)",
     BASE / "stratified_by_benchmark" / "stratified_by_benchmark_per_model.csv"),
    ("Unstratified\n(global pool)",
     BASE / "unstratified" / "unstratified_per_model.csv"),
])
SHORT_LABELS = ["Strat-Pair", "Strat-Bench", "Unstrat-Pool"]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]
OUT = Path(__file__).parent


# ── load CSVs ────────────────────────────────────────────────────────
def load_csv(path):
    """Return list of dicts, normalising column names."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def build_table(rows):
    """Return dict  (pair, model) → row-dict  with float conversions."""
    table = {}
    for r in rows:
        key = (r["modality_pair"], r["model"])
        for c in ("D_ci_lo", "D_ci_hi", "D_std", "D_mean",
                  "C12_ci_lo", "C12_ci_hi", "C12_std", "C12_mean",
                  "observed_D", "observed_C12"):
            if c in r:
                r[c] = float(r[c])
        table[key] = r
    return table


def ci_width(row, metric):
    return row[f"{metric}_ci_hi"] - row[f"{metric}_ci_lo"]


def ci_mid(row, metric):
    return (row[f"{metric}_ci_hi"] + row[f"{metric}_ci_lo"]) / 2.0


# ── load everything ──────────────────────────────────────────────────
tables = OrderedDict()
for label, path in CSVS.items():
    tables[label] = build_table(load_csv(path))

scenario_labels = list(tables.keys())
# common keys across all three
common_keys = sorted(
    set.intersection(*(set(t.keys()) for t in tables.values())),
    key=lambda k: (k[0], k[1]),
)
pairs_ordered = list(dict.fromkeys(k[0] for k in common_keys))
models_ordered = list(dict.fromkeys(k[1] for k in common_keys))
n_pairs = len(pairs_ordered)
n_models = len(models_ordered)
n_scenarios = len(scenario_labels)


# ═════════════════════════════════════════════════════════════════════
# PLOT A — Grouped bar chart: CI width per pair (averaged over models)
# ═════════════════════════════════════════════════════════════════════

def plot_ci_width_bars(metric, tag):
    """One bar chart for D or C12."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_pairs)
    width = 0.25

    for si, (slabel, tbl) in enumerate(tables.items()):
        means = []
        for p in pairs_ordered:
            vals = [ci_width(tbl[(p, m)], metric) for m in models_ordered
                    if (p, m) in tbl]
            means.append(np.mean(vals))
        ax.bar(x + si * width, means, width, label=SHORT_LABELS[si],
               color=COLORS[si], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(pairs_ordered, fontsize=9)
    ax.set_ylabel(f"{tag} 95 % CI width (mean over models)", fontsize=10)
    ax.set_title(f"{tag}: CI Width by Modality Pair — Three Scenarios",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / f"ci_width_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════
# PLOT B — Per-model CI width difference (unstratified – stratified)
# ═════════════════════════════════════════════════════════════════════

def plot_ci_width_diff_per_model(metric, tag):
    """Horizontal dot-plot: CI_width(unstrat) − CI_width(strat_pair)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    y_labels = []
    diffs_bench = []
    diffs_unstrat = []
    for p in pairs_ordered:
        for m in models_ordered:
            key = (p, m)
            if key not in tables[scenario_labels[0]]:
                continue
            w_strat = ci_width(tables[scenario_labels[0]][key], metric)
            w_bench = ci_width(tables[scenario_labels[1]][key], metric)
            w_unstr = ci_width(tables[scenario_labels[2]][key], metric)
            diffs_bench.append(w_bench - w_strat)
            diffs_unstrat.append(w_unstr - w_strat)
            y_labels.append(f"{p}  |  {m}")

    y = np.arange(len(y_labels))
    ax.scatter(diffs_bench, y, marker="D", s=30, color=COLORS[1],
               label=f"{SHORT_LABELS[1]} − {SHORT_LABELS[0]}", zorder=3)
    ax.scatter(diffs_unstrat, y, marker="o", s=30, color=COLORS[2],
               label=f"{SHORT_LABELS[2]} − {SHORT_LABELS[0]}", zorder=3)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel(f"Δ CI width ({tag})", fontsize=10)
    ax.set_title(f"{tag}: CI Width Difference vs Stratified-by-Pair",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    out = OUT / f"ci_width_diff_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════
# PLOT C — Heatmap: relative CI width change (%) pair × model
# ═════════════════════════════════════════════════════════════════════

def plot_heatmap_relative(metric, tag):
    """Heatmaps of (unstrat − strat)/strat × 100 and (bench − strat)/strat × 100."""
    comparisons = [
        (SHORT_LABELS[2], scenario_labels[2], scenario_labels[0]),
        (SHORT_LABELS[1], scenario_labels[1], scenario_labels[0]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (comp_label, s_alt, s_ref) in zip(axes, comparisons):
        mat = np.full((n_models, n_pairs), np.nan)
        for pi, p in enumerate(pairs_ordered):
            for mi, m in enumerate(models_ordered):
                key = (p, m)
                if key not in tables[s_ref] or key not in tables[s_alt]:
                    continue
                w_ref = ci_width(tables[s_ref][key], metric)
                w_alt = ci_width(tables[s_alt][key], metric)
                if abs(w_ref) > 1e-12:
                    mat[mi, pi] = 100.0 * (w_alt - w_ref) / w_ref

        vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 1)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(n_pairs))
        ax.set_xticklabels([p.replace(" + ", "\n+\n") for p in pairs_ordered],
                           fontsize=8)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(models_ordered, fontsize=9)
        ax.set_title(f"{comp_label} vs {SHORT_LABELS[0]}\n(% change in {tag} CI width)",
                     fontsize=10, fontweight="bold")

        # annotate cells
        for mi in range(n_models):
            for pi in range(n_pairs):
                val = mat[mi, pi]
                if not np.isnan(val):
                    ax.text(pi, mi, f"{val:+.1f}%", ha="center", va="center",
                            fontsize=7,
                            color="white" if abs(val) > vmax * 0.6 else "black")

        fig.colorbar(im, ax=ax, shrink=0.8, label="% change")

    fig.suptitle(f"{tag}: Relative CI Width Change (%) by Pair × Model",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = OUT / f"ci_width_pct_heatmap_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════
# PLOT D — Summary: mean CI width & std across all pair×model combos
# ═════════════════════════════════════════════════════════════════════

def plot_summary_bars():
    """One summary figure with both D and C12 CI widths."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (metric, tag) in zip(axes, [("D", "D"), ("C12", "C₁₂")]):
        all_widths = {sl: [] for sl in scenario_labels}
        for sl, tbl in tables.items():
            for key in common_keys:
                all_widths[sl].append(ci_width(tbl[key], metric))

        means = [np.mean(all_widths[sl]) for sl in scenario_labels]
        stds = [np.std(all_widths[sl]) for sl in scenario_labels]
        x = np.arange(n_scenarios)
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=COLORS, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_LABELS, fontsize=9)
        ax.set_ylabel(f"Mean CI width ({tag})", fontsize=10)
        ax.set_title(f"{tag}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # annotate bar values
        for i, (b, m) in enumerate(zip(bars, means)):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                    f"{m:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Mean 95% CI Width Across All Pair × Model Combinations",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT / "ci_width_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════
# PLOT E — CI overlay: forest-style per-model + cross-model mean
# ═════════════════════════════════════════════════════════════════════

# Config matching the individual stratified forest plots
_PAIR_ORDER = [
    "image + layout", "image + plain_text", "layout + plain_text",
    "plain_text + table", "image + table", "layout + table",
]
_PAIR_DISPLAY = {
    "image + layout":      "Image vs Layout",
    "image + plain_text":  "Image vs Text",
    "layout + plain_text": "Layout vs Text",
    "plain_text + table":  "Text vs Table",
    "image + table":       "Image vs Table",
    "layout + table":      "Layout vs Table",
}
_MOD_FULL = {
    "image": "Image", "layout": "Layout", "plain_text": "Text", "table": "Table",
}
_MODEL_ORDER = [
    "gemma3_4b", "gemma3_27b", "gpt-4o-mini", "qwen3-vl_8b", "qwen3-vl_30b",
]
_MODEL_DISPLAY = {
    "gemma3_4b":    "Gemma3-4B",
    "gemma3_27b":   "Gemma3-27B",
    "gpt-4o-mini":  "GPT-4o-mini",
    "qwen3-vl_8b":  "Qwen3-VL-8B",
    "qwen3-vl_30b": "Qwen3-VL-30B",
}
_MODEL_COLORS = {
    "gemma3_4b":    "#2962FF",
    "gemma3_27b":   "#7B1FA2",
    "gpt-4o-mini":  "#00897B",
    "qwen3-vl_8b":  "#D32F2F",
    "qwen3-vl_30b": "#FF6F00",
}
_MODEL_MARKERS = {
    "gemma3_4b":    "o",
    "gemma3_27b":   "s",
    "gpt-4o-mini":  "^",
    "qwen3-vl_8b":  "v",
    "qwen3-vl_30b": "P",
}
_SCENARIO_JSONS = OrderedDict([
    ("Strat-Pair",  BASE / "stratified" / "stratified_results.json"),
    ("Strat-Bench", BASE / "stratified_by_benchmark" / "stratified_by_benchmark_results.json"),
    ("Unstratified", BASE / "unstratified" / "unstratified_results.json"),
])
# Line styles to distinguish scenarios (all share model colours)
_SC_LINESTYLES = {"Strat-Pair": "-", "Strat-Bench": "--", "Unstratified": ":"}
_SC_ALPHA      = {"Strat-Pair": 0.90, "Strat-Bench": 0.75, "Unstratified": 0.60}

_ci_overlay_jsons = None   # lazy-loaded cache

def _load_overlay_jsons():
    global _ci_overlay_jsons
    if _ci_overlay_jsons is not None:
        return _ci_overlay_jsons
    import json as _json
    _ci_overlay_jsons = OrderedDict()
    for label, path in _SCENARIO_JSONS.items():
        with open(path, "r", encoding="utf-8") as f:
            raw = _json.load(f)
        _ci_overlay_jsons[label] = raw.get("per_pair", raw)
    return _ci_overlay_jsons


def _clamp_ci(lo, hi, obs):
    return max(0.0, obs - lo), max(0.0, hi - obs)


def plot_ci_overlay(metric, tag):
    """
    Forest-style overlay showing only cross-model mean + CI for each
    of the three scenarios, per modality pair.
    """
    all_data = _load_overlay_jsons()
    pair_labels = [p for p in _PAIR_ORDER if all(p in d for d in all_data.values())]
    n_pairs = len(pair_labels)
    if n_pairs == 0:
        return

    n_sc = len(all_data)
    sc_step = 0.14
    row_height = n_sc * sc_step + 0.30

    cross_key = f"cross_model_{metric}"

    fig, ax = plt.subplots(figsize=(10, row_height * n_pairs + 2.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # alternating row shading
    for pi in range(n_pairs):
        yc = pi * row_height
        if pi % 2 == 1:
            ax.axhspan(yc - row_height / 2, yc + row_height / 2,
                       color="#F5F5F5", zorder=0)

    # ── cross-model mean per scenario ─────────────────────────────
    scenario_labels = list(all_data.keys())
    sc_colors = {
        "Strat-Pair":  "#4C72B0",
        "Strat-Bench": "#DD8452",
        "Unstratified": "#55A868",
    }
    sc_markers = {
        "Strat-Pair":  "D",
        "Strat-Bench": "s",
        "Unstratified": "^",
    }

    for si, slabel in enumerate(scenario_labels):
        pairs = all_data[slabel]
        sc_off = (si - (n_sc - 1) / 2) * sc_step
        for pi, pl in enumerate(pair_labels):
            cross = pairs[pl].get(cross_key, {})
            if cross.get("n_valid_replicates", 0) == 0:
                continue
            ma = cross["mean_across_models"]
            obs = ma["mean"]
            excl = ma["ci_excludes_zero"]
            lo_e, hi_e = _clamp_ci(ma["ci_lo"], ma["ci_hi"], obs)
            fc = sc_colors[slabel] if excl else "white"
            y = pi * row_height + sc_off
            ax.errorbar(
                obs, y,
                xerr=[[lo_e], [hi_e]],
                fmt=sc_markers[slabel],
                color=sc_colors[slabel],
                markerfacecolor=fc,
                markeredgecolor=sc_colors[slabel],
                markeredgewidth=0.8,
                capsize=3, markersize=8, linewidth=1.8,
                zorder=5,
                label=slabel if pi == 0 else None,
            )

    # ── y-axis labels ─────────────────────────────────────────────
    first_pairs = list(all_data.values())[0]
    y_labels = []
    for pl in pair_labels:
        disp = _PAIR_DISPLAY.get(pl, pl).replace(" vs ", " + ")
        comp = first_pairs[pl]["comparison"]
        t_full = _MOD_FULL.get(comp["target"], comp["target"])
        r_full = _MOD_FULL.get(comp["reference"], comp["reference"])
        if metric == "D":
            y_labels.append(f"{disp}\nD = S({t_full}) \u2212 S({r_full})")
        else:
            y_labels.append(disp)

    y_positions = [pi * row_height for pi in range(n_pairs)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=13, rotation=15, ha="right")
    ax.set_ylim(y_positions[-1] + row_height / 2,
                y_positions[0] - row_height / 2)

    # n= annotation
    ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    pad = 0.08 * (xmax - xmin)
    ax.set_xlim(xmin, xmax + pad)
    xmin, xmax = ax.get_xlim()
    x_n = xmax - 0.01 * (xmax - xmin)
    for pi, pl in enumerate(pair_labels):
        n_q = first_pairs[pl]["n_questions"]
        ax.text(x_n, pi * row_height, f"n={n_q}", fontsize=12,
                va="center", ha="right", color="#444", style="italic")

    xlabel = "D (contribution difference)" if metric == "D" else r"$C_{12}$ (cooperation)"
    ax.set_xlabel(xlabel, fontsize=17)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.grid(axis="x", alpha=0.12)
    ax.tick_params(axis="x", labelsize=13)

    met_label = "Contribution" if metric == "D" else "Cooperation"
    ax.set_title(f"Pooled {met_label} Analysis — Scenario Comparison",
                 fontsize=15, fontweight="bold", pad=12)

    # ── legend ────────────────────────────────────────────────────
    ax.legend(fontsize=11, framealpha=0.9,
              loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=n_sc)

    fig.tight_layout()
    out = OUT / f"ci_overlay_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════
# PLOT E2 — Pairwise two-panel overlay (absolute + Δ from baseline)
# ═════════════════════════════════════════════════════════════════════

def plot_pairwise_overlay(metric, compare_key, output_path):
    """
    Two-panel figure comparing *compare_key* against Strat-Bench baseline.

    Panel A: absolute cross-model mean + CI for both scenarios.
    Panel B: differences from the Strat-Bench point estimate, with a
             dashed zero-line marking the baseline.
    Faint connecting lines link each marker between the two panels.
    """
    from matplotlib.patches import ConnectionPatch
    from matplotlib.lines import Line2D

    all_data = _load_overlay_jsons()
    baseline_key = "Strat-Bench"

    pair_labels = [p for p in _PAIR_ORDER
                   if all(p in all_data[s] for s in [baseline_key, compare_key])]
    n_pairs = len(pair_labels)
    if n_pairs == 0:
        return

    n_sc = 2
    sc_step = 0.18
    row_height = n_sc * sc_step + 0.35
    offsets = [(i - (n_sc - 1) / 2) * sc_step for i in range(n_sc)]

    cross_key = f"cross_model_{metric}"

    sc_colors = {
        "Strat-Pair":  "#4C72B0",
        "Strat-Bench": "#DD8452",
        "Unstratified": "#55A868",
    }
    sc_markers = {
        "Strat-Pair":  "D",
        "Strat-Bench": "s",
        "Unstratified": "^",
    }

    scenarios = [compare_key, baseline_key]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(14, row_height * n_pairs + 2.5),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    fig.patch.set_facecolor("white")
    ax_a.set_facecolor("white")
    ax_b.set_facecolor("white")

    # alternating row shading
    for pi in range(n_pairs):
        yc = pi * row_height
        if pi % 2 == 1:
            for ax in (ax_a, ax_b):
                ax.axhspan(yc - row_height / 2, yc + row_height / 2,
                           color="#F0F0F0", zorder=0)

    # ── Panel A: Absolute Values ──────────────────────────────────
    abs_coords = {}
    for si, slabel in enumerate(scenarios):
        for pi, pl in enumerate(pair_labels):
            cross = all_data[slabel][pl].get(cross_key, {})
            if cross.get("n_valid_replicates", 0) == 0:
                continue
            ma = cross["mean_across_models"]
            obs = ma["mean"]
            excl = ma["ci_excludes_zero"]
            lo_e, hi_e = _clamp_ci(ma["ci_lo"], ma["ci_hi"], obs)
            fc = sc_colors[slabel] if excl else "white"
            y = pi * row_height + offsets[si]
            ax_a.errorbar(
                obs, y,
                xerr=[[lo_e], [hi_e]],
                fmt=sc_markers[slabel],
                color=sc_colors[slabel],
                markerfacecolor=fc,
                markeredgecolor=sc_colors[slabel],
                markeredgewidth=0.8,
                capsize=3, markersize=8, linewidth=1.8,
                zorder=5,
                label=slabel if pi == 0 else None,
            )
            abs_coords[(slabel, pi)] = (obs, y)

    # ── Panel B: Differences from baseline ────────────────────────
    diff_coords = {}
    for si, slabel in enumerate(scenarios):
        for pi, pl in enumerate(pair_labels):
            base_cross = all_data[baseline_key][pl].get(cross_key, {})
            if base_cross.get("n_valid_replicates", 0) == 0:
                continue
            base_mean = base_cross["mean_across_models"]["mean"]

            cross = all_data[slabel][pl].get(cross_key, {})
            if cross.get("n_valid_replicates", 0) == 0:
                continue
            ma = cross["mean_across_models"]
            diff_mean = ma["mean"] - base_mean
            diff_lo = ma["ci_lo"] - base_mean
            diff_hi = ma["ci_hi"] - base_mean
            lo_e = max(0.0, diff_mean - diff_lo)
            hi_e = max(0.0, diff_hi - diff_mean)

            y = pi * row_height + offsets[si]
            ax_b.errorbar(
                diff_mean, y,
                xerr=[[lo_e], [hi_e]],
                fmt=sc_markers[slabel],
                color=sc_colors[slabel],
                markerfacecolor=sc_colors[slabel],
                markeredgecolor=sc_colors[slabel],
                markeredgewidth=0.8,
                capsize=3, markersize=8, linewidth=1.8,
                zorder=5,
            )
            diff_coords[(slabel, pi)] = (diff_mean, y)

    # dashed baseline at x=0 in Panel B
    ax_b.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5, zorder=1)

    # ── Connecting lines between panels ───────────────────────────
    for si, slabel in enumerate(scenarios):
        for pi in range(n_pairs):
            if (slabel, pi) in abs_coords and (slabel, pi) in diff_coords:
                x_a, y_a = abs_coords[(slabel, pi)]
                x_b, y_b = diff_coords[(slabel, pi)]
                con = ConnectionPatch(
                    xyA=(x_a, y_a), xyB=(x_b, y_b),
                    coordsA="data", coordsB="data",
                    axesA=ax_a, axesB=ax_b,
                    color=sc_colors[slabel], alpha=0.15, linewidth=0.8,
                    zorder=0,
                )
                fig.add_artist(con)

    # ── Y-axis labels ─────────────────────────────────────────────
    y_labels = [_PAIR_DISPLAY.get(pl, pl).replace(" vs ", " + ")
                for pl in pair_labels]
    y_positions = [pi * row_height for pi in range(n_pairs)]
    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels(y_labels, fontsize=12)
    ax_a.set_ylim(y_positions[-1] + row_height / 2,
                  y_positions[0] - row_height / 2)

    # n= annotations on right of Panel B
    ax_b.autoscale_view()
    xmin_b, xmax_b = ax_b.get_xlim()
    pad_b = 0.14 * (xmax_b - xmin_b)
    ax_b.set_xlim(xmin_b, xmax_b + pad_b)
    xmin_b, xmax_b = ax_b.get_xlim()
    x_n = xmax_b - 0.01 * (xmax_b - xmin_b)
    first_pairs = all_data[baseline_key]
    for pi, pl in enumerate(pair_labels):
        n_q = first_pairs[pl]["n_questions"]
        ax_b.text(x_n, pi * row_height, f"n={n_q}", fontsize=11,
                  va="center", ha="right", color="#444", style="italic")

    # ── Panel titles ──────────────────────────────────────────────
    ax_a.set_title("A. Absolute Values", fontsize=13, fontweight="bold",
                   loc="left")
    ax_b.set_title("B. Differences from Strat-Bench Baseline",
                   fontsize=13, fontweight="bold", loc="left")

    # ── X-axis labels ─────────────────────────────────────────────
    met_label = "Contribution" if metric == "D" else "Cooperation"
    ax_a.set_xlabel(f"{met_label} Estimate", fontsize=12)
    ax_b.set_xlabel(f"{met_label} Difference (\u0394 from Strat-Bench)",
                    fontsize=12)

    # ── Spine / grid ──────────────────────────────────────────────
    for ax in (ax_a, ax_b):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.grid(axis="x", alpha=0.12)
        ax.tick_params(axis="x", labelsize=11)

    # ── Legend ────────────────────────────────────────────────────
    sc_handles = [
        Line2D([0], [0], marker=sc_markers[s], color=sc_colors[s],
               markerfacecolor=sc_colors[s], markeredgecolor=sc_colors[s],
               linestyle="", markersize=8, label=s)
        for s in scenarios
    ]
    baseline_line = Line2D([0], [0], color="black", linewidth=1, linestyle="--",
                           label="Strat-Bench Baseline (Right Panel Only)")
    all_handles = sc_handles + [baseline_line]
    fig.legend(
        handles=all_handles,
        loc="lower center", ncol=len(all_handles),
        fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    # ── Titles ────────────────────────────────────────────────────
    subtitle = (f"Pooled {met_label} Analysis: "
                "Absolute Estimates and Differences from Baseline")
    fig.suptitle("Visualizing Subtle Differences in Bootstrap Scenarios\n"
                 + subtitle,
                 fontsize=14, fontweight="bold", y=1.06)

    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ═════════════════════════════════════════════════════════════════════
# PLOT E3 — Heatmaps: Δ mean & Δ CI width from Strat-Bench baseline
# ═════════════════════════════════════════════════════════════════════

def plot_diff_heatmaps(metric):
    """
    1×2 heatmap: Δ CI Width from Strat-Bench baseline.

    Left  : Strat-Pair − Strat-Bench
    Right : Unstratified − Strat-Bench

    Each panel: y = modality pairs, x = per-model cols + cross-model mean.
    """
    from matplotlib.colors import TwoSlopeNorm

    all_data = _load_overlay_jsons()
    baseline_key = "Strat-Bench"
    compare_keys = ["Strat-Pair", "Unstratified"]

    pair_labels = [p for p in _PAIR_ORDER
                   if all(p in all_data[s]
                          for s in [baseline_key] + compare_keys)]
    pair_display = [_PAIR_DISPLAY.get(p, p).replace(" vs ", " + ")
                    for p in pair_labels]

    col_display = ([_MODEL_DISPLAY.get(m, m) for m in _MODEL_ORDER]
                   + ["Cross-model\nmean"])

    per_key   = f"per_model_{metric}"
    cross_key = f"cross_model_{metric}"

    n_rows = len(pair_labels)
    n_cols = len(col_display)

    # ── build matrices ────────────────────────────────────────────
    matrices = {}
    for ck in compare_keys:
        mat = np.full((n_rows, n_cols), np.nan)
        for ri, pl in enumerate(pair_labels):
            for mi, model in enumerate(_MODEL_ORDER):
                base_pm = all_data[baseline_key][pl].get(
                    per_key, {}).get(model)
                comp_pm = all_data[ck][pl].get(
                    per_key, {}).get(model)
                if base_pm and comp_pm:
                    mat[ri, mi] = (
                        (comp_pm["ci_hi"] - comp_pm["ci_lo"])
                        - (base_pm["ci_hi"] - base_pm["ci_lo"]))

            base_c = (all_data[baseline_key][pl]
                      .get(cross_key, {})
                      .get("mean_across_models"))
            comp_c = (all_data[ck][pl]
                      .get(cross_key, {})
                      .get("mean_across_models"))
            if base_c and comp_c:
                mat[ri, -1] = (
                    (comp_c["ci_hi"] - comp_c["ci_lo"])
                    - (base_c["ci_hi"] - base_c["ci_lo"]))
        matrices[ck] = mat

    col_titles = [
        "Strat-Pair − Strat-Bench",
        "Unstratified − Strat-Bench",
    ]

    # shared colour scale across both panels
    all_vals = np.concatenate([m.ravel() for m in matrices.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vabs = np.max(np.abs(all_vals)) if len(all_vals) else 0.01
    if vabs == 0:
        vabs = 0.01
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"wspace": 0.18})
    fig.patch.set_facecolor("white")

    for ci, ck in enumerate(compare_keys):
        ax = axes[ci]
        ax.set_facecolor("white")
        mat = matrices[ck]

        im = ax.imshow(mat, cmap="RdBu_r", norm=norm,
                       aspect="auto", interpolation="nearest")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                color = "white" if abs(val) > 0.65 * vabs else "black"
                ax.text(j, i, f"{val:+.4f}", ha="center",
                        va="center", fontsize=9, color=color,
                        fontweight="bold" if j == n_cols - 1
                        else "normal")

        ax.axvline(n_cols - 1.5, color="black", lw=1.2, ls="-")

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_display, fontsize=10,
                           rotation=40, ha="right")
        if ci == 0:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(pair_display, fontsize=11)
        else:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels([])

        ax.set_title(f"Δ CI Width:  {col_titles[ci]}",
                     fontsize=12, fontweight="bold", pad=10)

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.8,
                        pad=0.02, aspect=25)
    cbar.ax.tick_params(labelsize=10)

    met_label = ("Contribution (D)" if metric == "D"
                 else r"Cooperation ($C_{12}$)")
    fig.suptitle(
        f"CI Width Differences — {met_label}",
        fontsize=15, fontweight="bold", y=1.02)

    out = OUT / f"diff_heatmap_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════
# PLOT F — Numerical diff table saved as CSV
# ═════════════════════════════════════════════════════════════════════

def write_diff_csv():
    rows = []
    for p in pairs_ordered:
        for m in models_ordered:
            key = (p, m)
            if key not in tables[scenario_labels[0]]:
                continue
            row = {"modality_pair": p, "model": m}
            for metric in ("D", "C12"):
                w_sp = ci_width(tables[scenario_labels[0]][key], metric)
                w_sb = ci_width(tables[scenario_labels[1]][key], metric)
                w_un = ci_width(tables[scenario_labels[2]][key], metric)
                row[f"{metric}_width_strat_pair"] = round(w_sp, 6)
                row[f"{metric}_width_strat_bench"] = round(w_sb, 6)
                row[f"{metric}_width_unstrat"] = round(w_un, 6)
                row[f"{metric}_diff_bench_vs_pair"] = round(w_sb - w_sp, 6)
                row[f"{metric}_diff_unstrat_vs_pair"] = round(w_un - w_sp, 6)
                if abs(w_sp) > 1e-12:
                    row[f"{metric}_pct_bench_vs_pair"] = round(100*(w_sb - w_sp)/w_sp, 3)
                    row[f"{metric}_pct_unstrat_vs_pair"] = round(100*(w_un - w_sp)/w_sp, 3)
            rows.append(row)
    path = OUT / "scenario_differences.csv"
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════
# PLOT G — Slopegraph: lines connecting CI width across 3 scenarios
# ═════════════════════════════════════════════════════════════════════

def plot_slopegraph(metric, tag):
    """
    One line per pair×model combo.  X-axis = scenario, Y-axis = CI width.
    Lines going up left-to-right = CI gets wider = more uncertainty.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    x = np.array([0, 1, 2])

    # Colours by pair
    pair_cmap = plt.cm.Set2(np.linspace(0, 1, n_pairs))

    for pi, p in enumerate(pairs_ordered):
        for mi, m in enumerate(models_ordered):
            key = (p, m)
            widths = []
            for sl in scenario_labels:
                if key in tables[sl]:
                    widths.append(ci_width(tables[sl][key], metric))
                else:
                    widths.append(np.nan)
            label = p if mi == 0 else None
            ax.plot(x, widths, "-o", color=pair_cmap[pi], markersize=4,
                    linewidth=1.2, alpha=0.7, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, fontsize=10)
    ax.set_ylabel(f"{tag} 95% CI width", fontsize=10)
    ax.set_title(f"{tag}: CI Width Across Scenarios\n(one line per pair × model)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", title="Modality pair",
              title_fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / f"slopegraph_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════
# PLOT H — Parallel coordinates: normalised to Strat-Pair baseline
# ═════════════════════════════════════════════════════════════════════

def plot_parallel_normalised(metric, tag):
    """
    Each line = pair×model.  Y shows % change relative to Strat-Pair
    (which is always 0%).  Highlights the direction & magnitude.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    x = np.array([0, 1, 2])
    pair_cmap = plt.cm.Set2(np.linspace(0, 1, n_pairs))

    for pi, p in enumerate(pairs_ordered):
        for mi, m in enumerate(models_ordered):
            key = (p, m)
            if key not in tables[scenario_labels[0]]:
                continue
            w_ref = ci_width(tables[scenario_labels[0]][key], metric)
            if abs(w_ref) < 1e-12:
                continue
            pcts = []
            for sl in scenario_labels:
                w = ci_width(tables[sl][key], metric)
                pcts.append(100.0 * (w - w_ref) / w_ref)
            label = p if mi == 0 else None
            ax.plot(x, pcts, "-o", color=pair_cmap[pi], markersize=4,
                    linewidth=1.2, alpha=0.7, label=label)

    ax.axhline(0, color="gray", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, fontsize=10)
    ax.set_ylabel(f"% change in {tag} CI width (vs Strat-Pair)", fontsize=10)
    ax.set_title(f"{tag}: CI Width % Change Relative to Strat-Pair\n"
                 f"(one line per pair × model)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", title="Modality pair",
              title_fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / f"parallel_normalised_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════
# PLOT I — Butterfly / tornado: % change both directions from baseline
# ═════════════════════════════════════════════════════════════════════

def plot_butterfly(metric, tag):
    """
    Horizontal bars showing % CI-width change for each pair
    (averaged over models).  Strat-Bench on the left, Unstrat on the right.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(n_pairs)

    pct_bench = []
    pct_unstrat = []
    for p in pairs_ordered:
        vals_b, vals_u = [], []
        for m in models_ordered:
            key = (p, m)
            w0 = ci_width(tables[scenario_labels[0]][key], metric)
            wb = ci_width(tables[scenario_labels[1]][key], metric)
            wu = ci_width(tables[scenario_labels[2]][key], metric)
            if abs(w0) > 1e-12:
                vals_b.append(100 * (wb - w0) / w0)
                vals_u.append(100 * (wu - w0) / w0)
        pct_bench.append(np.mean(vals_b))
        pct_unstrat.append(np.mean(vals_u))

    bars_b = ax.barh(y - 0.18, pct_bench, 0.35, color=COLORS[1],
                     edgecolor="white", label=f"{SHORT_LABELS[1]} vs {SHORT_LABELS[0]}")
    bars_u = ax.barh(y + 0.18, pct_unstrat, 0.35, color=COLORS[2],
                     edgecolor="white", label=f"{SHORT_LABELS[2]} vs {SHORT_LABELS[0]}")

    # annotate
    for bar, val in zip(bars_b, pct_bench):
        ax.text(bar.get_width() + (-0.15 if val < 0 else 0.05), bar.get_y() + bar.get_height()/2,
                f"{val:+.2f}%", va="center", fontsize=8, color=COLORS[1], fontweight="bold")
    for bar, val in zip(bars_u, pct_unstrat):
        ax.text(bar.get_width() + (-0.15 if val < 0 else 0.05), bar.get_y() + bar.get_height()/2,
                f"{val:+.2f}%", va="center", fontsize=8, color=COLORS[2], fontweight="bold")

    ax.axvline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(pairs_ordered, fontsize=9)
    ax.set_xlabel(f"% change in {tag} CI width (mean over models)", fontsize=10)
    ax.set_title(f"{tag}: CI Width Change (%) — Butterfly Chart",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = OUT / f"butterfly_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════
# PLOT J — Bootstrap Std ratio scatter (variance inflation factor)
# ═════════════════════════════════════════════════════════════════════

def plot_std_ratio(metric, tag):
    """
    Scatter of std_alt / std_ref for each pair×model.
    Ratio > 1 means the alternative scenario has MORE variance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    comparisons = [
        (SHORT_LABELS[1], scenario_labels[1], scenario_labels[0], COLORS[1]),
        (SHORT_LABELS[2], scenario_labels[2], scenario_labels[0], COLORS[2]),
    ]

    for ax, (comp_label, s_alt, s_ref, color) in zip(axes, comparisons):
        pair_cmap = plt.cm.Set2(np.linspace(0, 1, n_pairs))
        for pi, p in enumerate(pairs_ordered):
            ratios = []
            for m in models_ordered:
                key = (p, m)
                std_ref = tables[s_ref][key][f"{metric}_std"]
                std_alt = tables[s_alt][key][f"{metric}_std"]
                if abs(std_ref) > 1e-12:
                    ratios.append(std_alt / std_ref)
            ax.scatter(range(len(ratios)), ratios, color=pair_cmap[pi],
                       s=60, edgecolors="black", linewidth=0.3, zorder=3,
                       label=p)

        ax.axhline(1.0, color="gray", linewidth=1.2, linestyle="--")
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(models_ordered, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel(f"Std ratio ({tag})", fontsize=10)
        ax.set_title(f"{comp_label} / {SHORT_LABELS[0]}",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=7, loc="best", title="Pair", title_fontsize=7)

    fig.suptitle(f"{tag}: Bootstrap Std Ratio (Variance Inflation Factor)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT / f"std_ratio_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════
# PLOT K — Three-panel heatmap: raw CI widths side-by-side
# ═════════════════════════════════════════════════════════════════════

def plot_triple_heatmap(metric, tag):
    """Three heatmaps of CI width (same colour scale) so visual comparison is direct."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    mats = []
    for sl in scenario_labels:
        mat = np.full((n_models, n_pairs), np.nan)
        for pi, p in enumerate(pairs_ordered):
            for mi, m in enumerate(models_ordered):
                key = (p, m)
                if key in tables[sl]:
                    mat[mi, pi] = ci_width(tables[sl][key], metric)
        mats.append(mat)

    vmin = min(np.nanmin(m) for m in mats)
    vmax = max(np.nanmax(m) for m in mats)

    for ax, mat, slabel in zip(axes, mats, SHORT_LABELS):
        im = ax.imshow(mat, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_pairs))
        ax.set_xticklabels([p.replace(" + ", "\n+\n") for p in pairs_ordered],
                           fontsize=8)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(models_ordered, fontsize=9)
        ax.set_title(slabel, fontsize=11, fontweight="bold")
        for mi in range(n_models):
            for pi in range(n_pairs):
                val = mat[mi, pi]
                if not np.isnan(val):
                    ax.text(pi, mi, f"{val:.3f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if val > (vmin + vmax) * 0.6 else "black")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label=f"{tag} CI width")
    fig.suptitle(f"{tag}: 95% CI Width — Three Scenarios (same scale)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.15)
    out = OUT / f"triple_heatmap_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════
# PLOT L — Dumbbell chart: CI endpoints per pair×model, all 3 scenarios
# ═════════════════════════════════════════════════════════════════════

def plot_dumbbell(metric, tag):
    """
    For each pair×model, draw a thin line from CI_lo to CI_hi for each
    scenario.  Lines are stacked vertically with a small offset.
    """
    combos = [(p, m) for p in pairs_ordered for m in models_ordered
              if (p, m) in tables[scenario_labels[0]]]
    n_combos = len(combos)

    fig, ax = plt.subplots(figsize=(12, max(8, n_combos * 0.35)))
    offsets = [-0.22, 0.0, 0.22]

    for ci_idx, (p, m) in enumerate(combos):
        key = (p, m)
        y_base = ci_idx
        for si, sl in enumerate(scenario_labels):
            lo = tables[sl][key][f"{metric}_ci_lo"]
            hi = tables[sl][key][f"{metric}_ci_hi"]
            mid = (lo + hi) / 2.0
            y = y_base + offsets[si]
            ax.plot([lo, hi], [y, y], color=COLORS[si], linewidth=2.0,
                    solid_capstyle="round", alpha=0.8)
            ax.plot(lo, y, "|", color=COLORS[si], markersize=6)
            ax.plot(hi, y, "|", color=COLORS[si], markersize=6)
            ax.plot(mid, y, "o", color=COLORS[si], markersize=3, zorder=5)

    for si in range(n_scenarios):
        ax.plot([], [], color=COLORS[si], linewidth=2.5,
                label=SHORT_LABELS[si])

    ax.set_yticks(range(n_combos))
    ax.set_yticklabels([f"{p}  |  {m}" for p, m in combos], fontsize=7)
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel(f"{tag} value", fontsize=10)
    ax.set_title(f"{tag}: Dumbbell CI Comparison — All Pair × Model",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    fig.tight_layout()
    out = OUT / f"dumbbell_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════
# PLOT M — Combined metric: D & C12 % change scatter
# ═════════════════════════════════════════════════════════════════════

def plot_scatter_D_vs_C12():
    """
    X = % change in D CI width,  Y = % change in C12 CI width.
    One point per pair×model, coloured by comparison type.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    comparisons = [
        (SHORT_LABELS[1], scenario_labels[1], COLORS[1], "D"),
        (SHORT_LABELS[2], scenario_labels[2], COLORS[2], "o"),
    ]

    for comp_label, s_alt, color, marker in comparisons:
        xs, ys = [], []
        for p in pairs_ordered:
            for m in models_ordered:
                key = (p, m)
                w0_d = ci_width(tables[scenario_labels[0]][key], "D")
                wa_d = ci_width(tables[s_alt][key], "D")
                w0_c = ci_width(tables[scenario_labels[0]][key], "C12")
                wa_c = ci_width(tables[s_alt][key], "C12")
                if abs(w0_d) > 1e-12 and abs(w0_c) > 1e-12:
                    xs.append(100 * (wa_d - w0_d) / w0_d)
                    ys.append(100 * (wa_c - w0_c) / w0_c)
        ax.scatter(xs, ys, marker=marker, s=50, color=color, alpha=0.7,
                   edgecolors="black", linewidth=0.3,
                   label=f"vs {SHORT_LABELS[0]}: {comp_label}")

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("% change in D CI width", fontsize=10)
    ax.set_ylabel("% change in C₁₂ CI width", fontsize=10)
    ax.set_title("Joint D & C₁₂ CI Width Change (%)\nEach dot = one pair × model",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT / "scatter_D_vs_C12_pct.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for metric, tag in [("D", "D (contribution)"), ("C12", "C₁₂ (cooperation)")]:
        plot_ci_width_bars(metric, tag)
        plot_ci_width_diff_per_model(metric, tag)
        plot_heatmap_relative(metric, tag)
        plot_ci_overlay(metric, tag)
        for compare in ("Strat-Pair", "Unstratified"):
            tag_sc = "strat" if compare == "Strat-Pair" else "unstrat"
            plot_pairwise_overlay(
                metric, compare,
                OUT / f"overlay_{metric}_{tag_sc}_vs_bench.png")
        plot_diff_heatmaps(metric)
        plot_slopegraph(metric, tag)
        plot_parallel_normalised(metric, tag)
        plot_butterfly(metric, tag)
        plot_std_ratio(metric, tag)
        plot_triple_heatmap(metric, tag)
        plot_dumbbell(metric, tag)

    plot_summary_bars()
    plot_scatter_D_vs_C12()
    write_diff_csv()
    print("\nAll comparison plots saved.")


if __name__ == "__main__":
    main()
