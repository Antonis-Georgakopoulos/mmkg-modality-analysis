"""
Forest plots for cluster (per-document) bootstrap results.

Reads cluster_consistency_results.json and produces the same set of
figures as plot_consistency_compact.py:
  - Single-benchmark forest plots
  - Combined two-panel (mean-only) forest plots
  - Combined two-panel per-model forest plots

All visual parameters (fonts, line widths, colours) match the original.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_JSON = Path(__file__).parent / "cluster_consistency_results.json"
OUTPUT_DIR   = Path(__file__).parent

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

PAIR_ORDER = [
    "image + layout",
    "image + plain_text",
    "layout + plain_text",
    "plain_text + table",
    "image + table",
    "layout + table",
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

_POS  = "#2E7D32"
_NEG  = "#C62828"
_NEUT = "#616161"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(lo, hi, obs):
    """Return non-negative error bar half-widths."""
    return max(0.0, obs - lo), max(0.0, hi - obs)


# ── Single-benchmark forest plot ─────────────────────────────────────────────

def plot_forest(pairs, bench_name, metric, output_path):
    pair_labels = [p for p in PAIR_ORDER if p in pairs]
    n_pairs = len(pair_labels)
    if n_pairs == 0:
        return

    per_key   = f"per_model_{metric}"
    cross_key = f"cross_model_{metric}"
    obs_key   = f"observed_{metric}"

    n_m = len(MODEL_ORDER)
    n_lines = n_m + 1
    step = 0.12
    offsets_all = [(i - (n_lines - 1) / 2) * step for i in range(n_lines)]
    model_offsets = offsets_all[:n_m]
    mean_offset   = offsets_all[n_m]
    row_height = n_lines * step + 0.35

    fig, ax = plt.subplots(figsize=(8, row_height * n_pairs + 1.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for pi in range(n_pairs):
        yc = pi * row_height
        if pi % 2 == 1:
            ax.axhspan(yc - row_height / 2, yc + row_height / 2,
                       color="#F5F5F5", zorder=0)

    for mi, model in enumerate(MODEL_ORDER):
        xs, ys, elo, ehi = [], [], [], []
        for pi, pl in enumerate(pair_labels):
            pm = pairs[pl].get(per_key, {})
            if model not in pm:
                continue
            r = pm[model]
            obs = r[obs_key]
            lo_e, hi_e = _clamp(r["ci_lo"], r["ci_hi"], obs)
            xs.append(obs)
            ys.append(pi * row_height + model_offsets[mi])
            elo.append(lo_e)
            ehi.append(hi_e)
        if xs:
            ax.errorbar(
                xs, ys, xerr=[elo, ehi],
                fmt=MODEL_MARKERS.get(model, "o"),
                color=MODEL_COLORS.get(model, "#999"),
                capsize=2.5, markersize=7, linewidth=1.6, alpha=0.85,
                label=MODEL_DISPLAY.get(model, model),
                zorder=3,
            )

    for pi, pl in enumerate(pair_labels):
        cross = pairs[pl].get(cross_key, {})
        if cross.get("n_valid_replicates", 0) == 0:
            continue
        ma = cross["mean_across_models"]
        excl = ma["ci_excludes_zero"]
        lo_e, hi_e = _clamp(ma["ci_lo"], ma["ci_hi"], ma["mean"])
        fc = "black" if excl else "white"
        ax.errorbar(
            ma["mean"], pi * row_height + mean_offset,
            xerr=[[lo_e], [hi_e]],
            fmt="D", color="black", markerfacecolor=fc,
            markeredgecolor="black", markeredgewidth=0.9,
            capsize=4, markersize=10, linewidth=2.2, zorder=5,
            label="Cross-model mean" if pi == 0 else None,
        )

    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.35, zorder=1)

    y_labels = []
    for pl in pair_labels:
        disp = PAIR_DISPLAY.get(pl, pl).replace(" vs ", " + ")
        comp = pairs[pl]["comparison"]
        t_full = MOD_FULL.get(comp["target"], comp["target"])
        r_full = MOD_FULL.get(comp["reference"], comp["reference"])
        n_q = pairs[pl]["n_questions"]
        n_d = pairs[pl].get("n_documents", "?")
        if metric == "D":
            y_labels.append(
                f"{disp} (n={n_q}, d={n_d})\n"
                f"D = S({t_full}) \u2212 S({r_full})")
        else:
            y_labels.append(f"{disp} (n={n_q}, d={n_d})")

    y_positions = [pi * row_height for pi in range(n_pairs)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=13, rotation=15, ha="right")
    ax.set_ylim(y_positions[-1] + row_height / 2,
                y_positions[0] - row_height / 2)

    xlabel = "D (contribution difference)" if metric == "D" else r"$C_{12}$ (cooperation)"
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_title(f"{bench_name}  (cluster bootstrap)", fontsize=18, fontweight="bold")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.grid(axis="x", alpha=0.12)
    ax.tick_params(axis="x", labelsize=13)

    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Combined two-panel (mean-only) ──────────────────────────────────────────

def plot_forest_combined(all_results, metric, output_path):
    benchmarks = list(all_results.keys())
    n_bench = len(benchmarks)
    if n_bench == 0:
        return

    all_pairs_set = set()
    for bench in benchmarks:
        all_pairs_set.update(all_results[bench].keys())
    pair_labels = [p for p in PAIR_ORDER if p in all_pairs_set]
    n_pairs = len(pair_labels)
    if n_pairs == 0:
        return

    cross_key = f"cross_model_{metric}"
    row_height = 1.0

    fig, axes = plt.subplots(1, n_bench,
                             figsize=(7 * n_bench, row_height * n_pairs + 1.5),
                             sharey=True)
    if n_bench == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax_i, bench in enumerate(benchmarks):
        ax = axes[ax_i]
        ax.set_facecolor("white")
        pairs = all_results[bench]

        for pi in range(n_pairs):
            yc = pi * row_height
            if pi % 2 == 1:
                ax.axhspan(yc - row_height / 2, yc + row_height / 2,
                           color="#F5F5F5", zorder=0)

        for pi, pl in enumerate(pair_labels):
            if pl not in pairs:
                continue
            cross = pairs[pl].get(cross_key, {})
            if cross.get("n_valid_replicates", 0) == 0:
                continue
            ma = cross["mean_across_models"]
            excl = ma["ci_excludes_zero"]
            lo_e, hi_e = _clamp(ma["ci_lo"], ma["ci_hi"], ma["mean"])
            if excl:
                c = _POS if ma["mean"] > 0 else _NEG
            else:
                c = _NEUT
            fc = c if excl else "white"
            ax.errorbar(
                ma["mean"], pi * row_height,
                xerr=[[lo_e], [hi_e]],
                fmt="D", color=c, markerfacecolor=fc,
                markeredgecolor=c, markeredgewidth=1.0,
                capsize=4, markersize=10, linewidth=2.0, zorder=5,
            )

        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.35, zorder=1)

        xlabel = "D (contribution difference)" if metric == "D" else r"$C_{12}$ (cooperation)"
        ax.set_xlabel(xlabel, fontsize=17)
        ax.set_title(bench, fontsize=18, fontweight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.grid(axis="x", alpha=0.12)
        ax.tick_params(axis="x", labelsize=13)

    y_labels = []
    for pl in pair_labels:
        disp = PAIR_DISPLAY.get(pl, pl)
        comp = None
        for bench in benchmarks:
            if pl in all_results[bench]:
                comp = all_results[bench][pl]["comparison"]
                break
        if metric == "D" and comp:
            t_full = MOD_FULL.get(comp["target"], comp["target"])
            r_full = MOD_FULL.get(comp["reference"], comp["reference"])
            y_labels.append(f"{disp}\nD = S({t_full}) \u2212 S({r_full})")
        else:
            y_labels.append(disp)

    y_positions = [pi * row_height for pi in range(n_pairs)]
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=13, rotation=15, ha="right")
    axes[0].set_ylim(y_positions[-1] + row_height / 2,
                     y_positions[0] - row_height / 2)

    fig.suptitle("Cross-model consistency (cluster bootstrap)",
                 fontsize=19, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Combined two-panel per-model ─────────────────────────────────────────────

def plot_forest_permodel_twopanel(all_results, metric, output_path):
    benchmarks = list(all_results.keys())
    n_bench = len(benchmarks)
    if n_bench == 0:
        return

    all_pairs_set = set()
    for bench in benchmarks:
        all_pairs_set.update(all_results[bench].keys())
    pair_labels = [p for p in PAIR_ORDER if p in all_pairs_set]
    n_pairs = len(pair_labels)
    if n_pairs == 0:
        return

    per_key   = f"per_model_{metric}"
    cross_key = f"cross_model_{metric}"
    obs_key   = f"observed_{metric}"
    n_m = len(MODEL_ORDER)

    # 6 lines per row: 5 models + 1 cross-model mean, each at a fixed offset
    n_lines = n_m + 1
    step = 0.12
    offsets_all = [(i - (n_lines - 1) / 2) * step for i in range(n_lines)]
    model_offsets = offsets_all[:n_m]
    mean_offset   = offsets_all[n_m]

    row_height = n_lines * step + 0.35

    fig, axes = plt.subplots(
        1, n_bench,
        figsize=(7 * n_bench, row_height * n_pairs + 2.2),
        sharey=True,
    )
    if n_bench == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax_i, bench in enumerate(benchmarks):
        ax = axes[ax_i]
        ax.set_facecolor("white")
        pairs = all_results[bench]
        is_left = (ax_i == 0)

        # ── alternating row shading ─────────────────────────────────
        for pi in range(n_pairs):
            yc = pi * row_height
            if pi % 2 == 1:
                ax.axhspan(yc - row_height / 2, yc + row_height / 2,
                           color="#F5F5F5", zorder=0)

        # ── per-model points ────────────────────────────────────────
        for mi, model in enumerate(MODEL_ORDER):
            xs, ys, elo, ehi = [], [], [], []
            for pi, pl in enumerate(pair_labels):
                if pl not in pairs:
                    continue
                pm = pairs[pl].get(per_key, {})
                if model not in pm:
                    continue
                r = pm[model]
                obs = r[obs_key]
                lo_e, hi_e = _clamp(r["ci_lo"], r["ci_hi"], obs)
                xs.append(obs)
                ys.append(pi * row_height + model_offsets[mi])
                elo.append(lo_e)
                ehi.append(hi_e)

            if xs:
                ax.errorbar(
                    xs, ys, xerr=[elo, ehi],
                    fmt=MODEL_MARKERS.get(model, "o"),
                    color=MODEL_COLORS.get(model, "#999"),
                    capsize=2.5, markersize=7, linewidth=1.6, alpha=0.85,
                    label=MODEL_DISPLAY.get(model, model) if is_left else None,
                    zorder=3,
                )

        # ── cross-model mean diamond ────────────────────────────────
        for pi, pl in enumerate(pair_labels):
            if pl not in pairs:
                continue
            cross = pairs[pl].get(cross_key, {})
            if cross.get("n_valid_replicates", 0) == 0:
                continue
            ma = cross["mean_across_models"]
            excl = ma["ci_excludes_zero"]
            lo_e, hi_e = _clamp(ma["ci_lo"], ma["ci_hi"], ma["mean"])
            fc = "black" if excl else "white"
            ax.errorbar(
                ma["mean"], pi * row_height + mean_offset,
                xerr=[[lo_e], [hi_e]],
                fmt="D", color="black", markerfacecolor=fc,
                markeredgecolor="black", markeredgewidth=0.9,
                capsize=4, markersize=10, linewidth=2.2, zorder=5,
                label="Cross-model mean" if (is_left and pi == 0) else None,
            )

        # ── zero reference line ─────────────────────────────────────
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.35, zorder=1)

        # ── (n=X, d=Y) on the right edge ───────────────────────────
        ax.autoscale_view()
        xmin, xmax = ax.get_xlim()
        pad = 0.10 * (xmax - xmin)
        ax.set_xlim(xmin, xmax + pad)
        xmin, xmax = ax.get_xlim()
        x_n = xmax - 0.01 * (xmax - xmin)
        for pi, pl in enumerate(pair_labels):
            if pl not in pairs:
                continue
            n_q = pairs[pl]["n_questions"]
            n_d = pairs[pl].get("n_documents", "?")
            ax.text(x_n, pi * row_height, f"n={n_q}\nd={n_d}", fontsize=12,
                    va="center", ha="right", color="#444", style="italic")

        # ── cosmetics ───────────────────────────────────────────────
        xlabel = "D (contribution difference)" if metric == "D" else r"$C_{12}$ (cooperation)"
        ax.set_xlabel(xlabel, fontsize=17)
        ax.set_title(bench, fontsize=18, fontweight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.grid(axis="x", alpha=0.12)
        ax.tick_params(axis="x", labelsize=13)

    # ── shared y-axis labels ────────────────────────────────────────
    y_labels = []
    for pl in pair_labels:
        disp = PAIR_DISPLAY.get(pl, pl).replace(" vs ", " + ")
        comp = None
        for bench in benchmarks:
            if pl in all_results[bench]:
                comp = all_results[bench][pl]["comparison"]
                break
        if metric == "D" and comp:
            t_full = MOD_FULL.get(comp["target"], comp["target"])
            r_full = MOD_FULL.get(comp["reference"], comp["reference"])
            y_labels.append(f"{disp}\nD = S({t_full}) \u2212 S({r_full})")
        else:
            y_labels.append(disp)

    y_positions = [pi * row_height for pi in range(n_pairs)]
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=13, rotation=15, ha="right")
    axes[0].set_ylim(y_positions[-1] + row_height / 2,
                     y_positions[0] - row_height / 2)

    # ── shared legend ───────────────────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=min(7, len(handles)),
        fontsize=12, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    if metric == "D":
        sup_title = "Cross-Model Contribution Analysis (cluster bootstrap)"
    else:
        sup_title = "Cross-Model Cooperation Analysis (cluster bootstrap)"
    fig.suptitle(sup_title, fontsize=19, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    for bench, pairs in all_results.items():
        safe = bench.replace(" ", "_").replace("-", "_")
        for metric in ("D", "C12"):
            path = OUTPUT_DIR / f"cluster_forest_{metric}_{safe}.png"
            plot_forest(pairs, bench, metric, path)

    for metric in ("D", "C12"):
        path = OUTPUT_DIR / f"cluster_forest_{metric}_combined.png"
        plot_forest_combined(all_results, metric, path)

    for metric in ("D", "C12"):
        path = OUTPUT_DIR / f"cluster_forest_{metric}_permodel.png"
        plot_forest_permodel_twopanel(all_results, metric, path)

    print("Done.")


if __name__ == "__main__":
    main()
