"""
Forest plot for stratified-by-benchmark question-level bootstrap results.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────

RESULTS_JSON = Path(__file__).parent / "stratified_by_benchmark_results.json"
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


def _clamp(lo, hi, obs):
    return max(0.0, obs - lo), max(0.0, hi - obs)


def plot_forest_permodel(pairs, metric, output_path,
                         title_prefix="Stratified-by-Benchmark"):
    pair_labels = [p for p in PAIR_ORDER if p in pairs]
    n_pairs = len(pair_labels)
    if n_pairs == 0:
        return

    # Load bootstrap Spearman cross-model results
    spearman_json_path = Path(__file__).parent / "bootstrap_spearman_results.json"
    spearman_data = {}
    if spearman_json_path.exists():
        with open(spearman_json_path, "r", encoding="utf-8") as f:
            _spearman_all = json.load(f)
        spearman_data = _spearman_all.get("per_pair_cross_model", {})

    per_key   = f"per_model_{metric}"
    cross_key = f"cross_model_{metric}"
    obs_key   = f"observed_{metric}"

    n_m = len(MODEL_ORDER)
    n_lines = n_m + 1
    step = 0.05
    offsets_all = [(i - (n_lines - 1) / 2) * step for i in range(n_lines)]
    model_offsets = offsets_all[:n_m]
    mean_offset   = offsets_all[n_m]
    row_height = n_lines * step + 0.20

    fig, ax = plt.subplots(figsize=(9, 4.8))
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
                capsize=2, markersize=5, linewidth=1.2, alpha=0.85,
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
            markeredgecolor="black", markeredgewidth=0.7,
            capsize=3, markersize=7, linewidth=1.8, zorder=5,
            label="Cross-model mean" if pi == 0 else None,
        )

    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.35, zorder=1)

    ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    pad = 0.12 * (xmax - xmin)
    ax.set_xlim(xmin, xmax + pad)
    xmin, xmax = ax.get_xlim()
    x_n = xmax - 0.01 * (xmax - xmin)

    metric_key = "D" if metric == "D" else "C12"
    for pi, pl in enumerate(pair_labels):
        n_q = pairs[pl]["n_questions"]
        disp_name = PAIR_DISPLAY.get(pl, pl)
        rho_mean = None
        rho_std = None
        if metric_key in spearman_data:
            pair_spearman = spearman_data[metric_key].get(disp_name, {})
            rho_mean = pair_spearman.get("cross_model_mean_rho")
            rho_std = pair_spearman.get("cross_model_std_rho")

        ann_lines = [f"$n = {n_q}$"]
        if rho_mean is not None:
            ann_lines.append(r"$\bar{\rho}_s = " + f"{rho_mean:.3f}$")
        if rho_std is not None:
            ann_lines.append(r"$\mathrm{std}(\rho_s) = " + f"{rho_std:.3f}$")
        ann_text = "\n".join(ann_lines)
        ax.text(x_n, pi * row_height, ann_text, fontsize=8,
                fontweight="bold", va="center", ha="right", color="#333")

    y_labels = []
    for pl in pair_labels:
        comp = pairs[pl]["comparison"]
        t_full = MOD_FULL.get(comp["target"], comp["target"])
        r_full = MOD_FULL.get(comp["reference"], comp["reference"])
        if metric == "D":
            y_labels.append(
                rf"$D_{{\mathrm{{{t_full},{r_full}}}}} = S_{{\mathrm{{{t_full}}}}} - S_{{\mathrm{{{r_full}}}}}$"
            )
        else:
            disp = PAIR_DISPLAY.get(pl, pl).replace(" vs ", " + ")
            y_labels.append(disp)

    y_positions = [pi * row_height for pi in range(n_pairs)]
    ax.set_yticks(y_positions)
    ytick_fs = 12.5 if metric == "D" else 11
    ax.set_yticklabels(y_labels, fontsize=ytick_fs, rotation=0, ha="right")
    ax.set_ylim(y_positions[-1] + row_height / 2,
                y_positions[0] - row_height / 2)

    xlabel = (r"$D_{S_{m_2},S_{m_1}}$" if metric == "D"
              else r"$C_{\{m_1,m_2\}}$")
    ax.set_xlabel(xlabel, fontsize=14, labelpad=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.grid(axis="x", alpha=0.12)
    ax.tick_params(axis="x", labelsize=10)

    legend_x = 0.42 if metric == "D" else 0.5
    ax.legend(fontsize=7.5, framealpha=0.9,
              loc="lower center", bbox_to_anchor=(legend_x, -0.24),
              ncol=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_benchmark_twopanel(all_results, metric, output_path):
    """
    Two-panel figure (one per benchmark) in the same style as
    plot_consistency_compact.plot_forest_permodel_twopanel.

    Per-model coloured points + CIs and cross-model mean diamond.
    Shared y-axis on the left.  No n=X labels.
    """
    # Discover benchmarks present in per_benchmark
    all_benches = set()
    for pd in all_results.values():
        all_benches |= set(pd.get("per_benchmark", {}).keys())
    benchmarks = sorted(all_benches)
    n_bench = len(benchmarks)
    if n_bench == 0:
        return

    pair_labels = [p for p in PAIR_ORDER if p in all_results]
    n_pairs = len(pair_labels)
    if n_pairs == 0:
        return

    per_key   = f"per_model_{metric}"
    cross_key = f"cross_model_{metric}"
    obs_key   = f"observed_{metric}"
    n_m = len(MODEL_ORDER)

    n_lines = n_m + 1          # 5 models + 1 cross-model mean
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
                pb = all_results[pl].get("per_benchmark", {})
                if bench not in pb:
                    continue
                bdata = pb[bench]
                pm = bdata.get(per_key, {})
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

        # ── cross-model mean diamond ─────────────────────────────────
        for pi, pl in enumerate(pair_labels):
            pb = all_results[pl].get("per_benchmark", {})
            if bench not in pb:
                continue
            cross = pb[bench].get(cross_key, {})
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

        # ── cosmetics ───────────────────────────────────────────────
        xlabel = ("D (contribution difference)" if metric == "D"
                  else r"$C_{12}$ (cooperation)")
        ax.set_xlabel(xlabel, fontsize=17)
        ax.set_title(bench, fontsize=18, fontweight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.grid(axis="x", alpha=0.12)
        ax.tick_params(axis="x", labelsize=13)

    # ── shared y-axis labels (left only) ────────────────────────────
    y_labels = []
    for pl in pair_labels:
        disp = PAIR_DISPLAY.get(pl, pl).replace(" vs ", " + ")
        comp = all_results[pl]["comparison"]
        t_full = MOD_FULL.get(comp["target"], comp["target"])
        r_full = MOD_FULL.get(comp["reference"], comp["reference"])
        if metric == "D":
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

    met_label = "Contribution" if metric == "D" else "Cooperation"
    fig.suptitle(f"Per-Benchmark {met_label} Analysis",
                 fontsize=19, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_compact_pooled(pairs, output_path):
    """
    Compact single figure: D (left) and C12 (right) side-by-side.
    Shows cross-model mean diamond + per-model coloured points on a tight grid.
    Much shorter than the tall single-metric plots.
    """
    pair_labels = [p for p in PAIR_ORDER if p in pairs]
    n_pairs = len(pair_labels)
    if n_pairs == 0:
        return

    # Load bootstrap Spearman cross-model results
    spearman_json_path = Path(__file__).parent / "bootstrap_spearman_results.json"
    spearman_data = {}
    if spearman_json_path.exists():
        with open(spearman_json_path, "r", encoding="utf-8") as f:
            _spearman_all = json.load(f)
        spearman_data = _spearman_all.get("per_pair_cross_model", {})

    n_m = len(MODEL_ORDER)
    n_lines = n_m + 1
    step = 0.08
    offsets_all = [(i - (n_lines - 1) / 2) * step for i in range(n_lines)]
    model_offsets = offsets_all[:n_m]
    mean_offset   = offsets_all[n_m]
    row_height = n_lines * step + 0.20

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, row_height * n_pairs + 1.8),
        sharey=True,
    )
    fig.patch.set_facecolor("white")

    for ax_i, metric in enumerate(("D", "C12")):
        ax = axes[ax_i]
        ax.set_facecolor("white")
        is_left = (ax_i == 0)

        per_key   = f"per_model_{metric}"
        cross_key = f"cross_model_{metric}"
        obs_key   = f"observed_{metric}"

        # alternating row shading
        for pi in range(n_pairs):
            yc = pi * row_height
            if pi % 2 == 1:
                ax.axhspan(yc - row_height / 2, yc + row_height / 2,
                           color="#F5F5F5", zorder=0)

        # per-model points
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
                    capsize=1.5, markersize=5, linewidth=1.0, alpha=0.85,
                    label=MODEL_DISPLAY.get(model, model) if is_left else None,
                    zorder=3,
                )

        # cross-model mean diamond
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
                markeredgecolor="black", markeredgewidth=0.7,
                capsize=3, markersize=7, linewidth=1.6, zorder=5,
                label="Cross-model mean" if (is_left and pi == 0) else None,
            )

        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.35, zorder=1)

        xlabel = "D (contribution difference)" if metric == "D" else r"$C_{12}$ (cooperation)"
        ax.set_xlabel(xlabel, fontsize=12)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.grid(axis="x", alpha=0.12)
        ax.tick_params(axis="x", labelsize=10)

        # Add n, rho_s, std annotations per pair
        ax.autoscale_view()
        xmin_a, xmax_a = ax.get_xlim()
        pad_a = 0.06 * (xmax_a - xmin_a)
        ax.set_xlim(xmin_a, xmax_a + pad_a)
        xmin_a, xmax_a = ax.get_xlim()
        x_ann = xmax_a - 0.01 * (xmax_a - xmin_a)

        metric_key = "D" if metric == "D" else "C12"
        for pi, pl in enumerate(pair_labels):
            n_q = pairs[pl]["n_questions"]
            disp_name = PAIR_DISPLAY.get(pl, pl)
            rho_mean = None
            rho_std = None
            if metric_key in spearman_data:
                pair_spearman = spearman_data[metric_key].get(disp_name, {})
                rho_mean = pair_spearman.get("cross_model_mean_rho")
                rho_std = pair_spearman.get("cross_model_std_rho")

            ann_lines = [f"$n = {n_q}$"]
            if rho_mean is not None:
                ann_lines.append(r"$\bar{\rho}_s = " + f"{rho_mean:.3f}$")
            if rho_std is not None:
                ann_lines.append(r"$\mathrm{std}(\rho_s) = " + f"{rho_std:.3f}$")
            ann_text = "\n".join(ann_lines)
            ax.text(x_ann, pi * row_height, ann_text, fontsize=9,
                    fontweight="bold", va="center", ha="right", color="#333")

    # shared y-axis labels
    y_labels = []
    for pl in pair_labels:
        disp = PAIR_DISPLAY.get(pl, pl)
        comp = pairs[pl]["comparison"]
        t_full = MOD_FULL.get(comp["target"], comp["target"])
        r_full = MOD_FULL.get(comp["reference"], comp["reference"])
        y_labels.append(f"{disp}\nD = S({t_full}) \u2212 S({r_full})")

    y_positions = [pi * row_height for pi in range(n_pairs)]
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=10, rotation=15, ha="right")
    axes[0].set_ylim(y_positions[-1] + row_height / 2,
                     y_positions[0] - row_height / 2)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=min(7, len(handles)),
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_compact_per_benchmark(all_results, output_path):
    """
    Compact single figure: D (left) and C12 (right) for the per-benchmark
    two-panel layout.  Each panel is itself split into two sub-panels
    (one per benchmark).
    """
    all_benches = set()
    for pd in all_results.values():
        all_benches |= set(pd.get("per_benchmark", {}).keys())
    benchmarks = sorted(all_benches)
    n_bench = len(benchmarks)
    if n_bench == 0:
        return

    pair_labels = [p for p in PAIR_ORDER if p in all_results]
    n_pairs = len(pair_labels)
    if n_pairs == 0:
        return

    n_m = len(MODEL_ORDER)
    n_lines = n_m + 1
    step = 0.08
    offsets_all = [(i - (n_lines - 1) / 2) * step for i in range(n_lines)]
    model_offsets = offsets_all[:n_m]
    mean_offset   = offsets_all[n_m]
    row_height = n_lines * step + 0.20

    # n_bench rows × 2 metric columns
    fig, axes = plt.subplots(
        n_bench, 2,
        figsize=(14, (row_height * n_pairs + 1.2) * n_bench + 1.0),
        sharey=True,
    )
    if n_bench == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for bi, bench in enumerate(benchmarks):
        for mi_ax, metric in enumerate(("D", "C12")):
            ax = axes[bi][mi_ax]
            ax.set_facecolor("white")
            is_first = (bi == 0 and mi_ax == 0)

            per_key   = f"per_model_{metric}"
            cross_key = f"cross_model_{metric}"
            obs_key   = f"observed_{metric}"

            for pi in range(n_pairs):
                yc = pi * row_height
                if pi % 2 == 1:
                    ax.axhspan(yc - row_height / 2, yc + row_height / 2,
                               color="#F5F5F5", zorder=0)

            for mi, model in enumerate(MODEL_ORDER):
                xs, ys, elo, ehi = [], [], [], []
                for pi, pl in enumerate(pair_labels):
                    pb = all_results[pl].get("per_benchmark", {})
                    if bench not in pb:
                        continue
                    pm = pb[bench].get(per_key, {})
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
                        capsize=1.5, markersize=5, linewidth=1.0, alpha=0.85,
                        label=MODEL_DISPLAY.get(model, model) if is_first else None,
                        zorder=3,
                    )

            # cross-model mean diamond
            for pi, pl in enumerate(pair_labels):
                pb = all_results[pl].get("per_benchmark", {})
                if bench not in pb:
                    continue
                cross = pb[bench].get(cross_key, {})
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
                    markeredgecolor="black", markeredgewidth=0.7,
                    capsize=3, markersize=7, linewidth=1.6, zorder=5,
                    label="Cross-model mean" if (is_first and pi == 0) else None,
                )

            ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.35, zorder=1)

            met_label = "Contribution" if metric == "D" else "Cooperation"
            xlabel = "D (contribution diff.)" if metric == "D" else r"$C_{12}$ (cooperation)"
            ax.set_xlabel(xlabel, fontsize=10)
            # Column title on top row only, row label on left column only
            if bi == 0:
                ax.set_title(met_label, fontsize=13, fontweight="bold")
            if mi_ax == 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(bench, fontsize=12, fontweight="bold", rotation=270,
                              labelpad=15)

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_linewidth(0.5)
            ax.grid(axis="x", alpha=0.12)
            ax.tick_params(axis="x", labelsize=9)

    # shared y-axis labels (leftmost column)
    y_labels = []
    for pl in pair_labels:
        disp = PAIR_DISPLAY.get(pl, pl)
        comp = all_results[pl]["comparison"]
        t_full = MOD_FULL.get(comp["target"], comp["target"])
        r_full = MOD_FULL.get(comp["reference"], comp["reference"])
        y_labels.append(f"{disp}\nD = S({t_full}) \u2212 S({r_full})")

    y_positions = [pi * row_height for pi in range(n_pairs)]
    axes[0][0].set_yticks(y_positions)
    axes[0][0].set_yticklabels(y_labels, fontsize=9, rotation=15, ha="right")
    axes[0][0].set_ylim(y_positions[-1] + row_height / 2,
                        y_positions[0] - row_height / 2)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=min(7, len(handles)),
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle("Per-Benchmark Contribution & Cooperation Analysis",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Handle nested JSON (per_pair key)
    all_results = raw.get("per_pair", raw)

    # Pooled forest plots (tall, one per metric)
    for metric in ("D", "C12"):
        path = OUTPUT_DIR / f"stratified_by_benchmark_forest_{metric}_permodel.png"
        plot_forest_permodel(all_results, metric, path,
                             title_prefix="Stratified-by-Benchmark Bootstrap")

    # Per-benchmark two-panel forest plots (tall, one per metric)
    for metric in ("D", "C12"):
        path = OUTPUT_DIR / f"per_benchmark_forest_{metric}_permodel.png"
        plot_per_benchmark_twopanel(all_results, metric, path)

    # Compact versions (D + C12 side-by-side)
    plot_compact_pooled(all_results, OUTPUT_DIR / "pooled_compact.png")
    plot_compact_per_benchmark(all_results, OUTPUT_DIR / "per_benchmark_compact.png")

    print("Done.")


if __name__ == "__main__":
    main()
