"""
Forest plot for unstratified question-level bootstrap results.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_JSON = Path(__file__).parent / "unstratified_results.json"
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


def plot_forest_permodel(pairs, metric, output_path, title_prefix="Unstratified"):
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

    fig, ax = plt.subplots(figsize=(9, row_height * n_pairs + 2.0))
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

    ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    pad = 0.08 * (xmax - xmin)
    ax.set_xlim(xmin, xmax + pad)
    xmin, xmax = ax.get_xlim()
    x_n = xmax - 0.01 * (xmax - xmin)
    for pi, pl in enumerate(pair_labels):
        n_q = pairs[pl]["n_questions"]
        ax.text(x_n, pi * row_height, f"n={n_q}", fontsize=12,
                va="center", ha="right", color="#444", style="italic")

    y_labels = []
    for pl in pair_labels:
        disp = PAIR_DISPLAY.get(pl, pl).replace(" vs ", " + ")
        comp = pairs[pl]["comparison"]
        t_full = MOD_FULL.get(comp["target"], comp["target"])
        r_full = MOD_FULL.get(comp["reference"], comp["reference"])
        if metric == "D":
            y_labels.append(f"{disp}\nD = S({t_full}) \u2212 S({r_full})")
        else:
            y_labels.append(disp)

    y_positions = [pi * row_height for pi in range(n_pairs)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=13, rotation=15, ha="right")
    ax.set_ylim(y_positions[-1] + row_height / 2,
                y_positions[0] - row_height / 2)

    xlabel = "D (contribution difference)" if metric == "D" else r"$C_{12}$ (cooperation)"
    ax.set_xlabel(xlabel, fontsize=17)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.grid(axis="x", alpha=0.12)
    ax.tick_params(axis="x", labelsize=13)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

    met_label = "Contribution" if metric == "D" else "Cooperation"
    ax.set_title(f"{title_prefix} — Pooled Cross-Model {met_label} Analysis",
                 fontsize=15, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Handle nested JSON (per_pair key)
    all_results = raw.get("per_pair", raw)

    for metric in ("D", "C12"):
        path = OUTPUT_DIR / f"unstratified_forest_{metric}_permodel.png"
        plot_forest_permodel(all_results, metric, path,
                             title_prefix="Unstratified (Global-Pool) Bootstrap")

    print("Done.")


if __name__ == "__main__":
    main()
