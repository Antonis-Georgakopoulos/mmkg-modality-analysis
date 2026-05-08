#!/usr/bin/env python3
"""
Bootstrap Spearman Rank Correlation between Models
====================================================

Runs 100,000 stratified-by-benchmark bootstrap replicates (same as the
main bootstrap script).  On EACH replicate, collects the D (or C12)
value for every pair × model, forming a 6×5 matrix, then computes the
5×5 Spearman rank-correlation between models.

This yields a *distribution* of 100,000 Spearman correlations for each
model pair, from which we report mean, std, and 95% CI.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# ── shared infrastructure ────────────────────────────────────────────
_stat_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_stat_dir))
sys.path.insert(0, str(_stat_dir.parent))
sys.path.insert(0, str(_stat_dir / "cluster_bootstrap"))

from model_pairwise_bootstrap import (
    v0_for_idx, compute_shape,
    MODEL_NAMES,
)
from cluster_bootstrap.pooled_cluster_bootstrap import (
    pool_benchmarks, precompute_v0_lookup_pooled,
)
from cluster_bootstrap.cluster_bootstrap_consistency import _d_indices
from mmlongbench.eval.eval_score import eval_score as mmlong_eval_score
from longdocurl.utils.utils_score_v3 import eval_score as ldu_eval_score


# ── Config ───────────────────────────────────────────────────────────
N_BOOTSTRAP = 100_000
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "spearman_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY_NAMES = {
    "gemma3_4b":    "Gemma3-4B",
    "gemma3_27b":   "Gemma3-27B",
    "gpt-4o-mini":  "GPT-4o-mini",
    "qwen3-vl_8b":  "Qwen3-VL-8B",
    "qwen3-vl_30b": "Qwen3-VL-30B",
}


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap + Spearman on each replicate
# ═══════════════════════════════════════════════════════════════════════

def _build_benchmark_strata(benchmark_counts):
    strata = []
    offset = 0
    for bench_name, n_bench in benchmark_counts.items():
        strata.append((bench_name, np.arange(offset, offset + n_bench)))
        offset += n_bench
    return strata


def run_bootstrap_spearman(pooled_data, models, n_boot=N_BOOTSTRAP, seed=SEED):
    """
    Returns
    -------
    spearman_D  : ndarray (n_boot, n_models, n_models)  — across-pairs correlation
    spearman_C12: ndarray (n_boot, n_models, n_models)
    per_pair_boot_D   : dict  pair_label → ndarray (n_boot, n_models)
    per_pair_boot_C12 : dict  pair_label → ndarray (n_boot, n_models)
    pair_labels : list[str]
    """
    rng = np.random.default_rng(seed)
    n_m = len(models)
    pair_labels = list(pooled_data.keys())
    n_pairs = len(pair_labels)

    # ── Pre-compute V0 and strata per pair ────────────────────────────
    pair_ctx = {}
    for pl in pair_labels:
        pd = pooled_data[pl]
        n = pd["n_questions"]
        t_i, r_i = _d_indices(pd["m1_name"], pd["m2_name"], pd["comparison"])
        strata = _build_benchmark_strata(pd["benchmark_counts"])
        print(f"  {pl}: precomputing V0 ({n} questions) …")
        v0_mat, q_col = precompute_v0_lookup_pooled(pd["q_meta"],
                                                     pd["scorer_fns"])
        pair_ctx[pl] = {
            "n": n, "t_i": t_i, "r_i": r_i,
            "v0_mat": v0_mat, "q_col": q_col,
            "scores": pd["scores"],
            "strata": strata,
        }

    # ── Allocate output arrays ────────────────────────────────────────
    # Across-pairs Spearman (existing)
    spearman_D   = np.empty((n_boot, n_m, n_m))
    spearman_C12 = np.empty((n_boot, n_m, n_m))

    # Per-pair bootstrap values (NEW: needed for within-pair correlation)
    per_pair_boot_D   = {pl: np.empty((n_boot, n_m)) for pl in pair_labels}
    per_pair_boot_C12 = {pl: np.empty((n_boot, n_m)) for pl in pair_labels}

    # Temp arrays for one replicate: (n_pairs, n_models)
    rep_D   = np.empty((n_pairs, n_m))
    rep_C12 = np.empty((n_pairs, n_m))

    print(f"\n  Running {n_boot:,} bootstrap replicates + Spearman …")
    t0 = time.time()

    for b in range(n_boot):
        # Fill rep_D, rep_C12 for this replicate
        for pi, pl in enumerate(pair_labels):
            ctx = pair_ctx[pl]
            idx_parts = [rng.choice(si, size=len(si), replace=True)
                         for _, si in ctx["strata"]]
            idx = np.concatenate(idx_parts)
            v0_b = v0_for_idx(idx, ctx["v0_mat"], ctx["q_col"])

            for mi, m in enumerate(models):
                V12 = float(ctx["scores"][m]["both"][idx].mean())
                V1  = float(ctx["scores"][m]["m1_only"][idx].mean())
                V2  = float(ctx["scores"][m]["m2_only"][idx].mean())
                sh  = compute_shape(V12, V1, V2, v0_b)
                if isinstance(sh[0], float) and np.isnan(sh[0]):
                    rep_D[pi, mi]   = np.nan
                    rep_C12[pi, mi] = np.nan
                else:
                    rep_D[pi, mi]   = sh[ctx["t_i"]] - sh[ctx["r_i"]]
                    rep_C12[pi, mi] = sh[2]

            # Store per-pair values
            per_pair_boot_D[pl][b, :]   = rep_D[pi, :]
            per_pair_boot_C12[pl][b, :] = rep_C12[pi, :]

        # Compute Spearman between all model pairs for this replicate
        for metric, rep, out in [("D", rep_D, spearman_D),
                                  ("C12", rep_C12, spearman_C12)]:
            for i in range(n_m):
                out[b, i, i] = 1.0
                for j in range(i + 1, n_m):
                    col_i = rep[:, i]
                    col_j = rep[:, j]
                    valid = ~(np.isnan(col_i) | np.isnan(col_j))
                    if valid.sum() < 3:
                        out[b, i, j] = np.nan
                        out[b, j, i] = np.nan
                    else:
                        rho, _ = spearmanr(col_i[valid], col_j[valid])
                        out[b, i, j] = rho
                        out[b, j, i] = rho

        if (b + 1) % 20_000 == 0:
            print(f"    {b+1:,}/{n_boot:,}  ({time.time()-t0:.1f}s)")

    print(f"  Done ({time.time()-t0:.1f}s)")
    return spearman_D, spearman_C12, per_pair_boot_D, per_pair_boot_C12, pair_labels


# ═══════════════════════════════════════════════════════════════════════
# Summarise & plot
# ═══════════════════════════════════════════════════════════════════════

def _pci(arr, lo=2.5, hi=97.5):
    return float(np.nanpercentile(arr, lo)), float(np.nanpercentile(arr, hi))


def summarise_and_plot(spearman_arr, models, metric, out_dir):
    """Print summary and produce heatmaps for one metric."""
    n_m = len(models)
    display = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

    print(f"\n{'='*80}")
    print(f"  {metric} — Bootstrap Distribution of Spearman Correlation")
    print(f"  ({spearman_arr.shape[0]:,} replicates, {len(models)} models, "
          f"correlation computed across 6 modality pairs per replicate)")
    print(f"{'='*80}")

    mean_mat = np.empty((n_m, n_m))
    std_mat  = np.empty((n_m, n_m))
    ci_lo_mat = np.empty((n_m, n_m))
    ci_hi_mat = np.empty((n_m, n_m))

    print(f"\n  {'Model A':<16s}  {'Model B':<16s}  {'Mean':>7s}  {'Std':>7s}  "
          f"{'95% CI':>22s}")
    print(f"  {'-'*76}")

    for i in range(n_m):
        for j in range(n_m):
            vals = spearman_arr[:, i, j]
            mean_mat[i, j] = np.nanmean(vals)
            std_mat[i, j]  = np.nanstd(vals)
            lo, hi = _pci(vals)
            ci_lo_mat[i, j] = lo
            ci_hi_mat[i, j] = hi
            if j > i:
                print(f"  {display[i]:<16s}  {display[j]:<16s}  "
                      f"{mean_mat[i,j]:>7.3f}  {std_mat[i,j]:>7.3f}  "
                      f"[{lo:.3f}, {hi:.3f}]")

    # ── Heatmap: mean Spearman ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, data, title, fmt, cmap, vmin, vmax in [
        (axes[0], mean_mat,
         f"{metric} — Mean Spearman ρ\n(over {spearman_arr.shape[0]:,} bootstrap replicates)",
         ".3f", "YlOrRd", 0.3, 1.0),
        (axes[1], std_mat,
         f"{metric} — Std of Spearman ρ\n(uncertainty across replicates)",
         ".3f", "Blues", 0.0, None),
    ]:
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(n_m):
            for j in range(n_m):
                ax.text(j, i, f"{data[i, j]:{fmt}}", ha="center", va="center",
                        fontsize=11)
        ax.set_xticks(range(n_m))
        ax.set_xticklabels(display, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(n_m))
        ax.set_yticklabels(display, fontsize=10)
        ax.set_title(title, fontsize=12, pad=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = out_dir / f"bootstrap_spearman_{metric}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # ── Save results to JSON ──────────────────────────────────────────
    results = {}
    for i in range(n_m):
        for j in range(i + 1, n_m):
            key = f"{display[i]} vs {display[j]}"
            results[key] = {
                "mean_rho": float(mean_mat[i, j]),
                "std_rho":  float(std_mat[i, j]),
                "ci_lo":    float(ci_lo_mat[i, j]),
                "ci_hi":    float(ci_hi_mat[i, j]),
            }
    return results


def compute_per_pair_cross_model_spearman(per_pair_boot, models, metric, out_dir):
    """
    For each modality pair, compute Spearman correlation between models'
    bootstrap distributions (100k values per model, same resamples).

    Reports per model-pair ρ and a cross-model mean ρ per pair.
    """
    n_m = len(models)
    display = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    pair_labels = list(per_pair_boot.keys())

    PAIR_DISPLAY = {
        "image + layout":      "Image vs Layout",
        "image + plain_text":  "Image vs Text",
        "layout + plain_text": "Layout vs Text",
        "plain_text + table":  "Text vs Table",
        "image + table":       "Image vs Table",
        "layout + table":      "Layout vs Table",
    }

    print(f"\n{'='*80}")
    print(f"  {metric} — Per-Pair Cross-Model Spearman Correlation")
    print(f"  (correlation between models' bootstrap distributions within each pair)")
    print(f"{'='*80}")

    per_pair_results = {}

    for pl in pair_labels:
        boot_mat = per_pair_boot[pl]  # (n_boot, n_models)
        pair_name = PAIR_DISPLAY.get(pl, pl)

        # Compute pairwise Spearman between model columns
        corr_mat = np.empty((n_m, n_m))
        for i in range(n_m):
            corr_mat[i, i] = 1.0
            for j in range(i + 1, n_m):
                col_i = boot_mat[:, i]
                col_j = boot_mat[:, j]
                valid = ~(np.isnan(col_i) | np.isnan(col_j))
                if valid.sum() < 3:
                    corr_mat[i, j] = np.nan
                    corr_mat[j, i] = np.nan
                else:
                    rho, _ = spearmanr(col_i[valid], col_j[valid])
                    corr_mat[i, j] = rho
                    corr_mat[j, i] = rho

        # Cross-model mean: average of upper-triangle (off-diagonal)
        upper = corr_mat[np.triu_indices(n_m, k=1)]
        cross_mean = float(np.nanmean(upper))
        cross_std  = float(np.nanstd(upper))

        print(f"\n  {pair_name} ({pl}):")
        print(f"    {'Model A':<16s}  {'Model B':<16s}  {'ρ':>8s}")
        print(f"    {'-'*46}")
        pair_model_results = {}
        for i in range(n_m):
            for j in range(i + 1, n_m):
                key = f"{display[i]} vs {display[j]}"
                pair_model_results[key] = float(corr_mat[i, j])
                print(f"    {display[i]:<16s}  {display[j]:<16s}  "
                      f"{corr_mat[i, j]:>8.4f}")
        print(f"    {'─'*46}")
        print(f"    Cross-model mean ρ: {cross_mean:.4f}  (std: {cross_std:.4f})")

        per_pair_results[pair_name] = {
            "model_pairs": pair_model_results,
            "cross_model_mean_rho": cross_mean,
            "cross_model_std_rho":  cross_std,
        }

    # ── Plot: bar chart of cross-model mean per pair ──────────────────
    pair_names = [PAIR_DISPLAY.get(pl, pl) for pl in pair_labels]
    means = [per_pair_results[pn]["cross_model_mean_rho"] for pn in pair_names]
    stds  = [per_pair_results[pn]["cross_model_std_rho"] for pn in pair_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pair_names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#D32F2F", alpha=0.8,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Cross-Model Mean Spearman ρ", fontsize=11)
    ax.set_title(f"{metric} — Cross-Model Mean Spearman ρ per Pair\n"
                 f"(how similarly models respond to bootstrap perturbation)",
                 fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    path = out_dir / f"bootstrap_spearman_{metric}_per_pair_cross_model.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # ── Plot: heatmap per pair ────────────────────────────────────────
    n_pairs = len(pair_labels)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for pi, pl in enumerate(pair_labels):
        boot_mat = per_pair_boot[pl]
        pair_name = PAIR_DISPLAY.get(pl, pl)
        ax = axes_flat[pi]

        corr_mat = np.empty((n_m, n_m))
        for i in range(n_m):
            corr_mat[i, i] = 1.0
            for j in range(i + 1, n_m):
                col_i = boot_mat[:, i]
                col_j = boot_mat[:, j]
                valid = ~(np.isnan(col_i) | np.isnan(col_j))
                rho, _ = spearmanr(col_i[valid], col_j[valid])
                corr_mat[i, j] = rho
                corr_mat[j, i] = rho

        im = ax.imshow(corr_mat, cmap="YlOrRd", vmin=0.3, vmax=1.0)
        for i in range(n_m):
            for j in range(n_m):
                ax.text(j, i, f"{corr_mat[i, j]:.2f}", ha="center",
                        va="center", fontsize=9)
        ax.set_xticks(range(n_m))
        ax.set_xticklabels(display, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(n_m))
        ax.set_yticklabels(display, fontsize=8)
        ax.set_title(pair_name, fontsize=11, pad=6)

    fig.suptitle(f"{metric} — Within-Pair Spearman ρ between Models\n"
                 f"(correlation of bootstrap distributions, 100k replicates)",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / f"bootstrap_spearman_{metric}_per_pair_heatmaps.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return per_pair_results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    repo = Path(__file__).parent.parent.parent

    benchmark_configs = {
        "LongDocURL":      (repo / "results_longdocurl",   ldu_eval_score),
        "MMLongBench-Doc": (repo / "results_mmlongbench",  mmlong_eval_score),
    }

    print("Pooling benchmarks …")
    pooled_data, avail_models = pool_benchmarks(benchmark_configs, MODEL_NAMES)

    spearman_D, spearman_C12, pp_boot_D, pp_boot_C12, pair_labels = \
        run_bootstrap_spearman(pooled_data, avail_models)

    all_results = {}

    # ── Per-pair cross-model Spearman ─────────────────────────────────
    per_pair_results = {}
    for metric, pp_boot in [("D", pp_boot_D), ("C12", pp_boot_C12)]:
        per_pair_results[metric] = compute_per_pair_cross_model_spearman(
            pp_boot, avail_models, metric, OUTPUT_DIR)

    all_results["per_pair_cross_model"] = per_pair_results

    json_path = OUTPUT_DIR / "bootstrap_spearman_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  JSON: {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
