#!/usr/bin/env python3
"""
Stratified Question-Level Bootstrap (Independence Assumption)
=============================================================

Pools ALL questions from ALL benchmarks into one set.  Strata are the
**modality pairs** (image+layout, image+text, …).

A single joint bootstrap is run: every replicate resamples independently
within each modality-pair stratum (preserving the exact pair counts),
computes per-pair metrics, and also computes aggregate cross-pair
statistics.  Questions are treated as independent (no document clustering).
"""

import csv
import json
import sys
import time
import numpy as np
from pathlib import Path

# ── shared infrastructure ────────────────────────────────────────────
_stat_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_stat_dir))
sys.path.insert(0, str(_stat_dir.parent))
sys.path.insert(0, str(_stat_dir / "cluster_bootstrap"))

from model_pairwise_bootstrap import (
    v0_for_idx, compute_shape,
    MODEL_NAMES, MODEL_DISPLAY, _ser,
)
from cluster_bootstrap.cluster_bootstrap_consistency import (
    _pci, _d_indices,
    _cross_model_summary, _build_per_model,
    _interpret, _conclusion_D, _conclusion_C12,
)
from cluster_bootstrap.pooled_cluster_bootstrap import (
    pool_benchmarks, precompute_v0_lookup_pooled,
)
from mmlongbench.eval.eval_score import eval_score as mmlong_eval_score
from longdocurl.utils.utils_score_v3 import eval_score as ldu_eval_score


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

N_BOOTSTRAP = 100_000
SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# Joint stratified bootstrap (strata = modality pairs)
# ═══════════════════════════════════════════════════════════════════════

def run_joint_stratified_bootstrap(pooled_data, models,
                                   n_boot=N_BOOTSTRAP, seed=SEED):
    """
    Single joint bootstrap over all modality pairs.

    For each replicate, questions are resampled with replacement
    *within* each modality-pair stratum, so pair counts stay fixed.

    Returns
    -------
    per_pair : dict
        pair_label → {observed, boot_D, boot_C12, cross_D, cross_C12}
    agg_D    : ndarray (n_boot,)   mean-over-pairs D per replicate
    agg_C12  : ndarray (n_boot,)   mean-over-pairs C12 per replicate
    """
    rng = np.random.default_rng(seed)
    n_m = len(models)
    pair_labels = list(pooled_data.keys())

    # ── Pre-compute V0 and observed for every pair ───────────────────
    pair_ctx = {}
    for pl in pair_labels:
        pd = pooled_data[pl]
        n = pd["n_questions"]
        t_i, r_i = _d_indices(pd["m1_name"], pd["m2_name"], pd["comparison"])
        print(f"    {pl}: precomputing V0 ({n} questions) …")
        v0_mat, q_col = precompute_v0_lookup_pooled(pd["q_meta"],
                                                     pd["scorer_fns"])
        all_idx = np.arange(n)
        v0_obs = v0_for_idx(all_idx, v0_mat, q_col)
        observed = {}
        for m in models:
            observed[m] = compute_shape(
                float(pd["scores"][m]["both"].mean()),
                float(pd["scores"][m]["m1_only"].mean()),
                float(pd["scores"][m]["m2_only"].mean()),
                v0_obs,
            )
        pair_ctx[pl] = {
            "n": n, "t_i": t_i, "r_i": r_i,
            "v0_mat": v0_mat, "q_col": q_col,
            "scores": pd["scores"],
            "observed": observed,
        }

    # ── Allocate bootstrap arrays ────────────────────────────────────
    per_pair_boot = {}
    for pl in pair_labels:
        per_pair_boot[pl] = {
            "boot_D":   {m: np.empty(n_boot) for m in models},
            "boot_C12": {m: np.empty(n_boot) for m in models},
            "mat_D":    np.empty((n_boot, n_m)),
            "mat_C12":  np.empty((n_boot, n_m)),
        }
    # Cross-pair aggregates (mean-over-pairs per replicate, per model)
    agg_D_mat   = np.empty((n_boot, n_m))    # mean D  across pairs
    agg_C12_mat = np.empty((n_boot, n_m))    # mean C12 across pairs

    strata_info = ", ".join(f"{pl}: {pair_ctx[pl]['n']}" for pl in pair_labels)
    total_q = sum(pair_ctx[pl]["n"] for pl in pair_labels)
    print(f"\n    Running {n_boot:,} joint stratified bootstrap replicates")
    print(f"    Total questions: {total_q}  Strata (modality pairs): {strata_info}")

    t0 = time.time()
    for b in range(n_boot):
        # Per-pair: resample within each stratum
        for pi, pl in enumerate(pair_labels):
            ctx = pair_ctx[pl]
            n = ctx["n"]
            idx = rng.choice(n, size=n, replace=True)
            v0_b = v0_for_idx(idx, ctx["v0_mat"], ctx["q_col"])
            ppb = per_pair_boot[pl]
            for mi, m in enumerate(models):
                V12 = float(ctx["scores"][m]["both"][idx].mean())
                V1  = float(ctx["scores"][m]["m1_only"][idx].mean())
                V2  = float(ctx["scores"][m]["m2_only"][idx].mean())
                sh  = compute_shape(V12, V1, V2, v0_b)
                d_val = sh[ctx["t_i"]] - sh[ctx["r_i"]]
                ppb["boot_D"][m][b]   = d_val
                ppb["boot_C12"][m][b] = sh[2]
                ppb["mat_D"][b, mi]   = d_val
                ppb["mat_C12"][b, mi] = sh[2]

        # Cross-pair aggregate: mean over pairs for each model
        for mi, m in enumerate(models):
            agg_D_mat[b, mi]   = np.nanmean([per_pair_boot[pl]["mat_D"][b, mi]
                                              for pl in pair_labels])
            agg_C12_mat[b, mi] = np.nanmean([per_pair_boot[pl]["mat_C12"][b, mi]
                                              for pl in pair_labels])

        if (b + 1) % 20_000 == 0:
            print(f"      {b+1:,}/{n_boot:,}  ({time.time()-t0:.1f}s)")
    print(f"    Done ({time.time()-t0:.1f}s)")

    # ── Assemble per-pair results ────────────────────────────────────
    per_pair = {}
    for pl in pair_labels:
        ppb = per_pair_boot[pl]
        per_pair[pl] = {
            "observed":  pair_ctx[pl]["observed"],
            "boot_D":    ppb["boot_D"],
            "boot_C12":  ppb["boot_C12"],
            "cross_D":   _cross_model_summary(ppb["mat_D"], n_m),
            "cross_C12": _cross_model_summary(ppb["mat_C12"], n_m),
        }

    return per_pair, agg_D_mat, agg_C12_mat


def _aggregate_cross_pair(agg_mat, n_m):
    """
    From (n_boot, n_models) of mean-over-pairs values, compute
    a cross-model mean CI (average over models of average over pairs).
    """
    valid = ~np.isnan(agg_mat).any(axis=1)
    vals = agg_mat[valid]
    n_v = int(valid.sum())
    if n_v == 0:
        return {"n_valid_replicates": 0}
    mean_a = vals.mean(axis=1)          # mean over models per replicate
    lo, hi = _pci(mean_a)
    return {
        "n_valid_replicates": n_v,
        "mean": float(mean_a.mean()),
        "ci_lo": lo, "ci_hi": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


# ═══════════════════════════════════════════════════════════════════════
# Report printer
# ═══════════════════════════════════════════════════════════════════════

def print_report(all_results, agg_results=None, file=None):
    out = file or sys.stdout
    def pr(*a, **kw):
        print(*a, **kw, file=out)

    pr(f"\n{'='*100}")
    pr(f"STRATIFIED QUESTION-LEVEL BOOTSTRAP  (strata = modality pairs)")
    pr(f"{'='*100}")

    for pair_label, pd in all_results.items():
        comp = pd["comparison"]
        n_q  = pd["n_questions"]
        bc   = pd.get("benchmark_counts", {})
        bc_str = ", ".join(f"{k}: {v}" for k, v in bc.items())
        pr(f"\n  {'─'*90}")
        pr(f"  Modality pair: {pair_label}")
        pr(f"  Total questions: {n_q}  (from: {bc_str})")
        pr(f"  Comparison: D = S_{comp['target']} − S_{comp['reference']}")
        pr(f"  {'─'*90}")

        pr(f"\n  Per-model D  (S_{comp['target']} − S_{comp['reference']}):")
        pr(f"  {'Model':<20s} {'Obs D':>10s} {'Mean':>10s} {'Std':>10s} "
           f"{'95% CI':>26s} {'Excl 0?':>8s} {'Dir':>10s}")
        pr(f"  {'-'*94}")
        for m in MODEL_NAMES:
            r = pd["per_model_D"].get(m)
            if not r:
                continue
            dn = MODEL_DISPLAY.get(m, m)
            ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
            ex = "YES" if r["ci_excludes_zero"] else "no"
            pr(f"  {dn:<20s} {r['observed_D']:>10.4f} {r['bootstrap_mean']:>10.4f} "
               f"{r['bootstrap_std']:>10.4f} {ci:>26s} {ex:>8s} {r['direction']:>10s}")

        cd = pd["cross_model_D"]
        pr(f"\n  Cross-model summary (D):")
        if cd.get("n_valid_replicates", 0) > 0:
            ma = cd["mean_across_models"]
            md = cd["median_across_models"]
            pr(f"    Mean  D across models: {ma['mean']:.4f}  "
               f"95% CI [{ma['ci_lo']:.4f}, {ma['ci_hi']:.4f}]  "
               f"excl 0: {'YES' if ma['ci_excludes_zero'] else 'no'}")
            pr(f"    Median D across models: {md['mean']:.4f}  "
               f"95% CI [{md['ci_lo']:.4f}, {md['ci_hi']:.4f}]  "
               f"excl 0: {'YES' if md['ci_excludes_zero'] else 'no'}")
            pr(f"    Prop replicates all models D > 0: {cd['prop_all_positive']:.4f}")
            pr(f"    Prop replicates all models D < 0: {cd['prop_all_negative']:.4f}")
        else:
            pr("    (no valid replicates)")

        pr(f"\n  Strength: {pd['D_strength']}")
        pr(f"  Detail:   {pd['D_detail']}")
        pr(f"  >> {pd['D_conclusion']}")

        pr(f"\n  Per-model C12:")
        pr(f"  {'Model':<20s} {'Obs C12':>10s} {'Mean':>10s} {'Std':>10s} "
           f"{'95% CI':>26s} {'Excl 0?':>8s} {'Dir':>10s}")
        pr(f"  {'-'*94}")
        for m in MODEL_NAMES:
            r = pd["per_model_C12"].get(m)
            if not r:
                continue
            dn = MODEL_DISPLAY.get(m, m)
            ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
            ex = "YES" if r["ci_excludes_zero"] else "no"
            pr(f"  {dn:<20s} {r['observed_C12']:>10.4f} {r['bootstrap_mean']:>10.4f} "
               f"{r['bootstrap_std']:>10.4f} {ci:>26s} {ex:>8s} {r['direction']:>10s}")

        cc = pd["cross_model_C12"]
        pr(f"\n  Cross-model summary (C12):")
        if cc.get("n_valid_replicates", 0) > 0:
            ma = cc["mean_across_models"]
            md = cc["median_across_models"]
            pr(f"    Mean  C12 across models: {ma['mean']:.4f}  "
               f"95% CI [{ma['ci_lo']:.4f}, {ma['ci_hi']:.4f}]  "
               f"excl 0: {'YES' if ma['ci_excludes_zero'] else 'no'}")
            pr(f"    Median C12 across models: {md['mean']:.4f}  "
               f"95% CI [{md['ci_lo']:.4f}, {md['ci_hi']:.4f}]  "
               f"excl 0: {'YES' if md['ci_excludes_zero'] else 'no'}")
            pr(f"    Prop replicates all models C12 > 0: {cc['prop_all_positive']:.4f}")
            pr(f"    Prop replicates all models C12 < 0: {cc['prop_all_negative']:.4f}")
        else:
            pr("    (no valid replicates)")

        pr(f"\n  C12 strength: {pd['C12_strength']}")
        pr(f"  C12 detail:   {pd['C12_detail']}")
        pr(f"  >> {pd['C12_conclusion']}")

    # ── Aggregate across all pairs ───────────────────────────────────
    if agg_results:
        pr(f"\n\n{'='*100}")
        pr(f"AGGREGATE ACROSS ALL MODALITY PAIRS  (mean over pairs, then mean over models)")
        pr(f"{'='*100}")
        ad = agg_results.get("agg_D")
        ac = agg_results.get("agg_C12")
        if ad and ad.get("n_valid_replicates", 0) > 0:
            pr(f"  D  (mean across pairs & models): {ad['mean']:.4f}  "
               f"95% CI [{ad['ci_lo']:.4f}, {ad['ci_hi']:.4f}]  "
               f"excl 0: {'YES' if ad['ci_excludes_zero'] else 'no'}")
        if ac and ac.get("n_valid_replicates", 0) > 0:
            pr(f"  C12 (mean across pairs & models): {ac['mean']:.4f}  "
               f"95% CI [{ac['ci_lo']:.4f}, {ac['ci_hi']:.4f}]  "
               f"excl 0: {'YES' if ac['ci_excludes_zero'] else 'no'}")


# ═══════════════════════════════════════════════════════════════════════
# CSV writers
# ═══════════════════════════════════════════════════════════════════════

def write_csv_per_model(all_results, path):
    rows = []
    for pl, pd in all_results.items():
        comp = pd["comparison"]
        for m in MODEL_NAMES:
            rd = pd["per_model_D"].get(m)
            rc = pd["per_model_C12"].get(m)
            if not rd or not rc:
                continue
            rows.append({
                "benchmark": "Pooled-Stratified-ByPair",
                "modality_pair": pl,
                "comparison": f"S_{comp['target']} - S_{comp['reference']}",
                "model": MODEL_DISPLAY.get(m, m),
                "n_questions": pd["n_questions"],
                "observed_D": rd["observed_D"],
                "D_mean": rd["bootstrap_mean"],
                "D_std": rd["bootstrap_std"],
                "D_ci_lo": rd["ci_lo"],
                "D_ci_hi": rd["ci_hi"],
                "D_ci_excludes_zero": rd["ci_excludes_zero"],
                "D_direction": rd["direction"],
                "observed_C12": rc["observed_C12"],
                "C12_mean": rc["bootstrap_mean"],
                "C12_std": rc["bootstrap_std"],
                "C12_ci_lo": rc["ci_lo"],
                "C12_ci_hi": rc["ci_hi"],
                "C12_ci_excludes_zero": rc["ci_excludes_zero"],
                "C12_direction": rc["direction"],
            })
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"  CSV (per-model): {path}")


def write_csv_cross_model(all_results, agg_results, path):
    rows = []
    for pl, pd in all_results.items():
        comp = pd["comparison"]
        cd = pd["cross_model_D"]
        cc = pd["cross_model_C12"]
        row = {
            "benchmark": "Pooled-Stratified-ByPair",
            "modality_pair": pl,
            "n_questions": pd["n_questions"],
            "comparison": f"S_{comp['target']} - S_{comp['reference']}",
        }
        if cd.get("n_valid_replicates", 0) > 0:
            row.update({
                "D_mean_across": cd["mean_across_models"]["mean"],
                "D_mean_ci_lo": cd["mean_across_models"]["ci_lo"],
                "D_mean_ci_hi": cd["mean_across_models"]["ci_hi"],
                "D_mean_excl_zero": cd["mean_across_models"]["ci_excludes_zero"],
            })
        if cc.get("n_valid_replicates", 0) > 0:
            row.update({
                "C12_mean_across": cc["mean_across_models"]["mean"],
                "C12_mean_ci_lo": cc["mean_across_models"]["ci_lo"],
                "C12_mean_ci_hi": cc["mean_across_models"]["ci_hi"],
                "C12_mean_excl_zero": cc["mean_across_models"]["ci_excludes_zero"],
            })
        row.update({
            "D_strength": pd["D_strength"],
            "D_conclusion": pd["D_conclusion"],
            "C12_strength": pd["C12_strength"],
            "C12_conclusion": pd["C12_conclusion"],
        })
        rows.append(row)
    # Aggregate row
    if agg_results:
        agg_row = {
            "benchmark": "Pooled-Stratified-ByPair",
            "modality_pair": "ALL (aggregate)",
            "n_questions": sum(pd["n_questions"] for pd in all_results.values()),
            "comparison": "mean across pairs",
        }
        ad = agg_results.get("agg_D", {})
        ac = agg_results.get("agg_C12", {})
        if ad.get("n_valid_replicates", 0) > 0:
            agg_row.update({
                "D_mean_across": ad["mean"],
                "D_mean_ci_lo": ad["ci_lo"],
                "D_mean_ci_hi": ad["ci_hi"],
                "D_mean_excl_zero": ad["ci_excludes_zero"],
            })
        if ac.get("n_valid_replicates", 0) > 0:
            agg_row.update({
                "C12_mean_across": ac["mean"],
                "C12_mean_ci_lo": ac["ci_lo"],
                "C12_mean_ci_hi": ac["ci_hi"],
                "C12_mean_excl_zero": ac["ci_excludes_zero"],
            })
        rows.append(agg_row)
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"  CSV (cross-model): {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    repo    = Path(__file__).parent.parent.parent
    out_dir = Path(__file__).parent

    benchmark_configs = {
        "LongDocURL":       (repo / "results_longdocurl",   ldu_eval_score),
        "MMLongBench-Doc":  (repo / "results_mmlongbench",  mmlong_eval_score),
    }

    print("Pooling benchmarks …")
    pooled_data, avail_models = pool_benchmarks(benchmark_configs, MODEL_NAMES)
    n_models = len(avail_models)

    # ── Run one joint stratified bootstrap ────────────────────────────
    per_pair_boot, agg_D_mat, agg_C12_mat = \
        run_joint_stratified_bootstrap(pooled_data, avail_models)

    # ── Build per-pair result dicts ──────────────────────────────────
    all_results = {}
    for pair_label, pdata in pooled_data.items():
        comparison = pdata["comparison"]
        m1_name = pdata["m1_name"]
        m2_name = pdata["m2_name"]

        ppb = per_pair_boot[pair_label]
        t_i, r_i = _d_indices(m1_name, m2_name, comparison)

        pm_D = _build_per_model(
            ppb["observed"], ppb["boot_D"], avail_models, "D",
            lambda sh, _ti=t_i, _ri=r_i: sh[_ti] - sh[_ri],
        )
        pm_C12 = _build_per_model(
            ppb["observed"], ppb["boot_C12"], avail_models, "C12",
            lambda sh: sh[2],
        )

        d_strength, d_detail = _interpret(pm_D, ppb["cross_D"], n_models, "D")
        d_conclusion = _conclusion_D(comparison, d_strength, pm_D, n_models)

        c12_strength, c12_detail = _interpret(pm_C12, ppb["cross_C12"], n_models, "C12")
        c12_conclusion = _conclusion_C12(c12_strength, pm_C12, n_models)

        all_results[pair_label] = {
            "n_questions": pdata["n_questions"],
            "benchmark_counts": pdata["benchmark_counts"],
            "modalities": [m1_name, m2_name],
            "comparison": comparison,
            "per_model_D": pm_D,
            "cross_model_D": ppb["cross_D"],
            "D_strength": d_strength,
            "D_detail": d_detail,
            "D_conclusion": d_conclusion,
            "per_model_C12": pm_C12,
            "cross_model_C12": ppb["cross_C12"],
            "C12_strength": c12_strength,
            "C12_detail": c12_detail,
            "C12_conclusion": c12_conclusion,
        }

    # ── Cross-pair aggregates ────────────────────────────────────────
    agg_results = {
        "agg_D":   _aggregate_cross_pair(agg_D_mat, n_models),
        "agg_C12": _aggregate_cross_pair(agg_C12_mat, n_models),
    }

    # ── Save outputs ─────────────────────────────────────────────────
    save_payload = {"per_pair": all_results, "aggregate": agg_results}
    json_path = out_dir / "stratified_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2, default=_ser)
    print(f"\n  JSON: {json_path}")

    txt_path = out_dir / "stratified_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        print_report(all_results, agg_results, file=f)
    print(f"  Report: {txt_path}")

    write_csv_per_model(all_results, out_dir / "stratified_per_model.csv")
    write_csv_cross_model(all_results, agg_results,
                          out_dir / "stratified_cross_model.csv")

    print_report(all_results, agg_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
