#!/usr/bin/env python3
"""
Cluster (Per-Document) Bootstrap — Cross-Model Consistency Analysis
====================================================================

Identical to cross_model_consistency.py EXCEPT the bootstrap resamples
**entire documents** (with replacement) instead of individual questions.
This correctly accounts for within-document correlation (multiple
questions from the same document are not independent).

Output structure is identical to the original so the same plotting
script can be reused.
"""

import csv
import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── shared infrastructure ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_pairwise_bootstrap import (
    load_and_index,
    find_common_keys,
    build_score_arrays,
    precompute_v0_lookup,
    v0_for_idx,
    compute_shape,
    MODEL_NAMES,
    MODEL_DISPLAY,
    _ser,
)
from mmlongbench.eval.eval_score import eval_score as mmlong_eval_score
from longdocurl.utils.utils_score_v3 import eval_score as ldu_eval_score


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

N_BOOTSTRAP = 100_000
SEED = 42

COMPARISONS = {
    ("image", "layout"):      {"target": "layout",     "reference": "image"},
    ("image", "plain_text"):  {"target": "plain_text",  "reference": "image"},
    ("image", "table"):       {"target": "table",       "reference": "image"},
    ("layout", "plain_text"): {"target": "plain_text",  "reference": "layout"},
    ("layout", "table"):      {"target": "table",       "reference": "layout"},
    ("plain_text", "table"):  {"target": "table",       "reference": "plain_text"},
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _d_indices(m1_name, m2_name, comparison):
    names = {m1_name: 0, m2_name: 1}
    target = comparison["target"]
    reference = comparison["reference"]
    assert target in names
    assert reference in names
    assert target != reference
    return names[target], names[reference]


def _pci(arr, alpha=0.05):
    """Percentile CI (lo, hi) ignoring NaNs."""
    v = arr[~np.isnan(arr)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 100 * alpha / 2)), \
           float(np.percentile(v, 100 * (1 - alpha / 2)))


def _dir(val):
    if np.isnan(val):
        return "nan"
    return "positive" if val > 0 else ("negative" if val < 0 else "zero")


# ═══════════════════════════════════════════════════════════════════════
# Document-ID extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_doc_ids(all_data, mod_pair, common_keys, models):
    """
    For each common question (in order), extract the document identifier.

    LongDocURL entries have 'doc_no'; MMLongBench entries have 'doc_id'.
    Falls back to the base question ID (first element of the key tuple)
    if neither field is present.
    """
    doc_ids = []
    ref_model = models[0]
    for qk in common_keys:
        entry = all_data[ref_model][mod_pair][qk]["both"]
        did = entry.get("doc_no") or entry.get("doc_id") or qk[0]
        doc_ids.append(str(did))
    return doc_ids


def build_cluster_index(doc_ids):
    """
    Returns
    -------
    unique_docs : list[str]          sorted unique document IDs
    doc_to_qidx : dict[str, list]    doc_id → list of question indices
    """
    doc_to_qidx = defaultdict(list)
    for qi, did in enumerate(doc_ids):
        doc_to_qidx[did].append(qi)
    unique_docs = sorted(doc_to_qidx.keys())
    return unique_docs, dict(doc_to_qidx)


# ═══════════════════════════════════════════════════════════════════════
# Cluster bootstrap core
# ═══════════════════════════════════════════════════════════════════════

def run_cluster_bootstrap(scores, q_meta, scorer_fn, models,
                          comparison, m1_name, m2_name,
                          doc_ids,
                          n_boot=N_BOOTSTRAP, seed=SEED):
    """
    Cluster bootstrap: resample documents with replacement, include all
    questions from each sampled document.

    Returns the same structure as cross_model_consistency.run_consistency_bootstrap.
    """
    n = len(q_meta)
    if n == 0:
        return {}, {}, {}, {}, {}

    unique_docs, doc_to_qidx = build_cluster_index(doc_ids)
    n_docs = len(unique_docs)
    # Pre-convert to numpy arrays for speed
    doc_idx_arrays = [np.array(doc_to_qidx[d], dtype=np.intp)
                      for d in unique_docs]

    rng = np.random.default_rng(seed)
    n_m = len(models)
    t_i, r_i = _d_indices(m1_name, m2_name, comparison)

    # V0 precomputation
    print("    Precomputing V0 lookup …")
    v0_mat, q_col = precompute_v0_lookup(q_meta, scorer_fn)

    # Observed (full data)
    all_idx = np.arange(n)
    v0_obs = v0_for_idx(all_idx, v0_mat, q_col)
    observed = {}
    for m in models:
        observed[m] = compute_shape(
            float(scores[m]["both"].mean()),
            float(scores[m]["m1_only"].mean()),
            float(scores[m]["m2_only"].mean()),
            v0_obs,
        )

    # Allocate arrays
    boot_D   = {m: np.empty(n_boot) for m in models}
    boot_C12 = {m: np.empty(n_boot) for m in models}
    mat_D    = np.empty((n_boot, n_m))
    mat_C12  = np.empty((n_boot, n_m))

    print(f"    Running {n_boot:,} cluster bootstrap replicates "
          f"({n_docs} documents, {n} questions) …")
    t0 = time.time()
    for b in range(n_boot):
        # Resample documents with replacement
        sampled_docs = rng.choice(n_docs, size=n_docs, replace=True)
        idx = np.concatenate([doc_idx_arrays[d] for d in sampled_docs])

        v0_b = v0_for_idx(idx, v0_mat, q_col)
        for mi, m in enumerate(models):
            V12 = float(scores[m]["both"][idx].mean())
            V1  = float(scores[m]["m1_only"][idx].mean())
            V2  = float(scores[m]["m2_only"][idx].mean())
            sh  = compute_shape(V12, V1, V2, v0_b)
            d_val = sh[t_i] - sh[r_i]
            boot_D[m][b]   = d_val
            boot_C12[m][b] = sh[2]
            mat_D[b, mi]   = d_val
            mat_C12[b, mi] = sh[2]
        if (b + 1) % 20_000 == 0:
            print(f"      {b+1:,}/{n_boot:,}  ({time.time()-t0:.1f}s)")
    print(f"    Done ({time.time()-t0:.1f}s)")

    cross_D   = _cross_model_summary(mat_D, n_m)
    cross_C12 = _cross_model_summary(mat_C12, n_m)
    return observed, boot_D, boot_C12, cross_D, cross_C12


def _cross_model_summary(mat, n_m):
    """Aggregate (n_boot, n_models) → cross-model stats."""
    valid = ~np.isnan(mat).any(axis=1)
    vals = mat[valid]
    n_v = int(valid.sum())
    if n_v == 0:
        return {"n_valid_replicates": 0}

    mean_a  = vals.mean(axis=1)
    med_a   = np.median(vals, axis=1)
    n_pos   = (vals > 0).sum(axis=1)
    n_neg   = (vals < 0).sum(axis=1)
    all_pos = (n_pos == n_m)
    all_neg = (n_neg == n_m)

    mean_lo, mean_hi = _pci(mean_a)
    med_lo, med_hi   = _pci(med_a)

    return {
        "n_valid_replicates": n_v,
        "mean_across_models": {
            "mean": float(mean_a.mean()),
            "ci_lo": mean_lo, "ci_hi": mean_hi,
            "ci_excludes_zero": bool(mean_lo > 0 or mean_hi < 0),
        },
        "median_across_models": {
            "mean": float(med_a.mean()),
            "ci_lo": med_lo, "ci_hi": med_hi,
            "ci_excludes_zero": bool(med_lo > 0 or med_hi < 0),
        },
        "prop_all_positive": float(all_pos.mean()),
        "prop_all_negative": float(all_neg.mean()),
        "n_positive_distribution": {
            str(k): float((n_pos == k).mean()) for k in range(n_m + 1)
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Per-model summary
# ═══════════════════════════════════════════════════════════════════════

def _build_per_model(observed, boot_arr, models, value_key, idx_fn):
    out = {}
    for m in models:
        obs_val = idx_fn(observed[m])
        bv = boot_arr[m]
        v = bv[~np.isnan(bv)]
        lo, hi = _pci(bv)
        out[m] = {
            f"observed_{value_key}": float(obs_val),
            "bootstrap_mean": float(v.mean()) if len(v) else float("nan"),
            "bootstrap_std":  float(v.std())  if len(v) else float("nan"),
            "ci_lo": lo, "ci_hi": hi,
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "direction": _dir(obs_val),
            "n_valid": int(len(v)),
        }
    return out


# ═══════════════════════════════════════════════════════════════════════
# Interpretation (identical to original)
# ═══════════════════════════════════════════════════════════════════════

def _interpret(per_model, cross, n_models, value_key):
    if not cross or cross.get("n_valid_replicates", 0) == 0:
        return "insufficient_data", "Not enough valid replicates."

    dirs = [r["direction"] for r in per_model.values()]
    all_pos = all(d == "positive" for d in dirs)
    all_neg = all(d == "negative" for d in dirs)
    same_sign = all_pos or all_neg
    dominant_dir = "positive" if all_pos else ("negative" if all_neg else "mixed")

    n_ci = sum(1 for r in per_model.values() if r["ci_excludes_zero"])
    mean_excl = cross["mean_across_models"]["ci_excludes_zero"]

    prop_key = "prop_all_positive" if all_pos else "prop_all_negative"
    prop_agree = cross.get(prop_key, 0.0) if same_sign else \
                 max(cross["prop_all_positive"], cross["prop_all_negative"])

    detail = (f"obs direction: {dominant_dir}; "
              f"{n_ci}/{n_models} CIs exclude 0; "
              f"cross-model mean CI excludes 0: {mean_excl}; "
              f"agreement: {prop_agree:.1%}")

    if same_sign and n_ci >= n_models - 1 and mean_excl and prop_agree >= 0.80:
        return "strong", detail
    if same_sign and mean_excl:
        return "moderate", detail
    return "weak_or_unclear", detail


def _conclusion_D(comparison, strength, per_model, n_models):
    target = comparison["target"]
    reference = comparison["reference"]
    dirs = [r["direction"] for r in per_model.values()]
    all_pos = all(d == "positive" for d in dirs)
    if strength == "strong":
        winner = target if all_pos else reference
        loser  = reference if all_pos else target
        return (f"Strong evidence: {winner} contributes more than {loser}, "
                f"consistently across all {n_models} models.")
    if strength == "moderate":
        winner = target if all_pos else reference
        loser  = reference if all_pos else target
        return (f"Moderate evidence: {winner} tends to contribute more than "
                f"{loser} across models, but not all individual CIs exclude zero.")
    return (f"Weak or unclear evidence: models do not consistently agree "
            f"on whether {target} or {reference} contributes more.")


def _conclusion_C12(strength, per_model, n_models):
    dirs = [r["direction"] for r in per_model.values()]
    all_pos = all(d == "positive" for d in dirs)
    all_neg = all(d == "negative" for d in dirs)
    if strength == "strong":
        if all_pos:
            return (f"Strong evidence: modalities cooperate (C12 > 0) "
                    f"consistently across all {n_models} models.")
        return (f"Strong evidence: modalities are redundant (C12 < 0) "
                f"consistently across all {n_models} models.")
    if strength == "moderate":
        if all_pos:
            return f"Moderate evidence for cooperation (C12 > 0) across models."
        if all_neg:
            return f"Moderate evidence for redundancy (C12 < 0) across models."
        return "Moderate cross-model trend for C12, but direction not unanimous."
    return "Weak or unclear evidence for consistent cooperation or redundancy."


# ═══════════════════════════════════════════════════════════════════════
# Report printer
# ═══════════════════════════════════════════════════════════════════════

def print_report(all_results, file=None):
    out = file or sys.stdout
    def pr(*a, **kw):
        print(*a, **kw, file=out)

    for bench, pairs in all_results.items():
        pr(f"\n{'='*100}")
        pr(f"CLUSTER BOOTSTRAP — CROSS-MODEL CONSISTENCY — {bench}")
        pr(f"{'='*100}")

        for pair_label, pd in pairs.items():
            comp = pd["comparison"]
            n_q  = pd["n_questions"]
            n_d  = pd["n_documents"]
            pr(f"\n  {'─'*90}")
            pr(f"  Modality pair: {pair_label}  (n_questions = {n_q}, n_documents = {n_d})")
            pr(f"  Comparison: D = S_{comp['target']} − S_{comp['reference']}")
            pr(f"  {'─'*90}")

            # ── D per model ──────────────────────────────────────────
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

            # ── D cross-model ────────────────────────────────────────
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
                dist = cd["n_positive_distribution"]
                pr(f"    Distribution of n_positive_models: "
                   + "  ".join(f"{k}:{float(v):.3f}" for k, v in sorted(dist.items())))
            else:
                pr("    (no valid replicates)")

            pr(f"\n  Strength: {pd['D_strength']}")
            pr(f"  Detail:   {pd['D_detail']}")
            pr(f"  >> {pd['D_conclusion']}")

            # ── C12 per model ────────────────────────────────────────
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

            # ── C12 cross-model ──────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════
# CSV writers
# ═══════════════════════════════════════════════════════════════════════

def write_csv_per_model(all_results, path):
    rows = []
    for bench, pairs in all_results.items():
        for pl, pd in pairs.items():
            comp = pd["comparison"]
            for m in MODEL_NAMES:
                rd = pd["per_model_D"].get(m)
                rc = pd["per_model_C12"].get(m)
                if not rd or not rc:
                    continue
                rows.append({
                    "benchmark": bench,
                    "modality_pair": pl,
                    "comparison": f"S_{comp['target']} - S_{comp['reference']}",
                    "model": MODEL_DISPLAY.get(m, m),
                    "n_documents": pd["n_documents"],
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


def write_csv_cross_model(all_results, path):
    rows = []
    for bench, pairs in all_results.items():
        for pl, pd in pairs.items():
            comp = pd["comparison"]
            cd = pd["cross_model_D"]
            cc = pd["cross_model_C12"]
            row = {
                "benchmark": bench,
                "modality_pair": pl,
                "n_questions": pd["n_questions"],
                "n_documents": pd["n_documents"],
                "comparison": f"S_{comp['target']} - S_{comp['reference']}",
            }
            if cd.get("n_valid_replicates", 0) > 0:
                row.update({
                    "D_mean_across": cd["mean_across_models"]["mean"],
                    "D_mean_ci_lo": cd["mean_across_models"]["ci_lo"],
                    "D_mean_ci_hi": cd["mean_across_models"]["ci_hi"],
                    "D_mean_excl_zero": cd["mean_across_models"]["ci_excludes_zero"],
                    "D_prop_all_pos": cd["prop_all_positive"],
                    "D_prop_all_neg": cd["prop_all_negative"],
                })
            if cc.get("n_valid_replicates", 0) > 0:
                row.update({
                    "C12_mean_across": cc["mean_across_models"]["mean"],
                    "C12_mean_ci_lo": cc["mean_across_models"]["ci_lo"],
                    "C12_mean_ci_hi": cc["mean_across_models"]["ci_hi"],
                    "C12_mean_excl_zero": cc["mean_across_models"]["ci_excludes_zero"],
                    "C12_prop_all_pos": cc["prop_all_positive"],
                    "C12_prop_all_neg": cc["prop_all_negative"],
                })
            row.update({
                "D_strength": pd["D_strength"],
                "D_conclusion": pd["D_conclusion"],
                "C12_strength": pd["C12_strength"],
                "C12_conclusion": pd["C12_conclusion"],
            })
            rows.append(row)
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

    benchmarks = {
        "LongDocURL":       (repo / "FinalPassResultsLDU",                ldu_eval_score),
        "MMLongBench-Doc":  (repo / "FinalPassResultsMMLong" / "remapped", mmlong_eval_score),
    }

    all_results = {}

    for bench_name, (rdir, scorer) in benchmarks.items():
        print(f"\n{'='*80}")
        print(f"BENCHMARK: {bench_name}  (cluster bootstrap)")
        print(f"{'='*80}")

        print("  Loading …")
        all_data = load_and_index(rdir, MODEL_NAMES)
        avail = [m for m in MODEL_NAMES if m in all_data]
        print(f"  Models: {avail}")
        n_models = len(avail)

        # Discover modality pairs
        mod_pairs = set()
        for m in avail:
            mod_pairs |= set(all_data[m].keys())
        mod_pairs = sorted(mod_pairs, key=lambda x: str(sorted(x)))

        bench_results = {}

        for mp in mod_pairs:
            mods = sorted(mp)
            pair_label = f"{mods[0]} + {mods[1]}"
            pair_key = tuple(mods)

            comparison = COMPARISONS.get(pair_key)
            if comparison is None:
                print(f"\n  -- {pair_label} -- SKIPPED (no comparison defined)")
                continue

            print(f"\n  -- {pair_label} --")
            print(f"    Comparison: D = S_{comparison['target']} − S_{comparison['reference']}")

            # Step 1: valid questions
            common_keys = find_common_keys(all_data, mp, avail)
            print(f"    Common questions: {len(common_keys)}")
            if len(common_keys) < 2:
                print("    SKIP (< 2 questions)")
                continue

            # Build score arrays
            score_arrays, q_meta, m1_name, m2_name = build_score_arrays(
                all_data, mp, common_keys, avail)
            mods = [m1_name, m2_name]

            # Extract document IDs for clustering
            doc_ids = extract_doc_ids(all_data, mp, common_keys, avail)
            unique_docs = sorted(set(doc_ids))
            n_docs = len(unique_docs)
            print(f"    Documents: {n_docs}  "
                  f"(avg {len(common_keys)/n_docs:.1f} questions/doc)")

            # Steps 2-3: cluster bootstrap
            observed, boot_D, boot_C12, cross_D, cross_C12 = \
                run_cluster_bootstrap(
                    score_arrays, q_meta, scorer, avail,
                    comparison, m1_name, m2_name,
                    doc_ids,
                )

            # Step 5: per-model summaries
            t_i, r_i = _d_indices(m1_name, m2_name, comparison)
            pm_D = _build_per_model(
                observed, boot_D, avail, "D",
                lambda sh: sh[t_i] - sh[r_i],
            )
            pm_C12 = _build_per_model(
                observed, boot_C12, avail, "C12",
                lambda sh: sh[2],
            )

            # Steps 6-7: interpretation
            d_strength, d_detail = _interpret(pm_D, cross_D, n_models, "D")
            d_conclusion = _conclusion_D(comparison, d_strength, pm_D, n_models)

            c12_strength, c12_detail = _interpret(pm_C12, cross_C12, n_models, "C12")
            c12_conclusion = _conclusion_C12(c12_strength, pm_C12, n_models)

            bench_results[pair_label] = {
                "n_questions": len(common_keys),
                "n_documents": n_docs,
                "modalities": mods,
                "comparison": comparison,
                "per_model_D": pm_D,
                "cross_model_D": cross_D,
                "D_strength": d_strength,
                "D_detail": d_detail,
                "D_conclusion": d_conclusion,
                "per_model_C12": pm_C12,
                "cross_model_C12": cross_C12,
                "C12_strength": c12_strength,
                "C12_detail": c12_detail,
                "C12_conclusion": c12_conclusion,
            }

        all_results[bench_name] = bench_results

    # ── Save outputs ─────────────────────────────────────────────────
    json_path = out_dir / "cluster_consistency_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=_ser)
    print(f"\n  JSON: {json_path}")

    txt_path = out_dir / "cluster_consistency_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        print_report(all_results, file=f)
    print(f"  Report: {txt_path}")

    write_csv_per_model(all_results, out_dir / "cluster_consistency_per_model.csv")
    write_csv_cross_model(all_results, out_dir / "cluster_consistency_cross_model.csv")

    # Also print to stdout
    print_report(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
