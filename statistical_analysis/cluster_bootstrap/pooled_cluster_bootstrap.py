#!/usr/bin/env python3
"""
Pooled Cluster Bootstrap — Cross-Model Consistency Analysis
=============================================================

Pools questions and documents from ALL benchmarks into a single set
for each modality pair, then runs the cluster bootstrap on the combined
data.  Each question is scored with its own benchmark's eval_score
function (they are nearly identical but differ in type-label strings).

Output structure is identical to the per-benchmark version so the same
plotting script can be used.
"""

import csv
import io
import json
import sys
import time
import numpy as np
from pathlib import Path
from contextlib import redirect_stdout

# ── shared infrastructure ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_pairwise_bootstrap import (
    load_and_index,
    find_common_keys,
    build_score_arrays,
    _norm_ans,
    v0_for_idx,
    compute_shape,
    MODEL_NAMES,
    MODEL_DISPLAY,
    _ser,
)
from cluster_bootstrap_consistency import (
    _d_indices,
    _cross_model_summary, _build_per_model,
    _interpret, _conclusion_D, _conclusion_C12,
    extract_doc_ids, build_cluster_index,
    COMPARISONS,
)
from mmlongbench.eval.eval_score import eval_score as mmlong_eval_score
from longdocurl.utils.utils_score_v3 import eval_score as ldu_eval_score


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

N_BOOTSTRAP = 100_000
SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# Pooled V0 precomputation
# ═══════════════════════════════════════════════════════════════════════

def precompute_v0_lookup_pooled(q_meta, scorer_fns):
    """
    Like precompute_v0_lookup but each question can use a different
    scorer function.

    Parameters
    ----------
    q_meta      : list of (gold_answer, answer_format)
    scorer_fns  : list of callable, same length as q_meta
    """
    n = len(q_meta)
    normed = [_norm_ans(g) for g, _ in q_meta]
    uniq = sorted(set(normed))
    a2c = {a: j for j, a in enumerate(uniq)}
    q_col = np.array([a2c[a] for a in normed], dtype=np.intp)
    v0 = np.zeros((n, len(uniq)))
    for j, maj in enumerate(uniq):
        for i, (gold, fmt) in enumerate(q_meta):
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    sc = scorer_fns[i](gold, maj, fmt)
                v0[i, j] = float(sc)
            except Exception:
                v0[i, j] = 0.0
    return v0, q_col


# ═══════════════════════════════════════════════════════════════════════
# Pool data from multiple benchmarks
# ═══════════════════════════════════════════════════════════════════════

def pool_benchmarks(benchmark_configs, models):
    """
    For each modality pair present in ANY benchmark, pool all questions
    and documents together.

    Parameters
    ----------
    benchmark_configs : dict
        bench_name → (results_dir, scorer_fn)
    models : list[str]

    Returns
    -------
    pooled : dict
        pair_label → {
            "scores": {model: {"both": arr, "m1_only": arr, "m2_only": arr}},
            "q_meta": list of (answer, format),
            "scorer_fns": list of callable (one per question),
            "doc_ids": list of str (prefixed with benchmark name),
            "m1_name": str, "m2_name": str,
            "comparison": dict,
            "n_questions": int,
            "n_documents": int,
            "benchmark_counts": dict  bench_name → n_questions
        }
    """
    # Load all benchmarks
    all_bench_data = {}
    for bench_name, (rdir, scorer) in benchmark_configs.items():
        print(f"  Loading {bench_name} …")
        all_data = load_and_index(rdir, models)
        avail = [m for m in models if m in all_data]
        all_bench_data[bench_name] = (all_data, avail, scorer)

    # Use models available in ALL benchmarks
    avail_all = None
    for bench_name, (all_data, avail, _) in all_bench_data.items():
        s = set(avail)
        avail_all = s if avail_all is None else (avail_all & s)
    avail_models = [m for m in models if m in avail_all]
    print(f"  Models available in all benchmarks: {avail_models}")

    # Discover all modality pairs
    all_mod_pairs = set()
    for bench_name, (all_data, avail, _) in all_bench_data.items():
        for m in avail:
            all_mod_pairs |= set(all_data[m].keys())
    all_mod_pairs = sorted(all_mod_pairs, key=lambda x: str(sorted(x)))

    pooled = {}

    for mp in all_mod_pairs:
        mods = sorted(mp)
        pair_label = f"{mods[0]} + {mods[1]}"
        pair_key = tuple(mods)

        comparison = COMPARISONS.get(pair_key)
        if comparison is None:
            print(f"\n  -- {pair_label} -- SKIPPED (no comparison defined)")
            continue

        print(f"\n  -- {pair_label} --")

        # Collect per-benchmark data
        bench_scores = []   # list of (scores_dict, q_meta, doc_ids, scorer_fn, bench_name, n)
        m1_name = m2_name = None

        for bench_name, (all_data, avail, scorer) in all_bench_data.items():
            # Check this pair exists for all models in this benchmark
            common_keys = find_common_keys(all_data, mp, avail_models)
            if len(common_keys) < 2:
                print(f"    {bench_name}: {len(common_keys)} questions — skipping")
                continue

            sa, qm, m1, m2 = build_score_arrays(all_data, mp, common_keys, avail_models)
            doc_ids = extract_doc_ids(all_data, mp, common_keys, avail_models)
            # Prefix doc_ids with benchmark name to avoid collisions
            doc_ids = [f"{bench_name}::{d}" for d in doc_ids]

            if m1_name is None:
                m1_name, m2_name = m1, m2
            else:
                assert m1_name == m1 and m2_name == m2, \
                    f"Modality name mismatch: {m1_name},{m2_name} vs {m1},{m2}"

            bench_scores.append((sa, qm, doc_ids, scorer, bench_name, len(common_keys)))
            print(f"    {bench_name}: {len(common_keys)} questions, "
                  f"{len(set(doc_ids))} documents")

        if not bench_scores or m1_name is None:
            print(f"    No data to pool — skipping")
            continue

        # Concatenate
        total_n = sum(bs[5] for bs in bench_scores)
        pooled_scores = {m: {"both": np.empty(total_n),
                             "m1_only": np.empty(total_n),
                             "m2_only": np.empty(total_n)}
                         for m in avail_models}
        pooled_q_meta = []
        pooled_scorer_fns = []
        pooled_doc_ids = []
        benchmark_counts = {}

        offset = 0
        for sa, qm, doc_ids, scorer, bench_name, n in bench_scores:
            for m in avail_models:
                pooled_scores[m]["both"][offset:offset+n]    = sa[m]["both"]
                pooled_scores[m]["m1_only"][offset:offset+n] = sa[m]["m1_only"]
                pooled_scores[m]["m2_only"][offset:offset+n] = sa[m]["m2_only"]
            pooled_q_meta.extend(qm)
            pooled_scorer_fns.extend([scorer] * n)
            pooled_doc_ids.extend(doc_ids)
            benchmark_counts[bench_name] = n
            offset += n

        unique_docs = sorted(set(pooled_doc_ids))
        n_docs = len(unique_docs)
        print(f"    Pooled: {total_n} questions, {n_docs} documents")

        pooled[pair_label] = {
            "scores": pooled_scores,
            "q_meta": pooled_q_meta,
            "scorer_fns": pooled_scorer_fns,
            "doc_ids": pooled_doc_ids,
            "m1_name": m1_name,
            "m2_name": m2_name,
            "comparison": comparison,
            "n_questions": total_n,
            "n_documents": n_docs,
            "benchmark_counts": benchmark_counts,
        }

    return pooled, avail_models


# ═══════════════════════════════════════════════════════════════════════
# Pooled cluster bootstrap
# ═══════════════════════════════════════════════════════════════════════

def run_pooled_cluster_bootstrap(scores, q_meta, scorer_fns, models,
                                 comparison, m1_name, m2_name,
                                 doc_ids,
                                 n_boot=N_BOOTSTRAP, seed=SEED):
    """
    Cluster bootstrap on pooled data.  Uses per-question scorer functions
    for V0 precomputation.
    """
    n = len(q_meta)
    if n == 0:
        return {}, {}, {}, {}, {}

    unique_docs, doc_to_qidx = build_cluster_index(doc_ids)
    n_docs = len(unique_docs)
    doc_idx_arrays = [np.array(doc_to_qidx[d], dtype=np.intp)
                      for d in unique_docs]

    rng = np.random.default_rng(seed)
    n_m = len(models)
    t_i, r_i = _d_indices(m1_name, m2_name, comparison)

    # V0 precomputation (pooled — uses per-question scorer)
    print("    Precomputing V0 lookup (pooled) …")
    v0_mat, q_col = precompute_v0_lookup_pooled(q_meta, scorer_fns)

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

    print(f"    Running {n_boot:,} pooled cluster bootstrap replicates "
          f"({n_docs} documents, {n} questions) …")
    t0 = time.time()
    for b in range(n_boot):
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


# ═══════════════════════════════════════════════════════════════════════
# Report printer
# ═══════════════════════════════════════════════════════════════════════

def print_report(all_results, file=None):
    out = file or sys.stdout
    def pr(*a, **kw):
        print(*a, **kw, file=out)

    pr(f"\n{'='*100}")
    pr(f"POOLED CLUSTER BOOTSTRAP — CROSS-MODEL CONSISTENCY")
    pr(f"{'='*100}")

    for pair_label, pd in all_results.items():
        comp = pd["comparison"]
        n_q  = pd["n_questions"]
        n_d  = pd["n_documents"]
        bc   = pd.get("benchmark_counts", {})
        bc_str = ", ".join(f"{k}: {v}" for k, v in bc.items())
        pr(f"\n  {'─'*90}")
        pr(f"  Modality pair: {pair_label}")
        pr(f"  Total: n_questions = {n_q}, n_documents = {n_d}")
        pr(f"  Per benchmark: {bc_str}")
        pr(f"  Comparison: D = S_{comp['target']} − S_{comp['reference']}")
        pr(f"  {'─'*90}")

        # ── D per model
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

        # ── D cross-model
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

        # ── C12 per model
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

        # ── C12 cross-model
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
    for pl, pd in all_results.items():
        comp = pd["comparison"]
        for m in MODEL_NAMES:
            rd = pd["per_model_D"].get(m)
            rc = pd["per_model_C12"].get(m)
            if not rd or not rc:
                continue
            rows.append({
                "benchmark": "Pooled",
                "modality_pair": pl,
                "comparison": f"S_{comp['target']} - S_{comp['reference']}",
                "model": MODEL_DISPLAY.get(m, m),
                "n_questions": pd["n_questions"],
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
    for pl, pd in all_results.items():
        comp = pd["comparison"]
        cd = pd["cross_model_D"]
        cc = pd["cross_model_C12"]
        row = {
            "benchmark": "Pooled",
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

    benchmark_configs = {
        "LongDocURL":       (repo / "FinalPassResultsLDU",                ldu_eval_score),
        "MMLongBench-Doc":  (repo / "FinalPassResultsMMLong" / "remapped", mmlong_eval_score),
    }

    print("Pooling benchmarks …")
    pooled_data, avail_models = pool_benchmarks(benchmark_configs, MODEL_NAMES)
    n_models = len(avail_models)

    all_results = {}

    for pair_label, pdata in pooled_data.items():
        comparison = pdata["comparison"]
        m1_name = pdata["m1_name"]
        m2_name = pdata["m2_name"]

        print(f"\n  Running bootstrap for {pair_label} …")

        observed, boot_D, boot_C12, cross_D, cross_C12 = \
            run_pooled_cluster_bootstrap(
                pdata["scores"], pdata["q_meta"], pdata["scorer_fns"],
                avail_models, comparison, m1_name, m2_name,
                pdata["doc_ids"],
            )

        t_i, r_i = _d_indices(m1_name, m2_name, comparison)
        pm_D = _build_per_model(
            observed, boot_D, avail_models, "D",
            lambda sh: sh[t_i] - sh[r_i],
        )
        pm_C12 = _build_per_model(
            observed, boot_C12, avail_models, "C12",
            lambda sh: sh[2],
        )

        d_strength, d_detail = _interpret(pm_D, cross_D, n_models, "D")
        d_conclusion = _conclusion_D(comparison, d_strength, pm_D, n_models)

        c12_strength, c12_detail = _interpret(pm_C12, cross_C12, n_models, "C12")
        c12_conclusion = _conclusion_C12(c12_strength, pm_C12, n_models)

        all_results[pair_label] = {
            "n_questions": pdata["n_questions"],
            "n_documents": pdata["n_documents"],
            "benchmark_counts": pdata["benchmark_counts"],
            "modalities": [m1_name, m2_name],
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

    # ── Save outputs ─────────────────────────────────────────────────
    json_path = out_dir / "pooled_consistency_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=_ser)
    print(f"\n  JSON: {json_path}")

    txt_path = out_dir / "pooled_consistency_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        print_report(all_results, file=f)
    print(f"  Report: {txt_path}")

    write_csv_per_model(all_results, out_dir / "pooled_consistency_per_model.csv")
    write_csv_cross_model(all_results, out_dir / "pooled_consistency_cross_model.csv")

    # Also print to stdout
    print_report(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
