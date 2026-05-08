#!/usr/bin/env python3
"""
SHAPE Scores by Question Intent × Modality Group
==================================================

Pools ALL questions from both benchmarks (LongDocURL + MMLongBench-Doc),
maps each question to its ``question_intent`` label, and computes observed
SHAPE contribution (S_m1, S_m2, D) and cooperation (C12) scores for every
(intent × modality-pair) group with at least MIN_GROUP_SIZE questions.

Output
------
- intent_shape_results.json   structured results
- intent_shape_results.csv    flat CSV for easy analysis
- intent_shape_results.txt    human-readable report
"""

import csv
import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── shared infrastructure ────────────────────────────────────────────
_repo = Path(__file__).parent.parent
_stat_dir = _repo / "statistical_analysis"
sys.path.insert(0, str(_stat_dir))
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_stat_dir / "cluster_bootstrap"))

from model_pairwise_bootstrap import (
    load_and_index,
    find_common_keys,
    build_score_arrays,
    v0_for_idx,
    compute_shape,
    MODEL_NAMES,
    MODEL_DISPLAY,
    _ser,
)
from cluster_bootstrap.cluster_bootstrap_consistency import (
    COMPARISONS,
    _d_indices,
    extract_doc_ids,
)
from cluster_bootstrap.pooled_cluster_bootstrap import (
    precompute_v0_lookup_pooled,
)
from mmlongbench.eval.eval_score import eval_score as mmlong_eval_score
from longdocurl.utils.utils_score_v3 import eval_score as ldu_eval_score


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

MIN_GROUP_SIZE = 1
N_BOOTSTRAP = 100_000
SEED = 42


def _pci(arr, alpha=0.05):
    """Percentile CI (lo, hi) ignoring NaNs."""
    v = arr[~np.isnan(arr)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 100 * alpha / 2)), \
           float(np.percentile(v, 100 * (1 - alpha / 2)))


# ═══════════════════════════════════════════════════════════════════════
# Intent mapping
# ═══════════════════════════════════════════════════════════════════════

def load_intent_mappings(intent_dir):
    """Load question_id → question_intent from both intent files."""
    mapping = {}

    # MMLongBench
    mm_path = intent_dir / "samples_2modalities_zero_shot_reconciled.json"
    if mm_path.exists():
        with open(mm_path, encoding="utf-8") as f:
            for item in json.load(f):
                mapping[item["question_id"]] = item["question_intent"]

    # LongDocURL
    ldu_path = intent_dir / "LongDocURL_public_cleaned_2modalities_zero_shot_reconciled.jsonl"
    if ldu_path.exists():
        with open(ldu_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    mapping[item["question_id"]] = item["question_intent"]

    return mapping


# ═══════════════════════════════════════════════════════════════════════
# Pooling with question-key tracking
# ═══════════════════════════════════════════════════════════════════════

def pool_benchmarks_with_keys(benchmark_configs, models):
    """
    Like pool_benchmarks but also preserves the base question IDs
    (first element of each common_key tuple) so we can map to intents.
    """
    # Load all benchmarks
    all_bench_data = {}
    for bench_name, (rdir, scorer) in benchmark_configs.items():
        print(f"  Loading {bench_name} …")
        all_data = load_and_index(rdir, models)
        avail = [m for m in models if m in all_data]
        all_bench_data[bench_name] = (all_data, avail, scorer)

    # Models available in ALL benchmarks
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
            continue

        print(f"\n  -- {pair_label} --")

        bench_scores = []
        m1_name = m2_name = None

        for bench_name, (all_data, avail, scorer) in all_bench_data.items():
            common_keys = find_common_keys(all_data, mp, avail_models)
            if len(common_keys) < 2:
                print(f"    {bench_name}: {len(common_keys)} questions — skipping")
                continue

            sa, qm, m1, m2 = build_score_arrays(all_data, mp, common_keys, avail_models)
            doc_ids = extract_doc_ids(all_data, mp, common_keys, avail_models)
            doc_ids = [f"{bench_name}::{d}" for d in doc_ids]

            # Base question IDs — first element of each common_key tuple
            q_ids = [qk[0] for qk in common_keys]

            if m1_name is None:
                m1_name, m2_name = m1, m2

            bench_scores.append((sa, qm, doc_ids, scorer, bench_name,
                                 len(common_keys), q_ids))
            print(f"    {bench_name}: {len(common_keys)} questions")

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
        pooled_q_ids = []
        benchmark_counts = {}

        offset = 0
        for sa, qm, doc_ids, scorer, bench_name, n, q_ids in bench_scores:
            for m in avail_models:
                pooled_scores[m]["both"][offset:offset+n] = sa[m]["both"]
                pooled_scores[m]["m1_only"][offset:offset+n] = sa[m]["m1_only"]
                pooled_scores[m]["m2_only"][offset:offset+n] = sa[m]["m2_only"]
            pooled_q_meta.extend(qm)
            pooled_scorer_fns.extend([scorer] * n)
            pooled_doc_ids.extend(doc_ids)
            pooled_q_ids.extend(q_ids)
            benchmark_counts[bench_name] = n
            offset += n

        print(f"    Pooled: {total_n} questions")

        pooled[pair_label] = {
            "scores": pooled_scores,
            "q_meta": pooled_q_meta,
            "scorer_fns": pooled_scorer_fns,
            "doc_ids": pooled_doc_ids,
            "q_ids": pooled_q_ids,
            "m1_name": m1_name,
            "m2_name": m2_name,
            "comparison": comparison,
            "n_questions": total_n,
            "benchmark_counts": benchmark_counts,
        }

    return pooled, avail_models


# ═══════════════════════════════════════════════════════════════════════
# SHAPE computation for a question subset
# ═══════════════════════════════════════════════════════════════════════

def compute_group_shape(scores, q_meta, scorer_fns, indices, models,
                         comparison, m1_name, m2_name,
                         n_boot=N_BOOTSTRAP, seed=SEED):
    """Compute observed SHAPE scores + bootstrap CIs for a subset."""
    idx = np.array(indices, dtype=np.intp)
    n = len(idx)

    # Subset q_meta and scorer_fns for V0 computation
    sub_q_meta = [q_meta[i] for i in idx]
    sub_scorer_fns = [scorer_fns[i] for i in idx]

    print(f"      Precomputing V0 ({n} questions) …")
    v0_mat, q_col = precompute_v0_lookup_pooled(sub_q_meta, sub_scorer_fns)
    all_sub_idx = np.arange(n)
    v0_obs = v0_for_idx(all_sub_idx, v0_mat, q_col)

    t_i, r_i = _d_indices(m1_name, m2_name, comparison)

    observed = {}
    for m in models:
        V12 = float(scores[m]["both"][idx].mean())
        V1 = float(scores[m]["m1_only"][idx].mean())
        V2 = float(scores[m]["m2_only"][idx].mean())
        sh = compute_shape(V12, V1, V2, v0_obs)
        observed[m] = sh   # (S_m1, S_m2, C12)

    # ── Fast bootstrap ─────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    boot_D   = {m: np.empty(n_boot) for m in models}
    boot_C12 = {m: np.empty(n_boot) for m in models}

    t0 = time.time()
    for b in range(n_boot):
        bidx = rng.choice(n, size=n, replace=True)
        orig_bidx = idx[bidx]
        v0_b = v0_for_idx(bidx, v0_mat, q_col)
        for m in models:
            V12 = float(scores[m]["both"][orig_bidx].mean())
            V1  = float(scores[m]["m1_only"][orig_bidx].mean())
            V2  = float(scores[m]["m2_only"][orig_bidx].mean())
            sh  = compute_shape(V12, V1, V2, v0_b)
            if np.isnan(sh[0]):
                boot_D[m][b]   = np.nan
                boot_C12[m][b] = np.nan
            else:
                boot_D[m][b]   = sh[t_i] - sh[r_i]
                boot_C12[m][b] = sh[2]
    print(f"      Bootstrap ({n_boot} reps) {time.time()-t0:.1f}s")

    return observed, v0_obs, boot_D, boot_C12


# ═══════════════════════════════════════════════════════════════════════
# Report printer
# ═══════════════════════════════════════════════════════════════════════

def print_report(results, out_dir):
    lines = []
    def pr(s=""):
        lines.append(s)

    pr(f"{'='*120}")
    pr(f"SHAPE SCORES BY QUESTION INTENT × MODALITY GROUP")
    pr(f"(groups with ≥ {MIN_GROUP_SIZE} questions, pooled across both benchmarks)")
    pr(f"{'='*120}")

    for pair_label, pair_data in sorted(results.items()):
        m1 = pair_data["m1_name"]
        m2 = pair_data["m2_name"]
        comp = pair_data["comparison"]
        n_total = pair_data["n_total"]
        n_groups = len(pair_data["intent_groups"])

        pr(f"\n{'─'*120}")
        pr(f"Modality pair: {pair_label}  "
           f"(total pooled: {n_total} questions, "
           f"{n_groups} qualifying intent groups)")
        pr(f"D = S_{comp['target']} − S_{comp['reference']}")
        pr(f"{'─'*120}")

        for intent, idata in sorted(pair_data["intent_groups"].items()):
            n = idata["n_questions"]
            pr(f"\n  Intent: {intent}  (n={n})")
            pr(f"  {'Model':<20s} {'S_'+m1:>12s} {'S_'+m2:>12s} "
               f"{'C12':>12s} {'C12 95% CI':>22s} "
               f"{'D':>12s} {'D 95% CI':>22s}")
            pr(f"  {'-'*112}")

            def _f(v):
                return f"{v:.4f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "N/A"
            def _ci(lo, hi):
                if lo is None or hi is None:
                    return "N/A"
                if isinstance(lo, float) and np.isnan(lo):
                    return "N/A"
                return f"[{lo:.4f}, {hi:.4f}]"

            for m in MODEL_NAMES:
                obs = idata["observed"].get(m)
                if not obs:
                    continue
                pr(f"  {MODEL_DISPLAY.get(m,m):<20s} "
                   f"{_f(obs['S_m1']):>12s} {_f(obs['S_m2']):>12s} "
                   f"{_f(obs['C12']):>12s} {_ci(obs.get('C12_ci_lo'), obs.get('C12_ci_hi')):>22s} "
                   f"{_f(obs['D']):>12s} {_ci(obs.get('D_ci_lo'), obs.get('D_ci_hi')):>22s}")

    report = "\n".join(lines)
    print(report)

    txt_path = out_dir / "intent_shape_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report: {txt_path}")


# ═══════════════════════════════════════════════════════════════════════
# CSV writer
# ═══════════════════════════════════════════════════════════════════════

def write_csv(results, path):
    rows = []
    for pair_label, pair_data in sorted(results.items()):
        comp = pair_data["comparison"]
        m1 = pair_data["m1_name"]
        m2 = pair_data["m2_name"]

        for intent, idata in sorted(pair_data["intent_groups"].items()):
            for m in MODEL_NAMES:
                obs = idata["observed"].get(m)
                if not obs:
                    continue
                rows.append({
                    "modality_pair": pair_label,
                    "m1_name": m1,
                    "m2_name": m2,
                    "comparison": f"S_{comp['target']} - S_{comp['reference']}",
                    "question_intent": intent,
                    "n_questions": idata["n_questions"],
                    "model": MODEL_DISPLAY.get(m, m),
                    "S_m1": obs["S_m1"],
                    "S_m2": obs["S_m2"],
                    "C12": obs["C12"],
                    "C12_ci_lo": obs.get("C12_ci_lo"),
                    "C12_ci_hi": obs.get("C12_ci_hi"),
                    "D": obs["D"],
                    "D_ci_lo": obs.get("D_ci_lo"),
                    "D_ci_hi": obs.get("D_ci_hi"),
                    "S_target": obs["S_target"],
                    "S_reference": obs["S_reference"],
                })
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"  CSV: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    repo = Path(__file__).parent.parent
    out_dir = Path(__file__).parent
    intent_dir = out_dir

    # ── Step 1: Load intent mappings ───────────────────────────────────
    print("Loading question intent mappings …")
    intent_map = load_intent_mappings(intent_dir)
    print(f"  {len(intent_map)} questions with intents\n")

    # ── Step 2: Pool benchmarks (with question-key tracking) ──────────
    benchmark_configs = {
        "LongDocURL":      (repo / "results_longdocurl",   ldu_eval_score),
        "MMLongBench-Doc": (repo / "results_mmlongbench",  mmlong_eval_score),
    }

    print("Pooling benchmarks …")
    pooled_data, avail_models = pool_benchmarks_with_keys(
        benchmark_configs, MODEL_NAMES,
    )

    # ── Step 3: Split by intent, filter, compute SHAPE ────────────────
    print(f"\n{'='*80}")
    print(f"Computing SHAPE for intent × modality groups (min n={MIN_GROUP_SIZE})")
    print(f"{'='*80}")

    results = {}

    for pair_label, pdata in sorted(pooled_data.items()):
        q_ids = pdata["q_ids"]
        n = pdata["n_questions"]
        comparison = pdata["comparison"]
        m1_name = pdata["m1_name"]
        m2_name = pdata["m2_name"]
        t_i, r_i = _d_indices(m1_name, m2_name, comparison)

        # Map each pooled question to its intent
        intent_groups = defaultdict(list)
        unmapped = 0
        for i, qid in enumerate(q_ids):
            intent = intent_map.get(qid)
            if intent is None:
                unmapped += 1
                continue
            intent_groups[intent].append(i)

        if unmapped:
            print(f"\n  {pair_label}: {unmapped}/{n} questions without intent mapping")

        # Filter to qualifying groups
        qualifying = {intent: indices
                      for intent, indices in intent_groups.items()
                      if len(indices) >= MIN_GROUP_SIZE}

        if not qualifying:
            print(f"\n  {pair_label}: no qualifying intent groups")
            continue

        print(f"\n  {pair_label}: {len(qualifying)} qualifying intent groups "
              f"(out of {len(intent_groups)} total)")

        pair_results = {}
        for intent, indices in sorted(qualifying.items()):
            print(f"    {intent} (n={len(indices)}): computing SHAPE …")
            observed_raw, v0_obs, boot_D, boot_C12 = compute_group_shape(
                pdata["scores"], pdata["q_meta"], pdata["scorer_fns"],
                indices, avail_models,
                comparison, m1_name, m2_name,
            )

            # Build structured observed dict with D and bootstrap CIs
            observed = {}
            for m in avail_models:
                sh = observed_raw[m]
                if isinstance(sh, tuple) and not np.isnan(sh[0]):
                    s_m1, s_m2, c12 = float(sh[0]), float(sh[1]), float(sh[2])
                    d_val = [s_m1, s_m2][t_i] - [s_m1, s_m2][r_i]
                    d_lo, d_hi = _pci(boot_D[m])
                    c12_lo, c12_hi = _pci(boot_C12[m])
                    observed[m] = {
                        "S_m1": s_m1, "S_m2": s_m2, "C12": c12,
                        "D": d_val,
                        "S_target": [s_m1, s_m2][t_i],
                        "S_reference": [s_m1, s_m2][r_i],
                        "D_ci_lo": d_lo, "D_ci_hi": d_hi,
                        "C12_ci_lo": c12_lo, "C12_ci_hi": c12_hi,
                    }
                else:
                    observed[m] = {
                        "S_m1": None, "S_m2": None, "C12": None,
                        "D": None, "S_target": None, "S_reference": None,
                        "D_ci_lo": None, "D_ci_hi": None,
                        "C12_ci_lo": None, "C12_ci_hi": None,
                    }

            # Aggregated bootstrap CI (mean across models per replicate)
            all_boot_D = np.stack([boot_D[m] for m in avail_models])
            all_boot_C12 = np.stack([boot_C12[m] for m in avail_models])
            agg_D = np.nanmean(all_boot_D, axis=0)
            agg_C12 = np.nanmean(all_boot_C12, axis=0)
            agg_d_lo, agg_d_hi = _pci(agg_D)
            agg_c12_lo, agg_c12_hi = _pci(agg_C12)

            pair_results[intent] = {
                "n_questions": len(indices),
                "observed": observed,
                "agg_D_ci_lo": agg_d_lo, "agg_D_ci_hi": agg_d_hi,
                "agg_C12_ci_lo": agg_c12_lo, "agg_C12_ci_hi": agg_c12_hi,
            }

        results[pair_label] = {
            "m1_name": m1_name,
            "m2_name": m2_name,
            "comparison": comparison,
            "n_total": n,
            "intent_groups": pair_results,
        }

    # ── Step 4: Outputs ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Saving results …")

    json_path = out_dir / "intent_shape_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_ser)
    print(f"  JSON: {json_path}")

    write_csv(results, out_dir / "intent_shape_results.csv")
    print_report(results, out_dir)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("SUMMARY: Qualifying intent groups per modality pair")
    print(f"{'='*80}")
    print(f"{'Modality Pair':<25s} {'# Intent Groups':>15s} {'# Questions':>15s}")
    print(f"{'-'*55}")
    total_groups = 0
    total_questions = 0
    for pair_label, pair_data in sorted(results.items()):
        ng = len(pair_data["intent_groups"])
        nq = sum(ig["n_questions"] for ig in pair_data["intent_groups"].values())
        total_groups += ng
        total_questions += nq
        print(f"{pair_label:<25s} {ng:>15d} {nq:>15d}")
    print(f"{'-'*55}")
    print(f"{'TOTAL':<25s} {total_groups:>15d} {total_questions:>15d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
