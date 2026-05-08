#!/usr/bin/env python3
"""
SHAPE Scores by Task Tag × Modality Group (LongDocURL only)
=============================================================

For the LongDocURL benchmark, maps each question to its ``task_tag``
(Understanding, Locating, Reasoning) and computes observed SHAPE
contribution (S_m1, S_m2, D) and cooperation (C12) scores for every
(task_tag × modality-pair) group with at least MIN_GROUP_SIZE questions.

Includes bootstrap confidence intervals (100k replicates).

Output
------
- task_tag_shape_results.json   structured results
- task_tag_shape_results.csv    flat CSV for easy analysis
- task_tag_shape_results.txt    human-readable report
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
    precompute_v0_lookup,
    v0_for_idx,
    compute_shape,
    MODEL_NAMES,
    MODEL_DISPLAY,
    _ser,
)
from cluster_bootstrap.cluster_bootstrap_consistency import (
    COMPARISONS,
    _d_indices,
)
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
# Task-tag mapping
# ═══════════════════════════════════════════════════════════════════════

def load_task_tag_mapping(repo):
    """Load question_id → task_tag from LongDocURL JSONL."""
    mapping = {}
    ldu_path = repo / "longdocurl" / "LongDocURL_public_cleaned_2modalities.jsonl"
    if not ldu_path.exists():
        raise FileNotFoundError(f"JSONL not found: {ldu_path}")
    with open(ldu_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                mapping[item["question_id"]] = item["task_tag"]
    return mapping


# ═══════════════════════════════════════════════════════════════════════
# SHAPE computation for a question subset
# ═══════════════════════════════════════════════════════════════════════

def compute_group_shape(scores, q_meta, indices, models, scorer_fn,
                        comparison, m1_name, m2_name,
                        n_boot=N_BOOTSTRAP, seed=SEED):
    """Compute observed SHAPE scores + bootstrap CIs for a subset."""
    idx = np.array(indices, dtype=np.intp)
    n = len(idx)

    # Subset q_meta for V0 computation
    sub_q_meta = [q_meta[i] for i in idx]

    print(f"      Precomputing V0 ({n} questions) …")
    v0_mat, q_col = precompute_v0_lookup(sub_q_meta, scorer_fn)
    all_sub_idx = np.arange(n)
    v0_obs = v0_for_idx(all_sub_idx, v0_mat, q_col)

    t_i, r_i = _d_indices(m1_name, m2_name, comparison)

    observed = {}
    for m in models:
        V12 = float(scores[m]["both"][idx].mean())
        V1 = float(scores[m]["m1_only"][idx].mean())
        V2 = float(scores[m]["m2_only"][idx].mean())
        sh = compute_shape(V12, V1, V2, v0_obs)
        observed[m] = sh  # (S_m1, S_m2, C12)

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
    pr(f"SHAPE SCORES BY TASK TAG × MODALITY GROUP (LongDocURL)")
    pr(f"(groups with ≥ {MIN_GROUP_SIZE} questions, {N_BOOTSTRAP:,} bootstrap replicates)")
    pr(f"{'='*120}")

    for pair_label, pair_data in sorted(results.items()):
        m1 = pair_data["m1_name"]
        m2 = pair_data["m2_name"]
        comp = pair_data["comparison"]
        n_total = pair_data["n_total"]
        n_groups = len(pair_data["task_tag_groups"])

        pr(f"\n{'─'*120}")
        pr(f"Modality pair: {pair_label}  "
           f"(total: {n_total} questions, "
           f"{n_groups} qualifying task-tag groups)")
        pr(f"D = S_{comp['target']} − S_{comp['reference']}")
        pr(f"{'─'*120}")

        for tag, tdata in sorted(pair_data["task_tag_groups"].items()):
            n = tdata["n_questions"]
            pr(f"\n  Task tag: {tag}  (n={n})")
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
                obs = tdata["observed"].get(m)
                if not obs:
                    continue
                pr(f"  {MODEL_DISPLAY.get(m,m):<20s} "
                   f"{_f(obs['S_m1']):>12s} {_f(obs['S_m2']):>12s} "
                   f"{_f(obs['C12']):>12s} {_ci(obs.get('C12_ci_lo'), obs.get('C12_ci_hi')):>22s} "
                   f"{_f(obs['D']):>12s} {_ci(obs.get('D_ci_lo'), obs.get('D_ci_hi')):>22s}")

    report = "\n".join(lines)
    print(report)

    txt_path = out_dir / "task_tag_shape_results.txt"
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

        for tag, tdata in sorted(pair_data["task_tag_groups"].items()):
            for m in MODEL_NAMES:
                obs = tdata["observed"].get(m)
                if not obs:
                    continue
                rows.append({
                    "modality_pair": pair_label,
                    "m1_name": m1,
                    "m2_name": m2,
                    "comparison": f"S_{comp['target']} - S_{comp['reference']}",
                    "task_tag": tag,
                    "n_questions": tdata["n_questions"],
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

    # ── Step 1: Load task_tag mappings ─────────────────────────────────
    print("Loading task_tag mappings from LongDocURL …")
    tag_map = load_task_tag_mapping(repo)
    print(f"  {len(tag_map)} questions with task_tag\n")

    # ── Step 2: Load LongDocURL results ────────────────────────────────
    results_dir = repo / "results_longdocurl"
    print("Loading LongDocURL results …")
    all_data = load_and_index(results_dir, MODEL_NAMES)
    avail_models = [m for m in MODEL_NAMES if m in all_data]
    print(f"  Models available: {avail_models}\n")

    # Discover all modality pairs
    all_mod_pairs = set()
    for m in avail_models:
        all_mod_pairs |= set(all_data[m].keys())
    all_mod_pairs = sorted(all_mod_pairs, key=lambda x: str(sorted(x)))

    # ── Step 3: For each modality pair, split by task_tag ──────────────
    print(f"{'='*80}")
    print(f"Computing SHAPE for task_tag × modality groups (min n={MIN_GROUP_SIZE})")
    print(f"{'='*80}")

    results = {}

    for mp in all_mod_pairs:
        mods = sorted(mp)
        pair_label = f"{mods[0]} + {mods[1]}"
        pair_key = tuple(mods)

        comparison = COMPARISONS.get(pair_key)
        if comparison is None:
            continue

        common_keys = find_common_keys(all_data, mp, avail_models)
        if len(common_keys) < 2:
            print(f"\n  {pair_label}: {len(common_keys)} questions — skipping")
            continue

        sa, qm, m1_name, m2_name = build_score_arrays(all_data, mp, common_keys, avail_models)
        t_i, r_i = _d_indices(m1_name, m2_name, comparison)

        # Base question IDs — first element of each common_key tuple
        q_ids = [qk[0] for qk in common_keys]
        n = len(common_keys)

        print(f"\n  -- {pair_label} ({n} questions) --")

        # Map each question to its task_tag
        tag_groups = defaultdict(list)
        unmapped = 0
        for i, qid in enumerate(q_ids):
            tag = tag_map.get(qid)
            if tag is None:
                unmapped += 1
                continue
            tag_groups[tag].append(i)

        if unmapped:
            print(f"    {unmapped}/{n} questions without task_tag mapping")

        # Filter to qualifying groups
        qualifying = {tag: indices
                      for tag, indices in tag_groups.items()
                      if len(indices) >= MIN_GROUP_SIZE}

        if not qualifying:
            print(f"    No qualifying task_tag groups")
            continue

        print(f"    {len(qualifying)} qualifying groups "
              f"(out of {len(tag_groups)} total)")

        pair_results = {}
        for tag, indices in sorted(qualifying.items()):
            print(f"    {tag} (n={len(indices)}): computing SHAPE …")
            observed_raw, v0_obs, boot_D, boot_C12 = compute_group_shape(
                sa, qm, indices, avail_models, ldu_eval_score,
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
                        "C12_ci_lo": c12_lo, "C12_ci_hi": c12_hi,
                        "D": d_val,
                        "D_ci_lo": d_lo, "D_ci_hi": d_hi,
                        "S_target": [s_m1, s_m2][t_i],
                        "S_reference": [s_m1, s_m2][r_i],
                    }
                else:
                    observed[m] = {
                        "S_m1": None, "S_m2": None, "C12": None,
                        "C12_ci_lo": None, "C12_ci_hi": None,
                        "D": None,
                        "D_ci_lo": None, "D_ci_hi": None,
                        "S_target": None, "S_reference": None,
                    }

            # Aggregated bootstrap CI (mean across models per replicate)
            all_boot_D = np.stack([boot_D[m] for m in avail_models])
            all_boot_C12 = np.stack([boot_C12[m] for m in avail_models])
            agg_D = np.nanmean(all_boot_D, axis=0)
            agg_C12 = np.nanmean(all_boot_C12, axis=0)
            agg_d_lo, agg_d_hi = _pci(agg_D)
            agg_c12_lo, agg_c12_hi = _pci(agg_C12)

            pair_results[tag] = {
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
            "task_tag_groups": pair_results,
        }

    # ── Step 4: Outputs ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Saving results …")

    json_path = out_dir / "task_tag_shape_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_ser)
    print(f"  JSON: {json_path}")

    write_csv(results, out_dir / "task_tag_shape_results.csv")
    print_report(results, out_dir)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("SUMMARY: Qualifying task_tag groups per modality pair")
    print(f"{'='*80}")
    print(f"{'Modality Pair':<25s} {'# Tag Groups':>15s} {'# Questions':>15s}")
    print(f"{'-'*55}")
    total_groups = 0
    total_questions = 0
    for pair_label, pair_data in sorted(results.items()):
        ng = len(pair_data["task_tag_groups"])
        nq = sum(tg["n_questions"] for tg in pair_data["task_tag_groups"].values())
        total_groups += ng
        total_questions += nq
        print(f"{pair_label:<25s} {ng:>15d} {nq:>15d}")
    print(f"{'-'*55}")
    print(f"{'TOTAL':<25s} {total_groups:>15d} {total_questions:>15d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
