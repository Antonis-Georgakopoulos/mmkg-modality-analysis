#!/usr/bin/env python3
"""
Paired Bootstrap Analysis for SHAPE Metrics
It's for understanding how much model A differs from model B.
============================================

For each (benchmark, modality_pair):
  1. Identify base question IDs common across all models x all 3 conditions.
  2. Bootstrap-resample (with replacement) those base IDs B=10000 times.
     Same resampled indices used for every model (paired design).
  3. For each replicate compute S_m1, S_m2, C12 for every model.
     V0 (majority baseline) is recomputed on every replicate via a
     pre-computed score matrix so that the majority answer can change.
  4. For every unordered model pair compute bootstrap distribution of
     difference and report observed difference + 95% percentile CI.
  5. Paired permutation test (10000 permutations) for formal p-values.
     Holm-Bonferroni correction applied across all tests in a benchmark.
"""

import json, sys, re, io, time
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).parent.parent))
from mmlongbench.eval.eval_score import eval_score as mmlong_eval_score
from longdocurl.utils.utils_score_v3 import eval_score as ldu_eval_score

N_BOOTSTRAP = 10_000
N_PERMUTATIONS = 10_000
ALPHA = 0.05
SEED = 42

MODEL_NAMES = ["gemma3_4b", "gemma3_27b", "gpt-4o-mini", "qwen3-vl_8b", "qwen3-vl_30b"]
MODEL_DISPLAY = {
    "gemma3_4b": "Gemma3-4B", "gemma3_27b": "Gemma3-27B",
    "gpt-4o-mini": "GPT-4o-mini", "qwen3-vl_8b": "Qwen3-VL-8B",
    "qwen3-vl_30b": "Qwen3-VL-30B",
}
METRIC_LABELS = ["S_m1", "S_m2", "C12"]


def get_base_question_id(question_id, subset_modalities):
    if not subset_modalities:
        return question_id
    escaped = [re.escape(m) for m in subset_modalities]
    pattern = rf"_(?:{'|'.join(escaped)})(?:_and_(?:{'|'.join(escaped)}))*$"
    return re.sub(pattern, "", question_id)


def get_question_key(question):
    qid = question.get("question_id", "")
    subs = question.get("subset_modalities", [])
    return (get_base_question_id(qid, subs), question.get("question_text", ""))


def load_and_index(results_dir, models):
    all_data = {}
    for model in models:
        fp = Path(results_dir) / f"{model}_results_vlm.json"
        if not fp.exists():
            print(f"  [WARN] {fp} not found"); continue
        with open(fp, "r", encoding="utf-8") as f:
            entries = json.load(f)
        two_mod = [e for e in entries if len(e.get("gold_modalities", [])) == 2]
        idx = defaultdict(lambda: defaultdict(dict))
        for e in two_mod:
            mp = frozenset(e["gold_modalities"])
            q_key = get_question_key(e)
            subset = frozenset(e.get("subset_modalities", []))
            mods = sorted(mp)
            if subset == mp:           cond = "both"
            elif subset == {mods[0]}: cond = "m1_only"
            elif subset == {mods[1]}: cond = "m2_only"
            else: continue
            idx[mp][q_key][cond] = e
        all_data[model] = dict(idx)
    return all_data


def find_common_keys(all_data, mod_pair, models):
    common = None
    for model in models:
        if model not in all_data or mod_pair not in all_data[model]:
            return []
        ok = {qk for qk, c in all_data[model][mod_pair].items()
              if {"both","m1_only","m2_only"} <= c.keys()}
        common = ok if common is None else (common & ok)
    return sorted(common) if common else []


def build_score_arrays(all_data, mod_pair, common_keys, models):
    """
    Returns (scores, q_meta, m1_name, m2_name).

    m1_name / m2_name are the modality strings that correspond to the
    "m1_only" and "m2_only" keys set by load_and_index (sorted order).
    Callers should use these names to map S_m1 / S_m2 to real modalities
    instead of re-deriving them from the frozenset.
    """
    mods_sorted = sorted(mod_pair)
    m1_name, m2_name = mods_sorted[0], mods_sorted[1]
    n = len(common_keys)
    scores, q_meta = {}, []
    for mi, model in enumerate(models):
        s_b, s_1, s_2 = np.empty(n), np.empty(n), np.empty(n)
        for i, qk in enumerate(common_keys):
            eb = all_data[model][mod_pair][qk]["both"]
            e1 = all_data[model][mod_pair][qk]["m1_only"]
            e2 = all_data[model][mod_pair][qk]["m2_only"]
            s_b[i] = eb.get("score", 0.0)
            s_1[i] = e1.get("score", 0.0)
            s_2[i] = e2.get("score", 0.0)
            if mi == 0:
                q_meta.append((eb.get("answer",""),
                               eb.get("answer_format", eb.get("answer_type","Str"))))
        scores[model] = {"both": s_b, "m1_only": s_1, "m2_only": s_2}
    return scores, q_meta, m1_name, m2_name


def _norm_ans(ans):
    if isinstance(ans, (list, dict)):
        return json.dumps(ans, sort_keys=True, ensure_ascii=True)
    return str(ans)


def precompute_v0_lookup(q_meta, scorer_fn):
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
                    sc = scorer_fn(gold, maj, fmt)
                v0[i, j] = float(sc)
            except Exception:
                v0[i, j] = 0.0
    return v0, q_col


def v0_for_idx(idx, v0_mat, q_col):
    cols = q_col[idx]
    counts = np.bincount(cols, minlength=v0_mat.shape[1])
    return float(v0_mat[idx, int(np.argmax(counts))].mean())


def compute_shape(V12, V1, V2, V0):
    if V12 == 0:
        return np.nan, np.nan, np.nan
    return ((V12-V2+V1-V0)/(2*V12), (V12-V1+V2-V0)/(2*V12), V12-V1-V2+V0)


def run_bootstrap(scores, q_meta, scorer_fn, models, n_boot=N_BOOTSTRAP, seed=SEED):
    n = len(q_meta)
    if n == 0: return {}, {}
    rng = np.random.default_rng(seed)
    print("    Precomputing V0 lookup …")
    v0_mat, q_col = precompute_v0_lookup(q_meta, scorer_fn)
    all_idx = np.arange(n)
    v0_obs = v0_for_idx(all_idx, v0_mat, q_col)
    observed = {}
    for m in models:
        observed[m] = compute_shape(
            scores[m]["both"].mean(), scores[m]["m1_only"].mean(),
            scores[m]["m2_only"].mean(), v0_obs)
    boot = {m: np.empty((n_boot, 3)) for m in models}
    t0 = time.time()
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        v0_b = v0_for_idx(idx, v0_mat, q_col)
        for m in models:
            boot[m][b] = compute_shape(
                scores[m]["both"][idx].mean(), scores[m]["m1_only"][idx].mean(),
                scores[m]["m2_only"][idx].mean(), v0_b)
        if (b+1) % 2000 == 0:
            print(f"      {b+1}/{n_boot}  ({time.time()-t0:.1f}s)")
    return observed, boot


def pairwise_bootstrap_ci(observed, boot, models, mod_pair):
    mods = sorted(mod_pair)
    results = []
    for ma, mb in combinations(models, 2):
        if ma not in observed or mb not in observed: continue
        diff = boot[ma] - boot[mb]
        for k, raw in enumerate(METRIC_LABELS):
            od = observed[ma][k] - observed[mb][k]
            col = diff[:, k]; v = col[~np.isnan(col)]
            if len(v)==0 or np.isnan(od): continue
            lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
            label = raw.replace("m1",mods[0]).replace("m2",mods[1]) if raw!="C12" else "C12"
            results.append({"model_a":ma,"model_b":mb,"metric":label,
                "observed_diff":float(od),"ci_lo":lo,"ci_hi":hi,
                "significant":(lo>0)or(hi<0),"n_valid":int(len(v))})
    return results


def paired_permutation_test(scores, v0_obs, models, mod_pair, observed,
                            n_perm=N_PERMUTATIONS, seed=SEED+1):
    n = next(iter(scores.values()))["both"].shape[0]
    if n == 0: return []
    rng = np.random.default_rng(seed)
    mods = sorted(mod_pair)
    results = []
    for ma, mb in combinations(models, 2):
        if ma not in observed or mb not in observed: continue
        obs_d = np.array([observed[ma][k]-observed[mb][k] for k in range(3)])
        sa, sb = scores[ma], scores[mb]
        pd = np.empty((n_perm, 3))
        for p in range(n_perm):
            sw = rng.random(n) < 0.5
            ba = np.where(sw,sb["both"],sa["both"])
            bb = np.where(sw,sa["both"],sb["both"])
            a1 = np.where(sw,sb["m1_only"],sa["m1_only"])
            b1 = np.where(sw,sa["m1_only"],sb["m1_only"])
            a2 = np.where(sw,sb["m2_only"],sa["m2_only"])
            b2 = np.where(sw,sa["m2_only"],sb["m2_only"])
            sha = compute_shape(ba.mean(),a1.mean(),a2.mean(),v0_obs)
            shb = compute_shape(bb.mean(),b1.mean(),b2.mean(),v0_obs)
            pd[p] = [sha[k]-shb[k] for k in range(3)]
        for k, raw in enumerate(METRIC_LABELS):
            if np.isnan(obs_d[k]): continue
            col = pd[:,k]; v = col[~np.isnan(col)]
            if len(v)==0: continue
            pv = float((np.abs(v)>=np.abs(obs_d[k])).mean())
            label = raw.replace("m1",mods[0]).replace("m2",mods[1]) if raw!="C12" else "C12"
            results.append({"model_a":ma,"model_b":mb,"metric":label,
                "observed_diff":float(obs_d[k]),"p_value":pv})
    return results


def apply_holm(perm_results):
    if not perm_results: return perm_results
    ordered = sorted(perm_results, key=lambda r: r["p_value"])
    m = len(ordered)
    for i, r in enumerate(ordered):
        r["p_value_holm"] = min(r["p_value"]*(m-i), 1.0)
    for i in range(1, m):
        ordered[i]["p_value_holm"] = max(ordered[i]["p_value_holm"],
                                         ordered[i-1]["p_value_holm"])
    for r in ordered:
        r["significant_holm_005"] = r["p_value_holm"] < 0.05
    return ordered


def _ser(obj):
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(type(obj))


def print_summary(all_results, file=None):
    import sys as _s
    out = file or _s.stdout
    def pr(*a, **kw): print(*a, **kw, file=out)
    for bn, bd in all_results.items():
        pr(f"\n{'='*100}"); pr(f"RESULTS — {bn}"); pr('='*100)
        for pl, pd in bd.items():
            ms = pd["modalities"]
            pr(f"\n  {'─'*90}")
            pr(f"  Modality pair: {pl}  (n={pd['n_questions']})")
            pr(f"  {'─'*90}")
            pr(f"\n  Observed SHAPE scores:")
            pr(f"  {'Model':<20s} {'S_'+ms[0]:>12s} {'S_'+ms[1]:>12s} {'C12':>12s}")
            pr(f"  {'-'*56}")
            _f = lambda v: f"{v:.4f}" if v is not None and not (isinstance(v,float) and np.isnan(v)) else "N/A"
            for m in MODEL_NAMES:
                o = pd["observed"].get(m)
                if not o: continue
                pr(f"  {MODEL_DISPLAY.get(m,m):<20s} {_f(o['S_m1']):>12s} {_f(o['S_m2']):>12s} {_f(o['C12']):>12s}")
            pr(f"\n  Per-model bootstrap 95% CIs:")
            pr(f"  {'Model':<20s} {'S_'+ms[0]+' CI':>28s} {'S_'+ms[1]+' CI':>28s} {'C12 CI':>28s}")
            pr(f"  {'-'*104}")
            for m in MODEL_NAMES:
                bs = pd.get("bootstrap_summary",{}).get(m)
                if not bs: continue
                _ci = lambda l,h: f"[{l:.4f}, {h:.4f}]"
                pr(f"  {MODEL_DISPLAY.get(m,m):<20s} {_ci(bs['S_m1_ci_lo'],bs['S_m1_ci_hi']):>28s} "
                   f"{_ci(bs['S_m2_ci_lo'],bs['S_m2_ci_hi']):>28s} {_ci(bs['C12_ci_lo'],bs['C12_ci_hi']):>28s}")
            pr(f"\n  Pairwise — bootstrap 95% CI of difference:")
            pr(f"  {'Comparison':<32s} {'Metric':<14s} {'Obs D':>10s} {'95% CI':>26s} {'Sig?':>6s}")
            pr(f"  {'-'*88}")
            for c in pd.get("bootstrap_comparisons",[]):
                a = MODEL_DISPLAY.get(c["model_a"],c["model_a"])
                b = MODEL_DISPLAY.get(c["model_b"],c["model_b"])
                ci = f"[{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]"
                sig = "YES" if c["significant"] else "no"
                pr(f"  {a+' - '+b:<32s} {c['metric']:<14s} {c['observed_diff']:>10.4f} {ci:>26s} {sig:>6s}")
            perm = pd.get("permutation_tests",[])
            if perm:
                pr(f"\n  Permutation tests (Holm-corrected):")
                pr(f"  {'Comparison':<32s} {'Metric':<14s} {'Obs D':>10s} {'p-raw':>10s} {'p-Holm':>10s} {'Sig?':>6s}")
                pr(f"  {'-'*82}")
                for r in perm:
                    a = MODEL_DISPLAY.get(r["model_a"],r["model_a"])
                    b = MODEL_DISPLAY.get(r["model_b"],r["model_b"])
                    sig = "YES" if r.get("significant_holm_005",False) else "no"
                    pr(f"  {a+' - '+b:<32s} {r['metric']:<14s} {r['observed_diff']:>10.4f} "
                       f"{r['p_value']:>10.4f} {r.get('p_value_holm',r['p_value']):>10.4f} {sig:>6s}")


def main():
    repo = Path(__file__).parent.parent
    out_dir = Path(__file__).parent
    benchmarks = {
        "LongDocURL": (repo/"results_longdocurl", ldu_eval_score),
        "MMLongBench-Doc": (repo/"results_mmlongbench", mmlong_eval_score),
    }
    all_results = {}
    for bn, (rdir, sfn) in benchmarks.items():
        print(f"\n{'='*80}\nBENCHMARK: {bn}\n{'='*80}")
        print("  Loading …")
        ad = load_and_index(rdir, MODEL_NAMES)
        avail = [m for m in MODEL_NAMES if m in ad]
        print(f"  Models: {avail}")
        mps = set()
        for m in avail: mps |= set(ad[m].keys())
        mps = sorted(mps, key=lambda x: str(sorted(x)))
        br, all_perm = {}, []
        for mp in mps:
            ms = sorted(mp); pl = f"{ms[0]} + {ms[1]}"
            print(f"\n  -- {pl} --")
            ck = find_common_keys(ad, mp, avail)
            print(f"    Common questions: {len(ck)}")
            if len(ck) < 2: print("    SKIP"); continue
            sa, qm, m1_name, m2_name = build_score_arrays(ad, mp, ck, avail)
            ms = [m1_name, m2_name]
            print("    Bootstrap …")
            obs, bd = run_bootstrap(sa, qm, sfn, avail)
            bsum = {}
            for m in avail:
                d = bd[m]
                def _pci(c):
                    v=c[~np.isnan(c)]
                    if len(v)==0: return np.nan,np.nan,np.nan,np.nan
                    return float(np.mean(v)),float(np.std(v)),float(np.percentile(v,2.5)),float(np.percentile(v,97.5))
                s1m,s1s,s1l,s1h = _pci(d[:,0])
                s2m,s2s,s2l,s2h = _pci(d[:,1])
                cm,cs,cl,ch = _pci(d[:,2])
                bsum[m] = {"S_m1_mean":s1m,"S_m1_std":s1s,"S_m1_ci_lo":s1l,"S_m1_ci_hi":s1h,
                           "S_m2_mean":s2m,"S_m2_std":s2s,"S_m2_ci_lo":s2l,"S_m2_ci_hi":s2h,
                           "C12_mean":cm,"C12_std":cs,"C12_ci_lo":cl,"C12_ci_hi":ch}
            comp = pairwise_bootstrap_ci(obs, bd, avail, mp)
            print("    Permutation test …")
            v0m, qc = precompute_v0_lookup(qm, sfn)
            v0o = v0_for_idx(np.arange(len(qm)), v0m, qc)
            perm = paired_permutation_test(sa, v0o, avail, mp, obs)
            all_perm.extend(perm)
            br[pl] = {"n_questions":len(ck),"modalities":ms,
                "observed":{m:{"S_m1":float(v[0]),"S_m2":float(v[1]),"C12":float(v[2])} for m,v in obs.items()},
                "bootstrap_summary":bsum,"bootstrap_comparisons":comp,"permutation_tests":perm}
        print(f"\n  Holm correction across {len(all_perm)} tests …")
        corrected = apply_holm(all_perm)
        hl = {}
        for r in corrected:
            hl[(r["model_a"],r["model_b"],r["metric"])] = {
                "p_value_holm":r["p_value_holm"],"significant_holm_005":r["significant_holm_005"]}
        for pdata in br.values():
            for pr in pdata.get("permutation_tests",[]):
                k = (pr["model_a"],pr["model_b"],pr["metric"])
                if k in hl: pr.update(hl[k])
        all_results[bn] = br
    jp = out_dir/"bootstrap_results.json"
    with open(jp,"w",encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=_ser)
    print(f"\nJSON: {jp}")
    tp = out_dir/"bootstrap_results.txt"
    with open(tp,"w",encoding="utf-8") as f:
        print_summary(all_results, file=f)
    print(f"Report: {tp}")
    print_summary(all_results)


if __name__ == "__main__":
    main()
