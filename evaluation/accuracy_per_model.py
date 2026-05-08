import json
import glob
import os
from collections import defaultdict


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BENCHMARKS = {
    "LongDocURL": os.path.join(BASE_DIR, "results_longdocurl"),
    "MMLongBench-Doc": os.path.join(BASE_DIR, "results_mmlongbench"),
}


def get_model_name(filename: str) -> str:
    return filename.replace("_results_vlm.json", "")


def load_results(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def is_full_gold_subset(item: dict) -> bool:
    """Return True if the item's subset_modalities equals its gold_modalities."""
    gold = sorted(item.get("gold_modalities", []))
    subset = sorted(item.get("subset_modalities", []))
    return gold == subset


def modality_pair_key(modalities: list[str]) -> str:
    """Create a canonical string key for a modality combination."""
    return " + ".join(sorted(modalities))


def collect_benchmark_data(results_dir: str) -> dict:
    """Collect accuracy data for all models in a results directory.

    Returns: {model_name: {"overall": (n, avg), combo_key: (n, avg), ...}}
    """
    files = sorted(glob.glob(os.path.join(results_dir, "*_results_vlm.json")))
    all_combos = set()
    model_data = {}

    for fpath in files:
        model = get_model_name(os.path.basename(fpath))
        data = load_results(fpath)

        # Filter: 2 gold modalities AND subset == gold (full gold set only)
        two_mod_full = [
            q for q in data
            if len(q.get("gold_modalities", [])) == 2 and is_full_gold_subset(q)
        ]

        # Overall accuracy
        scores = [q.get("score", 0.0) for q in two_mod_full]
        total = len(scores)
        avg_score = sum(scores) / total if total > 0 else 0.0

        # Per modality combination
        combo_scores = defaultdict(list)
        for q in two_mod_full:
            key = modality_pair_key(q["gold_modalities"])
            combo_scores[key].append(q.get("score", 0.0))
            all_combos.add(key)

        entry = {"overall": (total, avg_score)}
        for combo, s in combo_scores.items():
            entry[combo] = (len(s), sum(s) / len(s) if s else 0.0)

        model_data[model] = entry

    return model_data, sorted(all_combos)


def print_table(bench_name: str, results_dir: str):
    model_data, combos = collect_benchmark_data(results_dir)
    models = list(model_data.keys())

    print()
    print("=" * 90)
    print(f"BENCHMARK: {bench_name} ({os.path.relpath(results_dir, BASE_DIR)})")
    print("=" * 90)

    # ── Table 1: Accuracy per modality combination (models as columns) ────
    col_w = 16
    header_combo = f"{'Modality Combination':<30s}"
    for m in models:
        header_combo += f" | {m:>{col_w}s}"
    sep = "-" * len(header_combo)

    print(f"\n  Table: Accuracy (%) per modality combination\n")
    print(f"  {header_combo}")
    print(f"  {sep}")

    for combo in combos:
        row = f"  {combo:<30s}"
        for m in models:
            n, avg = model_data[m].get(combo, (0, 0.0))
            row += f" | {avg*100:>{col_w - 6}.2f}% (n={n:<3d})"
        print(row)

    # Overall row
    row = f"  {'OVERALL':<30s}"
    for m in models:
        n, avg = model_data[m]["overall"]
        row += f" | {avg*100:>{col_w - 6}.2f}% (n={n:<3d})"
    print(f"  {sep}")
    print(row)
    print()

    # ── Table 2: Accuracy only (compact, no n) ───────────────────────────
    col_w2 = 14
    header2 = f"{'Modality Combination':<30s}"
    for m in models:
        header2 += f" | {m:>{col_w2}s}"
    sep2 = "-" * len(header2)

    print(f"  Compact table: Accuracy (%) only\n")
    print(f"  {header2}")
    print(f"  {sep2}")

    for combo in combos:
        row = f"  {combo:<30s}"
        for m in models:
            _, avg = model_data[m].get(combo, (0, 0.0))
            row += f" | {avg*100:>{col_w2}.2f}"
        print(row)

    row = f"  {'OVERALL':<30s}"
    for m in models:
        _, avg = model_data[m]["overall"]
        row += f" | {avg*100:>{col_w2}.2f}"
    print(f"  {sep2}")
    print(row)
    print()


def main():
    for bench_name, results_dir in BENCHMARKS.items():
        print_table(bench_name, results_dir)


if __name__ == "__main__":
    main()
