#!/usr/bin/env python3
"""
SHAPE Metric Calculator for Multimodal Models

This script calculates the SHAPE
metric for multimodal models based on results from the evaluation pipeline.

The SHAPE metric computes Shapley values (marginal contributions) for each modality
and a cooperation term to understand how modalities contribute to model performance.

For 2 modalities M1 and M2:
- V12 = V(M1, M2): accuracy with both modalities
- V1 = V(M1, 0_2): accuracy with only modality 1
- V2 = V(0_1, M2): accuracy with only modality 2  
- V0 = V(0_1, 0_2): baseline accuracy (majority class)

Shapley contributions:
- φ1 = 0.5 * [(V1 - V0) + (V12 - V2)]
- φ2 = 0.5 * [(V2 - V0) + (V12 - V1)]

SHAPE scores (normalized, with Zf = V12):
- S1 = (V12 - V2 + V1 - V0) / (2 * Zf)
- S2 = (V12 - V1 + V2 - V0) / (2 * Zf)

Cooperation score:
- C12 = V12 - V1 - V2 + V0
"""

import json
import sys
import re
import io
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean
from typing import Dict, List, Any, Optional
from contextlib import redirect_stdout

# Add parent directory to path to import eval modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from mmlongbench.eval.eval_score import eval_score as mmlong_eval_score
from longdocurl.eval import eval_score as ldu_eval_score


def load_results(filepath: str) -> List[Dict]:
    """Load results from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_modality_set(modalities: List[str]) -> frozenset:
    """Convert a list of modalities to a frozenset for grouping."""
    return frozenset(modalities)


def filter_two_modality_questions(results: List[Dict]) -> List[Dict]:
    """Filter questions that have exactly 2 modalities in gold_modalities."""
    return [
        q for q in results 
        if len(q.get('gold_modalities', [])) == 2
    ]


def group_by_modality_pairs(questions: List[Dict]) -> Dict[frozenset, List[Dict]]:
    """Group questions by their gold_modalities pair."""
    groups = defaultdict(list)
    for q in questions:
        modality_pair = get_modality_set(q['gold_modalities'])
        groups[modality_pair].append(q)
    return dict(groups)


def calculate_accuracy(questions: List[Dict]) -> float:
    """Calculate mean accuracy (score) for a list of questions."""
    if not questions:
        return 0.0
    scores = [q.get('score', 0.0) for q in questions]
    return mean(scores)


def get_base_question_id(question_id: str, subset_modalities: List[str]) -> str:
    """Return the question_id without modality suffixes."""
    if not subset_modalities:
        return question_id
    escaped = [re.escape(modality) for modality in subset_modalities]
    pattern = rf"_(?:{'|'.join(escaped)})(?:_and_(?:{'|'.join(escaped)}))*$"
    return re.sub(pattern, "", question_id)


def get_question_key(question: Dict) -> tuple:
    """Return a stable key for matching questions across modality conditions."""
    question_id = question.get('question_id', '')
    subset_modalities = question.get('subset_modalities', [])
    base_id = get_base_question_id(question_id, subset_modalities)
    return (
        base_id,
        question.get('question_text', '')
    )


def get_questions_with_both_modalities(
    questions: List[Dict], 
    modality_pair: frozenset
) -> List[Dict]:
    """Get questions where subset_modalities is exactly the modality pair."""
    target = set(modality_pair)
    return [q for q in questions if set(q.get('subset_modalities', [])) == target]


def get_questions_with_single_modality(
    questions: List[Dict], 
    target_modality: str,
    modality_pair: frozenset
) -> List[Dict]:
    """
    Get questions where subset_modalities is exactly {target_modality}.
    """
    target = {target_modality}
    return [q for q in questions if set(q.get('subset_modalities', [])) == target]


def calculate_majority_baseline_score(
    questions: List[Dict],
    question_keys: set,
    scorer_fn=None
) -> float:
    """
    Calculate V0 (both modalities absent) by scoring constant majority-answer predictions
    using the actual benchmark scorer.
    
    Args:
        questions: List of question dicts
        question_keys: Set of question keys to include
        scorer_fn: The eval_score function to use (mmlong_eval_score or ldu_eval_score)
    
    For the subset of questions (filtered by question_keys):
    1. Find the most frequent ground-truth answer string
    2. For each question, compute the score using scorer_fn(gold, majority_answer, answer_type)
    3. Return the mean score across all questions
    
    This properly handles all benchmark scoring types.
    """
    if scorer_fn is None:
        scorer_fn = mmlong_eval_score
    if not questions or not question_keys:
        return 0.0
    
    # Filter to only the questions we're evaluating (by question_id + question_text)
    filtered_questions = [
        q for q in questions
        if get_question_key(q) in question_keys
    ]
    
    if not filtered_questions:
        return 0.0
    
    # Get unique questions by question_id + question_text (take first occurrence)
    unique_questions = {}
    for q in filtered_questions:
        q_key = get_question_key(q)
        if q_key not in unique_questions:
            unique_questions[q_key] = q
    
    questions_list = list(unique_questions.values())
    
    # Get all gold answers and find the majority
    def normalize_answer(answer: Any) -> str:
        if isinstance(answer, (list, dict)):
            return json.dumps(answer, sort_keys=True, ensure_ascii=True)
        return str(answer)

    answers = [normalize_answer(q.get('answer', '')) for q in questions_list]
    answer_counts = Counter(answers)
    majority_answer, _ = answer_counts.most_common(1)[0]
    
    # For each question, compute the score if we predicted the majority answer
    # using the actual benchmark scorer
    total_score = 0.0
    for q in questions_list:
        gold_answer = q.get('answer', '')
        # Get answer format/type from the question (field is 'answer_format' in our data)
        answer_type = q.get('answer_format', q.get('answer_type', 'Str'))
        
        # Use the benchmark's eval_score function, suppressing debug prints
        # and catching edge-case crashes gracefully
        try:
            devnull = io.StringIO()
            with redirect_stdout(devnull):
                score = scorer_fn(gold_answer, majority_answer, answer_type)
            total_score += float(score)
        except Exception:
            total_score += 0.0
    
    return total_score / len(questions_list)


def shape_two_modalities(
    V12: float, 
    V1: float, 
    V2: float, 
    V0: float,
    modality1: str,
    modality2: str
) -> Optional[Dict[str, Any]]:
    """
    Calculate SHAPE metrics for two modalities.
    
    Args:
        V12: Accuracy with both modalities
        V1: Accuracy with only modality 1
        V2: Accuracy with only modality 2
        V0: Baseline accuracy (both absent, majority class)
        modality1: Name of first modality
        modality2: Name of second modality
    
    Returns:
        Dictionary with SHAPE metrics, or None if V12 is 0 (cannot normalize)
    """
    # Cannot compute SHAPE scores if V12 is 0 (division by zero)
    if V12 == 0:
        return None
    
    Zf = V12
    
    # Unscaled Shapley contributions
    phi1 = 0.5 * ((V1 - V0) + (V12 - V2))
    phi2 = 0.5 * ((V2 - V0) + (V12 - V1))
    
    # Scaled SHAPE scores (paper form with Zf in denominator)
    S1 = (V12 - V2 + V1 - V0) / (2 * Zf)
    S2 = (V12 - V1 + V2 - V0) / (2 * Zf)
    
    # Cooperation score
    C12 = V12 - V1 - V2 + V0
    
    return {
        f"S_{modality1}": S1,
        f"S_{modality2}": S2,
        f"phi_{modality1}": phi1,
        f"phi_{modality2}": phi2,
        "C12": C12,
        "V12": V12,
        f"V_{modality1}": V1,
        f"V_{modality2}": V2,
        "V0": V0
    }


def analyze_modality_group(
    questions: List[Dict],
    modality_pair: frozenset,
    scorer_fn=None
) -> Dict[str, Any]:
    """
    Analyze a group of questions with a specific modality pair.
    
    Returns accuracy metrics and SHAPE scores for the modality pair.
    Ensures we compare the SAME questions (by base question_id + question_text) across all conditions.
    """
    modalities = sorted(list(modality_pair))
    m1, m2 = modalities[0], modalities[1]
    
    # Get questions for each condition
    q_both = get_questions_with_both_modalities(questions, modality_pair)
    q_m1_only = get_questions_with_single_modality(questions, m1, modality_pair)
    q_m2_only = get_questions_with_single_modality(questions, m2, modality_pair)
    
    # Extract question keys (question_id + question_text) for each condition
    keys_both = {get_question_key(q) for q in q_both}
    keys_m1 = {get_question_key(q) for q in q_m1_only}
    keys_m2 = {get_question_key(q) for q in q_m2_only}
    
    # Find common question keys that appear in ALL three conditions
    common_keys = keys_both & keys_m1 & keys_m2
    
    # Filter questions to only those with common keys
    q_both_filtered = [q for q in q_both if get_question_key(q) in common_keys]
    q_m1_filtered = [q for q in q_m1_only if get_question_key(q) in common_keys]
    q_m2_filtered = [q for q in q_m2_only if get_question_key(q) in common_keys]
    
    # Calculate accuracies on the filtered (matched) questions
    V12 = calculate_accuracy(q_both_filtered) if q_both_filtered else None
    V1 = calculate_accuracy(q_m1_filtered) if q_m1_filtered else None
    V2 = calculate_accuracy(q_m2_filtered) if q_m2_filtered else None
    
    # Calculate majority baseline V0 using the common questions
    # This scores constant predictions (majority answer) with the real evaluator
    V0 = calculate_majority_baseline_score(questions, common_keys, scorer_fn=scorer_fn) if common_keys else 0.0
    
    # Get unique question count
    n_unique_questions = len({get_question_key(q) for q in questions})
    
    result = {
        "modality_pair": f"{{{m1}, {m2}}}",
        "modalities": modalities,
        "total_questions_in_group": len(questions),
        "unique_question_ids": n_unique_questions,
        "n_common_questions": len(common_keys),
        "n_both_modalities_raw": len(q_both),
        f"n_{m1}_only_raw": len(q_m1_only),
        f"n_{m2}_only_raw": len(q_m2_only),
        "n_both_modalities_matched": len(q_both_filtered),
        f"n_{m1}_only_matched": len(q_m1_filtered),
        f"n_{m2}_only_matched": len(q_m2_filtered),
        "V12_accuracy_both": V12,
        f"V1_accuracy_{m1}_only": V1,
        f"V2_accuracy_{m2}_only": V2,
        "V0_majority_baseline": V0,
    }
    
    # Calculate SHAPE metrics if we have all required data
    if V12 is not None and V1 is not None and V2 is not None:
        shape_metrics = shape_two_modalities(V12, V1, V2, V0, m1, m2)
        if shape_metrics is not None:
            result.update({
                f"S_{m1}": shape_metrics[f"S_{m1}"],
                f"S_{m2}": shape_metrics[f"S_{m2}"],
                f"phi_{m1}": shape_metrics[f"phi_{m1}"],
                f"phi_{m2}": shape_metrics[f"phi_{m2}"],
                "C12_cooperation": shape_metrics["C12"],
            })
        else:
            result["shape_computable"] = False
            result["reason"] = "V12 (accuracy with both modalities) is 0, cannot normalize SHAPE scores"
    else:
        result["shape_computable"] = False
        result["missing_data"] = []
        if V12 is None:
            result["missing_data"].append("V12 (both modalities) - no common questions")
        if V1 is None:
            result["missing_data"].append(f"V1 ({m1} only) - no common questions")
        if V2 is None:
            result["missing_data"].append(f"V2 ({m2} only) - no common questions")
    
    return result


def analyze_model_results(
    results_path: str,
    output_path: Optional[str] = None,
    scorer_fn=None
) -> Dict[str, Any]:
    """
    Analyze a model's results and compute SHAPE metrics for all modality pairs.
    
    Args:
        results_path: Path to the model's results JSON file
        output_path: Optional path to save the analysis results
        scorer_fn: The eval_score function to use (mmlong_eval_score or ldu_eval_score)
    
    Returns:
        Dictionary with complete analysis results
    """
    # Load results
    results = load_results(results_path)
    
    # Extract model name from filename
    model_name = Path(results_path).stem.replace('_results_vlm', '')
    
    # Filter to 2-modality questions
    two_mod_questions = filter_two_modality_questions(results)
    
    # Group by modality pairs
    modality_groups = group_by_modality_pairs(two_mod_questions)
    
    # Analyze each group
    group_analyses = []
    for modality_pair, questions in sorted(modality_groups.items(), key=lambda x: str(x[0])):
        analysis = analyze_modality_group(questions, modality_pair, scorer_fn=scorer_fn)
        group_analyses.append(analysis)
    
    # Summary statistics
    total_2mod_questions = len(two_mod_questions)
    total_questions = len(results)
    
    analysis_result = {
        "model": model_name,
        "results_file": results_path,
        "total_questions_in_file": total_questions,
        "total_2_modality_questions": total_2mod_questions,
        "num_modality_groups": len(modality_groups),
        "modality_groups": group_analyses
    }
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    return analysis_result


def print_analysis_report(analysis: Dict[str, Any]) -> None:
    """Print a formatted report of the SHAPE analysis."""
    print("=" * 80)
    print(f"SHAPE METRIC ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nModel: {analysis['model']}")
    print(f"Results file: {analysis['results_file']}")
    print(f"Total questions: {analysis['total_questions_in_file']}")
    print(f"Questions with 2 gold modalities: {analysis['total_2_modality_questions']}")
    print(f"Number of modality groups: {analysis['num_modality_groups']}")
    
    print("\n" + "-" * 80)
    print("MODALITY GROUP ANALYSIS")
    print("-" * 80)
    
    for group in analysis['modality_groups']:
        m1, m2 = group['modalities']
        print(f"\n{'='*60}")
        print(f"Modality Pair: {group['modality_pair']}")
        print(f"{'='*60}")
        print(f"  Total questions in group: {group['total_questions_in_group']}")
        print(f"  Unique question IDs: {group['unique_question_ids']}")
        print(f"  Common questions (matched across all conditions): {group['n_common_questions']}")
        print(f"\n  RAW COUNTS (before matching):")
        print(f"    Both modalities: {group['n_both_modalities_raw']}")
        print(f"    {m1} only: {group.get(f'n_{m1}_only_raw', 'N/A')}")
        print(f"    {m2} only: {group.get(f'n_{m2}_only_raw', 'N/A')}")
        print(f"\n  MATCHED COUNTS (same question_ids):")
        print(f"    Both modalities: {group['n_both_modalities_matched']}")
        print(f"    {m1} only: {group.get(f'n_{m1}_only_matched', 'N/A')}")
        print(f"    {m2} only: {group.get(f'n_{m2}_only_matched', 'N/A')}")
        
        print(f"\n  ACCURACY METRICS (on matched questions):")
        print(f"    V12 (both modalities):  {group.get('V12_accuracy_both', 'N/A'):.4f}" if group.get('V12_accuracy_both') is not None else "    V12 (both modalities):  N/A")
        print(f"    V1 ({m1} only):         {group.get(f'V1_accuracy_{m1}_only', 'N/A'):.4f}" if group.get(f'V1_accuracy_{m1}_only') is not None else f"    V1 ({m1} only):         N/A")
        print(f"    V2 ({m2} only):         {group.get(f'V2_accuracy_{m2}_only', 'N/A'):.4f}" if group.get(f'V2_accuracy_{m2}_only') is not None else f"    V2 ({m2} only):         N/A")
        print(f"    V0 (majority baseline): {group.get('V0_majority_baseline', 0):.4f}")
        
        if group.get('shape_computable', True) and f'S_{m1}' in group:
            print(f"\n  SHAPE METRICS:")
            print(f"    Shapley value φ_{m1}: {group.get(f'phi_{m1}', 0):.4f}")
            print(f"    Shapley value φ_{m2}: {group.get(f'phi_{m2}', 0):.4f}")
            print(f"    SHAPE score S_{m1}:   {group.get(f'S_{m1}', 0):.4f} ({group.get(f'S_{m1}', 0)*100:.2f}%)")
            print(f"    SHAPE score S_{m2}:   {group.get(f'S_{m2}', 0):.4f} ({group.get(f'S_{m2}', 0)*100:.2f}%)")
            print(f"    Cooperation C12:      {group.get('C12_cooperation', 0):.4f}")
            
            # Interpretation
            c12 = group.get('C12_cooperation', 0)
            if c12 > 0.01:
                print("    → Synergy: modalities cooperate positively")
            elif c12 < -0.01:
                print("    → Redundancy: modalities interfere/overlap")
            else:
                print("    → Independent: modalities contribute independently")
        else:
            print("\n  SHAPE METRICS: Cannot compute")
            if 'reason' in group:
                print(f"    Reason: {group['reason']}")
            if 'missing_data' in group:
                for missing in group['missing_data']:
                    print(f"    - Missing: {missing}")
    
    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


# ── Display mappings for summary tables ─────────────────────────────────────────
MODEL_DISPLAY_ORDER = ["gemma3_4b", "gemma3_27b", "gpt-4o-mini", "qwen3-vl_8b", "qwen3-vl_30b"]
MODEL_DISPLAY_NAMES = {
    "gemma3_4b": "Gemma3-4B",
    "gemma3_27b": "Gemma3-27B",
    "gpt-4o-mini": "GPT-4o-mini",
    "qwen3-vl_8b": "Qwen3-VL-8B",
    "qwen3-vl_30b": "Qwen3-VL-30B",
}

MODALITY_PAIR_ORDER = [
    ("image", "layout"),
    ("image", "plain_text"),
    ("layout", "plain_text"),
    ("plain_text", "table"),
    ("image", "table"),
    ("layout", "table"),
]

MODALITY_SHORT = {
    "image": "img",
    "layout": "lay",
    "plain_text": "txt",
    "table": "table",
}

PAIR_HEADERS = {
    ("image", "layout"): "Img + Lay",
    ("image", "plain_text"): "Img + Txt",
    ("layout", "plain_text"): "Lay + Txt",
    ("plain_text", "table"): "Txt + Tab",
    ("image", "table"): "Img + Tab",
    ("layout", "table"): "Lay + Tab",
}


def collect_shape_scores(
    analyses: List[Dict[str, Any]]
) -> Dict[str, Dict[tuple, Dict[str, float]]]:
    """
    From a list of per-model analyses, build:
      { model_name: { (m1, m2): {"S_m1": val, "S_m2": val, "C12": val}, ... } }
    """
    result = {}
    for analysis in analyses:
        model = analysis["model"]
        pairs = {}
        for group in analysis["modality_groups"]:
            m1, m2 = group["modalities"]
            pair_key = (m1, m2)
            s1 = group.get(f"S_{m1}")
            s2 = group.get(f"S_{m2}")
            c12 = group.get("C12_cooperation")
            phi1 = group.get(f"phi_{m1}")
            phi2 = group.get(f"phi_{m2}")
            pairs[pair_key] = {"S_m1": s1, "S_m2": s2, "C12": c12, "phi_m1": phi1, "phi_m2": phi2}
        result[model] = pairs
    return result


def print_shape_summary_table(
    bench_name: str,
    analyses: List[Dict[str, Any]]
) -> None:
    """
    Print a summary table of SHAPE scores (S) across all models and modality pairs.
    Format matches the reference image: rows = models, column groups = modality pairs,
    each with S(mod1) and S(mod2) sub-columns. Values are plain decimals (not %).
    The dominant modality (higher S) per pair is marked with * .
    """
    scores = collect_shape_scores(analyses)
    models = [m for m in MODEL_DISPLAY_ORDER if m in scores]
    pairs = MODALITY_PAIR_ORDER

    # Column widths
    model_col_w = 16
    val_col_w = 10

    # ── Build header rows ─────────────────────────────────────────────────
    # Row 1: pair group headers
    header1 = f"{'':>{model_col_w}}"
    for m1, m2 in pairs:
        pair_label = PAIR_HEADERS.get((m1, m2), f"{m1}+{m2}")
        span = 2 * val_col_w + 3  # two value columns + separator
        header1 += f"  {pair_label:^{span}s}"

    # Row 2: sub-column headers S(mod1) S(mod2)
    header2 = f"{'Model':>{model_col_w}}"
    for m1, m2 in pairs:
        s1_label = f"S({MODALITY_SHORT[m1]})"
        s2_label = f"S({MODALITY_SHORT[m2]})"
        header2 += f"  {s1_label:>{val_col_w}s} {s2_label:>{val_col_w}s}"

    sep = "-" * len(header2)

    # ── Print ──────────────────────────────────────────────────────────────
    print()
    print("=" * len(header2))
    print(f"Marginal Contribution Scores (S) — {bench_name}")
    print("=" * len(header2))
    print()
    print(header1)
    print(sep)
    print(header2)
    print(sep)

    for model in models:
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        row = f"{display_name:>{model_col_w}}"
        model_scores = scores.get(model, {})

        for m1, m2 in pairs:
            pair_data = model_scores.get((m1, m2), {})
            s1 = pair_data.get("S_m1")
            s2 = pair_data.get("S_m2")

            s1_str = f"{s1:.4f}" if s1 is not None else "N/A"
            s2_str = f"{s2:.4f}" if s2 is not None else "N/A"

            # Mark dominant modality with *
            if s1 is not None and s2 is not None:
                if s1 > s2:
                    s1_str += "*"
                elif s2 > s1:
                    s2_str += "*"

            row += f"  {s1_str:>{val_col_w}s} {s2_str:>{val_col_w}s}"

        print(row)

    print(sep)
    print(f"{'(* = dominant modality in the pair)':>{model_col_w + 40}s}")
    print()


def print_phi_summary_table(
    bench_name: str,
    analyses: List[Dict[str, Any]]
) -> None:
    """
    Print a summary table of unnormalized Shapley values (φ) across all models and modality pairs.
    Same layout as the S table but with φ values.
    """
    scores = collect_shape_scores(analyses)
    models = [m for m in MODEL_DISPLAY_ORDER if m in scores]
    pairs = MODALITY_PAIR_ORDER

    model_col_w = 16
    val_col_w = 10

    header1 = f"{'':>{model_col_w}}"
    for m1, m2 in pairs:
        pair_label = PAIR_HEADERS.get((m1, m2), f"{m1}+{m2}")
        span = 2 * val_col_w + 3
        header1 += f"  {pair_label:^{span}s}"

    header2 = f"{'Model':>{model_col_w}}"
    for m1, m2 in pairs:
        p1_label = f"φ({MODALITY_SHORT[m1]})"
        p2_label = f"φ({MODALITY_SHORT[m2]})"
        header2 += f"  {p1_label:>{val_col_w}s} {p2_label:>{val_col_w}s}"

    sep = "-" * len(header2)

    print()
    print("=" * len(header2))
    print(f"Absolute Shapley Contributions (φ) — {bench_name}")
    print("=" * len(header2))
    print()
    print(header1)
    print(sep)
    print(header2)
    print(sep)

    for model in models:
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        row = f"{display_name:>{model_col_w}}"
        model_scores = scores.get(model, {})

        for m1, m2 in pairs:
            pair_data = model_scores.get((m1, m2), {})
            p1 = pair_data.get("phi_m1")
            p2 = pair_data.get("phi_m2")

            p1_str = f"{p1:.4f}" if p1 is not None else "N/A"
            p2_str = f"{p2:.4f}" if p2 is not None else "N/A"

            if p1 is not None and p2 is not None:
                if p1 > p2:
                    p1_str += "*"
                elif p2 > p1:
                    p2_str += "*"

            row += f"  {p1_str:>{val_col_w}s} {p2_str:>{val_col_w}s}"

        print(row)

    print(sep)
    print(f"{'(* = dominant modality in the pair)':>{model_col_w + 40}s}")
    print()


def print_cooperation_summary_table(
    bench_name: str,
    analyses: List[Dict[str, Any]]
) -> None:
    """
    Print a summary table of Cooperation scores (C12) across all models and modality pairs.
    """
    scores = collect_shape_scores(analyses)
    models = [m for m in MODEL_DISPLAY_ORDER if m in scores]
    pairs = MODALITY_PAIR_ORDER

    model_col_w = 16
    val_col_w = 12

    header = f"{'Model':>{model_col_w}}"
    for m1, m2 in pairs:
        pair_label = PAIR_HEADERS.get((m1, m2), f"{m1}+{m2}")
        header += f"  {pair_label:>{val_col_w}s}"

    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print(f"Cooperation Scores (C12) — {bench_name}")
    print("=" * len(header))
    print()
    print(header)
    print(sep)

    for model in models:
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        row = f"{display_name:>{model_col_w}}"
        model_scores = scores.get(model, {})

        for m1, m2 in pairs:
            pair_data = model_scores.get((m1, m2), {})
            c12 = pair_data.get("C12")
            c12_str = f"{c12:.4f}" if c12 is not None else "N/A"
            row += f"  {c12_str:>{val_col_w}s}"

        print(row)

    print(sep)
    print()


def main():
    import glob

    repo_root = Path(__file__).parent.parent
    benchmarks = {
        "LongDocURL": (repo_root / "results_longdocurl", ldu_eval_score),
        "MMLongBench-Doc": (repo_root / "results_mmlongbench", mmlong_eval_score),
    }

    output_dir = Path(__file__).parent / "shape_metric_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "shape_metric_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        import sys as _sys
        original_stdout = _sys.stdout
        _sys.stdout = f

        # Collect all analyses per benchmark for summary tables
        all_analyses = {}

        for bench_name, (bench_dir, scorer_fn) in benchmarks.items():
            print("=" * 110)
            print(f"BENCHMARK: {bench_name}")
            print("=" * 110)
            files = sorted(glob.glob(str(bench_dir / "*_results_vlm.json")))
            if not files:
                print(f"  No *_results_vlm.json files found in {bench_dir}")
                continue
            bench_analyses = []
            for path in files:
                analysis = analyze_model_results(path, scorer_fn=scorer_fn)
                print_analysis_report(analysis)
                bench_analyses.append(analysis)
            all_analyses[bench_name] = bench_analyses
            print()

        # Print summary tables at the end
        for bench_name, analyses in all_analyses.items():
            print_shape_summary_table(bench_name, analyses)
            print_phi_summary_table(bench_name, analyses)
            print_cooperation_summary_table(bench_name, analyses)

        _sys.stdout = original_stdout
    print(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
