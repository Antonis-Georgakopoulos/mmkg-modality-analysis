#!/usr/bin/env python
"""
Create samples_2modalities.json from samples.json for MMLongBench-Doc.

Steps:
1. Parse evidence_sources for each question
2. Map evidence_sources to internal modality types:
       Chart                      -> image
       Figure                     -> image
       Table                      -> table
       Pure-text (Plain-text)     -> plain_text  (grouped: text, list)
       Generalized-text (Layout)  -> layout      (grouped: header, footer, ...)
3. Keep only questions with exactly 2 unique mapped modalities

Note: "Chart" and "Figure" both map to "image", so a question with
['Chart', 'Figure', 'Pure-text (Plain-text)'] has 2 modalities (image + plain_text)
and IS kept.

Output: data/samples_2modalities.json
"""

import json
from pathlib import Path
from collections import Counter

# ── Modality mapping (mirrors config.py) ─────────────────────────────────────

MODALITY_MAPPING = {
    'Chart': ['image'],
    'Figure': ['image'],
    'Table': ['table'],
    'Pure-text (Plain-text)': ['text', 'list'],
    'Generalized-text (Layout)': ['header', 'footer', 'page_number', 'page_footnote'],
}

GROUPED_MODALITIES = {
    'layout': ['header', 'footer', 'page_number', 'page_footnote'],
    'plain_text': ['text', 'list'],
}

# Reverse mapping: internal modality -> group name
INTERNAL_TO_GROUP = {}
for group_name, internal_mods in GROUPED_MODALITIES.items():
    for mod in internal_mods:
        INTERNAL_TO_GROUP[mod] = group_name


# ── Mapping function ─────────────────────────────────────────────────────────

def map_to_modality_types(evidence_sources: list) -> list:
    """
    Map benchmark evidence_sources to internal modality types.
    Uses grouped names (plain_text, layout) where applicable.
    Returns list of unique modality types.
    """
    modality_types = []
    for source in evidence_sources:
        source = source.strip()
        if source in MODALITY_MAPPING:
            internal_mods = MODALITY_MAPPING[source]
            if internal_mods and internal_mods[0] in INTERNAL_TO_GROUP:
                group_name = INTERNAL_TO_GROUP[internal_mods[0]]
                if group_name not in modality_types:
                    modality_types.append(group_name)
            else:
                for mod in internal_mods:
                    if mod not in modality_types:
                        modality_types.append(mod)
    return modality_types


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    input_path = Path(__file__).parent / "data" / "samples.json"
    output_path = Path(__file__).parent / "data" / "samples_2modalities.json"

    print(f"Reading from: {input_path}")

    with open(input_path, 'r') as f:
        all_samples = json.load(f)

    total_count = len(all_samples)
    excluded_not_2mod = 0
    kept_records = []

    # Track per-document question index for question_id generation
    doc_question_index = {}

    for sample in all_samples:
        doc_id = sample.get('doc_id', '').replace('\n', '').replace('\r', '').strip()

        # Track position within document (0-based across ALL questions, not just kept ones)
        if doc_id not in doc_question_index:
            doc_question_index[doc_id] = 0
        else:
            doc_question_index[doc_id] += 1
        question_position = doc_question_index[doc_id]

        # Parse evidence_sources (stored as string repr of list in MMLongBench)
        evidence_sources = sample.get('evidence_sources', '[]')
        if isinstance(evidence_sources, str):
            try:
                evidence_sources = eval(evidence_sources)
            except Exception:
                evidence_sources = []

        # Map to internal modality types
        modality_types = map_to_modality_types(evidence_sources)

        # Keep only questions with exactly 2 modalities
        if len(modality_types) != 2:
            excluded_not_2mod += 1
            continue

        # Add question_id: {doc_id}_{position}
        sample['question_id'] = f"{doc_id}_{question_position}"
        kept_records.append(sample)

    # Write output
    with open(output_path, 'w') as f:
        json.dump(kept_records, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print("FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Total records read:          {total_count}")
    print(f"Excluded (not 2 modalities): {excluded_not_2mod}")
    print(f"Kept (exactly 2 modalities): {len(kept_records)}")
    print(f"\nOutput written to: {output_path}")

    # Distribution
    print(f"\n{'='*60}")
    print("EVIDENCE SOURCES DISTRIBUTION (2-modality questions)")
    print(f"{'='*60}")
    combos = Counter()
    for r in kept_records:
        es = r.get('evidence_sources', '[]')
        if isinstance(es, str):
            es = eval(es)
        combos[tuple(sorted(es))] += 1
    for combo, count in combos.most_common():
        print(f"  {count:4d}: {list(combo)}")


if __name__ == "__main__":
    main()
