#!/usr/bin/env python
"""
Create LongDocURL_public_cleaned_2modalities.jsonl from LongDocURL_public.jsonl.

Steps:
1. Clean evidence_sources (merge duplicate "Figure" entries, deduplicate)
2. Exclude questions containing "Other" or "Others" in evidence_sources
3. Map evidence_sources to internal modality types:
       Figure  -> image
       Table   -> table
       Text    -> plain_text  (grouped: text, list)
       Layout  -> layout      (grouped: header, footer, page_number, ...)
4. Keep only questions with exactly 2 unique mapped modalities

Output: LongDocURL_public_cleaned_2modalities.jsonl
"""

import json
from pathlib import Path
from collections import Counter

# ── Modality mapping (mirrors config.py) ─────────────────────────────────────

MODALITY_MAPPING = {
    'Figure': ['image'],
    'Table': ['table'],
    'Text': ['text', 'list'],
    'Layout': ['header', 'footer', 'page_number', 'page_footnote', 'aside_text'],
}

GROUPED_MODALITIES = {
    'layout': ['header', 'footer', 'page_number', 'page_footnote', 'aside_text'],
    'plain_text': ['text', 'list'],
}

# Reverse mapping: internal modality -> group name
INTERNAL_TO_GROUP = {}
for group_name, internal_mods in GROUPED_MODALITIES.items():
    for mod in internal_mods:
        INTERNAL_TO_GROUP[mod] = group_name


# ── Cleaning functions ────────────────────────────────────────────────────────

def clean_evidence_sources(evidence_sources: list) -> list:
    """
    Clean evidence_sources:
    - Deduplicate entries (e.g., "Figure + Figure" -> single "Figure")
    - Keep unique values preserving order
    """
    seen = set()
    cleaned = []
    for source in evidence_sources:
        source = source.strip()
        if source not in seen:
            seen.add(source)
            cleaned.append(source)
    return cleaned


def has_other_modality(evidence_sources: list) -> bool:
    """Check if evidence_sources contains 'Other' or 'Others'."""
    for source in evidence_sources:
        if source.lower().strip() in ("other", "others"):
            return True
    return False


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
    input_path = Path(__file__).parent / "LongDocURL_public.jsonl"
    output_path = Path(__file__).parent / "LongDocURL_public_cleaned_2modalities.jsonl"

    print(f"Reading from: {input_path}")

    total_count = 0
    excluded_other = 0
    excluded_not_2mod = 0
    kept_records = []

    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            total_count += 1
            record = json.loads(line.strip())
            evidence_sources = record.get("evidence_sources", [])

            # Step 1: Exclude "Other"/"Others"
            if has_other_modality(evidence_sources):
                excluded_other += 1
                continue

            # Step 2: Clean (deduplicate)
            cleaned_sources = clean_evidence_sources(evidence_sources)
            record["evidence_sources"] = cleaned_sources

            # Step 3: Map to internal modality types
            modality_types = map_to_modality_types(cleaned_sources)

            # Step 4: Keep only questions with exactly 2 modalities
            if len(modality_types) != 2:
                excluded_not_2mod += 1
                continue

            kept_records.append(record)

    # Write output
    with open(output_path, 'w') as f:
        for record in kept_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Summary
    print(f"\n{'='*60}")
    print("FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Total records read:          {total_count}")
    print(f"Excluded (Other/Others):     {excluded_other}")
    print(f"Excluded (not 2 modalities): {excluded_not_2mod}")
    print(f"Kept (exactly 2 modalities): {len(kept_records)}")
    print(f"\nOutput written to: {output_path}")

    # Distribution
    print(f"\n{'='*60}")
    print("EVIDENCE SOURCES DISTRIBUTION (2-modality questions)")
    print(f"{'='*60}")
    combos = Counter(tuple(sorted(r['evidence_sources'])) for r in kept_records)
    for combo, count in combos.most_common():
        print(f"  {count:4d}: {list(combo)}")


if __name__ == "__main__":
    main()
