#!/usr/bin/env python
"""
Script to create a cleaned version of LongDocURL_public.jsonl:
1. Merge "Figure + Figure" -> "Figure" in evidence_sources
2. Exclude all questions containing "Other" or "Others" in evidence_sources
"""

import json
from pathlib import Path


def clean_evidence_sources(evidence_sources: list) -> list:
    """
    Clean evidence_sources:
    - Merge "Figure + Figure" (or multiple Figures) -> single "Figure"
    - Keep unique values
    """
    cleaned = []
    for source in evidence_sources:
        # Normalize: strip whitespace
        source = source.strip()
        
        # Skip if already added (deduplication)
        if source not in cleaned:
            cleaned.append(source)
    
    # If we have multiple "Figure" entries, keep only one
    figure_count = cleaned.count("Figure")
    if figure_count > 1:
        # Remove all Figures and add just one
        cleaned = [s for s in cleaned if s != "Figure"]
        cleaned.append("Figure")
    
    return cleaned


def has_other_modality(evidence_sources: list) -> bool:
    """Check if evidence_sources contains 'Other' or 'Others'."""
    for source in evidence_sources:
        source_lower = source.lower().strip()
        if source_lower in ("other", "others"):
            return True
    return False


def main():
    input_path = Path(__file__).parent / "LongDocURL_public.jsonl"
    output_path = Path(__file__).parent / "LongDocURL_public_cleaned.jsonl"
    
    print(f"Reading from: {input_path}")
    
    total_count = 0
    excluded_count = 0
    modified_count = 0
    cleaned_records = []
    
    # Track evidence_sources statistics
    original_sources_stats = {}
    cleaned_sources_stats = {}
    
    with open(input_path, 'r') as f:
        for line in f:
            total_count += 1
            record = json.loads(line.strip())
            evidence_sources = record.get("evidence_sources", [])
            
            # Track original stats
            key = tuple(sorted(evidence_sources))
            original_sources_stats[key] = original_sources_stats.get(key, 0) + 1
            
            # Skip records with "Other" or "Others"
            if has_other_modality(evidence_sources):
                excluded_count += 1
                continue
            
            # Clean evidence_sources
            original_sources = evidence_sources.copy()
            cleaned_sources = clean_evidence_sources(evidence_sources)
            
            if cleaned_sources != original_sources:
                modified_count += 1
                record["evidence_sources"] = cleaned_sources
            
            # Track cleaned stats
            key = tuple(sorted(cleaned_sources))
            cleaned_sources_stats[key] = cleaned_sources_stats.get(key, 0) + 1
            
            cleaned_records.append(record)
    
    # Write cleaned records
    with open(output_path, 'w') as f:
        for record in cleaned_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*60}")
    print("CLEANING SUMMARY")
    print(f"{'='*60}")
    print(f"Total records:     {total_count}")
    print(f"Excluded (Other):  {excluded_count}")
    print(f"Modified:          {modified_count}")
    print(f"Final records:     {len(cleaned_records)}")
    print(f"\nOutput written to: {output_path}")
    
    print(f"\n{'='*60}")
    print("EVIDENCE SOURCES DISTRIBUTION (CLEANED)")
    print(f"{'='*60}")
    for key, count in sorted(cleaned_sources_stats.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}: {list(key)}")


if __name__ == "__main__":
    main()
