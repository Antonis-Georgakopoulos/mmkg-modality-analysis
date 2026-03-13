# LongDocURL Modality Contribution Analysis

Adapted from `mmlongbench/modality_contribution_with_images_prod_code_going_backwards` for the LongDocURL benchmark.

## Overview

This module evaluates the contribution of different modalities (Text, Figure, Table, Layout) to question answering performance on the LongDocURL benchmark.

## Usage

```bash
# Run from project root
python longdocurl/modality_contribution/main.py --api-key YOUR_KEY

# With options
python longdocurl/modality_contribution/main.py \
    --samples ./longdocurl/LongDocURL_public_cleaned.jsonl \
    --documents ./longdocurl/pdfs \
    --processed-docs-dir ./processed_documents_longdocurl \
    --results-dir ./results_longdocurl \
    --limit 5
```

## Key Differences from MMLongBench

1. **Data Format**: Uses JSONL format (one JSON per line) instead of JSON
2. **Document ID**: Uses `doc_no` field instead of `doc_id`
3. **PDF Organization**: PDFs are organized as `pdfs/{prefix}/{doc_no}.pdf` where prefix is first 4 digits
4. **Evidence Sources**: LongDocURL uses `Text`, `Table`, `Figure`, `Layout` (vs MMLongBench's `Pure-text`, `Chart`, etc.)

## Data Cleaning

The cleaned JSONL file (`LongDocURL_public_cleaned.jsonl`) was created by:
1. Merging "Figure + Figure" → "Figure" in `evidence_sources`
2. Excluding questions with "Other" or "Others" in `evidence_sources`

Run `clean_longdocurl_data.py` to regenerate:
```bash
python longdocurl/clean_longdocurl_data.py
```

## Pipeline

For each document:
1. Parse PDF with MinerU (or load existing parsed data)
2. Build Knowledge Graph with RAGAnything
3. For each question:
   - Generate modality subsets from gold modalities
   - Answer with each modality subset
   - Evaluate score using ANLS metric

## Configuration

Edit `config.py` to:
- Change models to evaluate (`MODELS_TO_EVALUATE`)
- Modify modality mappings (`MODALITY_MAPPING`)
