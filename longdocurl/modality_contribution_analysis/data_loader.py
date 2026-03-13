"""
Data loading utilities for modality contribution analysis - LongDocURL benchmark.

Loads JSONL format and groups questions by doc_no.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from lightrag.utils import logger

from .config import MODALITY_MAPPING, GROUPED_MODALITIES

# Reverse mapping: internal modality -> group name
INTERNAL_TO_GROUP = {}
for group_name, internal_mods in GROUPED_MODALITIES.items():
    for mod in internal_mods:
        INTERNAL_TO_GROUP[mod] = group_name


def load_samples_jsonl(samples_path: str) -> Dict[str, List[Dict]]:
    """
    Load LongDocURL JSONL samples and group by doc_no.
    
    Each line in the JSONL is a question with fields:
    - question_id: unique identifier
    - doc_no: document number (used for grouping)
    - question: the question text
    - answer: ground truth answer
    - answer_format: format type (String, List, Integer, etc.)
    - evidence_sources: list of modality sources (e.g., ["Text", "Figure"])
    - evidence_pages: list of page numbers
    - etc.
    
    Returns:
        Dict mapping doc_no -> list of question dicts
    """
    with open(samples_path, 'r') as f:
        lines = f.readlines()
    
    doc_questions = defaultdict(list)
    
    for line in lines:
        if not line.strip():
            continue
            
        sample = json.loads(line.strip())
        
        # Get doc_no (used for grouping)
        doc_no = sample.get('doc_no', '').strip()
        if not doc_no:
            logger.warning(f"Skipping sample without doc_no: {sample.get('question_id', 'unknown')}")
            continue
        
        # Evidence sources are already a list in LongDocURL
        evidence_sources = sample.get('evidence_sources', [])
        if isinstance(evidence_sources, str):
            # Handle case where it might be a string representation
            try:
                evidence_sources = eval(evidence_sources)
            except (SyntaxError, ValueError, NameError):
                evidence_sources = [evidence_sources]
        
        # Map to modality types
        modality_types = []
        for source in evidence_sources:
            source = source.strip()
            if source in MODALITY_MAPPING:
                internal_mods = MODALITY_MAPPING[source]
                # Check if this maps to a grouped modality
                if internal_mods and internal_mods[0] in INTERNAL_TO_GROUP:
                    # Use the group name instead of individual modalities
                    group_name = INTERNAL_TO_GROUP[internal_mods[0]]
                    if group_name not in modality_types:
                        modality_types.append(group_name)
                else:
                    # Not grouped - use individual modalities
                    for mod in internal_mods:
                        if mod not in modality_types:
                            modality_types.append(mod)
            else:
                logger.warning(f"Unknown modality: {source}")
        
        # Store with both names and types
        sample['evidence_sources_list'] = evidence_sources
        sample['gold_modality_types'] = modality_types
        sample['gold_modality_names'] = evidence_sources
        
        doc_questions[doc_no].append(sample)
    
    total_questions = sum(len(q) for q in doc_questions.values())
    logger.info(f"Loaded {total_questions} questions from {len(doc_questions)} documents")
    
    # Log modality distribution
    modality_counts = defaultdict(int)
    for questions in doc_questions.values():
        for q in questions:
            for mod in q.get('gold_modality_types', []):
                modality_counts[mod] += 1
    
    logger.info("Modality distribution:")
    for mod, count in sorted(modality_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {mod}: {count}")
    
    return doc_questions


def find_pdf_path(doc_no: str, pdfs_dir: str) -> str:
    """
    Find the PDF path for a given doc_no in the LongDocURL pdfs directory.
    
    PDFs are organized as: pdfs/{prefix}/{doc_no}.pdf
    where prefix is the first 4 digits of doc_no.
    
    Args:
        doc_no: Document number (e.g., "4026369")
        pdfs_dir: Base directory for PDFs (e.g., "./longdocurl/pdfs")
        
    Returns:
        Full path to the PDF file
    """
    pdfs_path = Path(pdfs_dir)
    prefix = doc_no[:4]  # First 4 digits
    
    # Look in the prefix subdirectory
    subdir = pdfs_path / prefix
    if subdir.exists():
        # Find PDF with matching doc_no
        for pdf_file in subdir.glob(f"{doc_no}.pdf"):
            return str(pdf_file)
        
        # Also try without exact match (in case of naming variations)
        for pdf_file in subdir.glob(f"{doc_no}*.pdf"):
            return str(pdf_file)
    
    # Fallback: search all subdirectories
    for pdf_file in pdfs_path.glob(f"**/{doc_no}.pdf"):
        return str(pdf_file)
    
    logger.warning(f"PDF not found for doc_no: {doc_no}")
    return None
