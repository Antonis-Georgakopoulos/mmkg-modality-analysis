"""
Data loading utilities for modality contribution analysis.
"""

import json
from collections import defaultdict
from typing import Dict, List

from lightrag.utils import logger

from .config import MODALITY_MAPPING, GROUPED_MODALITIES

# Reverse mapping: internal modality -> group name
INTERNAL_TO_GROUP = {}
for group_name, internal_mods in GROUPED_MODALITIES.items():
    for mod in internal_mods:
        INTERNAL_TO_GROUP[mod] = group_name


def load_samples(samples_path: str) -> Dict[str, List[Dict]]:
    """Load samples and group by document (including 'Not answerable' questions)"""
    with open(samples_path, 'r') as f:
        all_samples = json.load(f)
    
    doc_questions = defaultdict(list)
    
    for sample in all_samples:
        # Normalize doc_id by removing newlines and extra whitespace
        doc_id = sample.get('doc_id', '').replace('\n', '').replace('\r', '').strip()
        sample['doc_id'] = doc_id
        
        # Parse evidence_sources
        evidence_sources = eval(sample.get('evidence_sources', '[]'))
        
        # Map to modality types
        # For grouped modalities (layout, plain_text), keep the group name instead of expanding
        modality_types = []
        for source in evidence_sources:
            if source in MODALITY_MAPPING:
                internal_mods = MODALITY_MAPPING[source]
                # Check if this maps to a grouped modality
                if internal_mods and internal_mods[0] in INTERNAL_TO_GROUP:
                    # Use the group name instead of individual modalities
                    group_name = INTERNAL_TO_GROUP[internal_mods[0]]
                    modality_types.append(group_name)
                else:
                    # Not grouped - use individual modalities
                    modality_types.extend(internal_mods)
            else:
                logger.warning(f"Unknown modality: {source}")
        
        # For questions without modality types (e.g., "Not answerable"), use empty list
        if not modality_types:
            modality_types = []
            evidence_sources = []
        
        # Store with both names and types
        sample['evidence_sources_list'] = evidence_sources
        sample['gold_modality_types'] = list(set(modality_types))  # unique types
        sample['gold_modality_names'] = evidence_sources
        doc_questions[doc_id].append(sample)
    
    total_questions = sum(len(q) for q in doc_questions.values())
    not_answerable_count = sum(1 for samples in doc_questions.values() for s in samples if s.get('answer') == 'Not answerable')
    logger.info(f"Loaded {total_questions} questions from {len(doc_questions)} documents")
    logger.info(f"Including {not_answerable_count} 'Not answerable' questions")
    
    return doc_questions
