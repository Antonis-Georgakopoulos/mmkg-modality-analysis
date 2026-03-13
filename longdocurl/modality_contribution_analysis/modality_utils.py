"""
Modality utility functions for subset generation and checkpoint checking.
Adapted for LongDocURL benchmark.
"""

from itertools import combinations
from typing import Dict, List

from lightrag.utils import logger

from .config import MODELS_TO_EVALUATE


def generate_modality_subsets(gold_modalities: List[str]) -> List[tuple]:
    """
    Generate all non-empty subsets of gold modalities.
    Returns list of tuples: (subset_size, frozenset_of_modalities, experiment_type)
    
    Example: ['image', 'table'] -> [
        (1, {'image'}, 'normal'),
        (1, {'table'}, 'normal'),
        (2, {'image', 'table'}, 'normal')
    ]
    """
    unique_modalities = set(gold_modalities)
    subsets = []
    
    # Start from 1 to skip the empty set
    for size in range(1, len(unique_modalities) + 1):
        for subset in combinations(unique_modalities, size):
            subsets.append((size, frozenset(subset), 'normal'))
    
    return sorted(subsets, key=lambda x: x[0])  # Sort by size


def check_question_answered(question_id: str, model_name: str, subset_modalities: tuple, results_by_model: Dict[str, List]) -> bool:
    """
    Check if a specific question with specific modality subset has been answered for a given model.
    
    Args:
        question_id: Question identifier
        model_name: Model name (e.g., "gpt-4o-mini")
        subset_modalities: Tuple of modalities (e.g., ('image', 'table'))
        results_by_model: Dictionary of results grouped by model
    
    Returns:
        True if question is already answered, False otherwise
    """
    if model_name not in results_by_model:
        return False
    
    # Normalize subset_modalities to tuple for comparison
    subset_modalities_normalized = tuple(sorted(subset_modalities))
    
    for result in results_by_model[model_name]:
        # Get existing subset_modalities (could be list or tuple from JSON)
        existing_subset = result.get('subset_modalities')
        if existing_subset is None:
            continue
        
        # Normalize existing subset to tuple for comparison (handle both list and tuple)
        existing_subset_normalized = tuple(sorted(existing_subset))
        
        if (result.get('question_id') == question_id and 
            existing_subset_normalized == subset_modalities_normalized and
            result.get('model') == model_name):
            return True
    
    return False


def check_all_questions_answered_for_document(
    doc_no: str, 
    questions: List[Dict], 
    results_by_model: Dict[str, List]
) -> bool:
    """
    Check if all question IDs for a document exist in the results for at least one model.
    
    Since question_ids now include subset indices (e.g., base_id_1, base_id_2, base_id_3),
    we need to generate all expected question_ids for each question based on its modality subsets.
    
    Args:
        doc_no: Document number
        questions: List of questions for this document
        results_by_model: Dictionary of results grouped by model
    
    Returns:
        True if all question IDs exist in results for any model, False otherwise
    """
    # Generate all expected question_ids (base_id + subset_index)
    expected_question_ids = set()
    for q in questions:
        base_id = q['question_id']
        gold_modalities = q.get('gold_modality_types', [])
        
        # Generate subsets to count how many question_ids we expect
        subsets = generate_modality_subsets(gold_modalities)
        if not subsets:
            # Fallback: one subset with all modalities
            subsets = [(3, frozenset(['text', 'image', 'table']), 'normal')]
        
        # Each subset gets a unique question_id: base_id_1, base_id_2, etc.
        for subset_idx in range(1, len(subsets) + 1):
            expected_question_ids.add(f"{base_id}_{subset_idx}")
    
    logger.info(f"🔍 Checking skip for doc: {doc_no}...")
    logger.info(f"   Expected {len(expected_question_ids)} question IDs (including subset variants)")
    
    # Check if all question IDs exist for ALL models
    all_models_complete = True
    for model_name in MODELS_TO_EVALUATE:
        if model_name not in results_by_model:
            logger.info(f"   Model {model_name}: no results yet")
            all_models_complete = False
            continue
        
        # Get all question IDs present in this model's results for THIS document
        existing_question_ids = {
            result.get('question_id') 
            for result in results_by_model[model_name]
            if result.get('doc_no') == doc_no
        }
        
        logger.info(f"   Model {model_name}: found {len(existing_question_ids)} existing IDs")
        
        if not expected_question_ids.issubset(existing_question_ids):
            all_models_complete = False
    
    if all_models_complete:
        logger.info(f"   ✅ All {len(expected_question_ids)} questions found for ALL models - SKIPPING!")
        return True
    
    logger.info("   ⚠️  Document needs processing - not all models have all questions answered")
    return False
