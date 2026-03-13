"""
Modality utility functions for subset generation and checkpoint checking.
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
    
    Example: ['text', 'image', 'table'] -> [
        (1, {'text'}, 'normal'),
        (1, {'image'}, 'normal'),
        (1, {'table'}, 'normal'),
        (2, {'text', 'image'}, 'normal'),
        (2, {'text', 'table'}, 'normal'),
        (2, {'image', 'table'}, 'normal'),
        (3, {'text', 'image', 'table'}, 'normal')
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
        question_id: Question identifier (e.g., "doc123_1")
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
    doc_id: str, 
    questions: List[Dict], 
    results_by_model: Dict[str, List]
) -> bool:
    """
    Check if all question IDs for a document exist in the results for at least one model.
    This is a simpler check that just verifies if the questions have been processed,
    without checking specific modality combinations.
    
    Args:
        doc_id: Document identifier
        questions: List of questions for this document
        results_by_model: Dictionary of results grouped by model
    
    Returns:
        True if all question IDs exist in results for any model, False otherwise
    """
    # Create set of all question IDs for this document
    question_ids = {f"{doc_id}_{i}" for i in range(1, len(questions) + 1)}
    
    logger.info(f"Checking skip for doc: {doc_id[:60]}...")
    logger.info(f"Expected {len(question_ids)} question IDs: {sorted(list(question_ids))[:2]}...")
    
    # Check if all question IDs exist in ANY of the model results
    for model_name in MODELS_TO_EVALUATE:
        if model_name not in results_by_model:
            continue
        
        # Get all question IDs present in this model's results for THIS document
        existing_question_ids = {
            result.get('question_id') 
            for result in results_by_model[model_name]
            if result.get('doc_id') == doc_id
        }
        
        logger.info(f"Model {model_name}: found {len(existing_question_ids)} existing IDs")
        if existing_question_ids and len(existing_question_ids) < 3:
            logger.info(f"      Existing IDs: {sorted(list(existing_question_ids))}")
        
        # If all question IDs exist for this model, we can skip the document
        if question_ids.issubset(existing_question_ids):
            logger.info(f"   All {len(question_ids)} questions found for {model_name} - SKIPPING!")
            return True
    
    logger.info("   Document needs processing - not all questions found")
    return False
