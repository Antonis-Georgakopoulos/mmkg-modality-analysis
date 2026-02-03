"""
Modality Contribution Analysis Pipeline (WITH IMAGES)

This module provides functionality for evaluating how different modality subsets
contribute to question answering performance in multimodal documents.

This version includes actual images in the prompt for vision-capable models.

1. For each question with gold evidence_sources (e.g., ['Chart', 'Table']):
   - Test all subsets: ['Chart'], ['Table'], ['Chart', 'Table']
   - Measure accuracy for each subset
   - Track which subset sizes reach accuracy threshold
2. Aggregate across questions to find modality minimalism patterns
3. Answer: "Can we answer with just 1 modality? When do we need 2? 3?"
"""

from .config import MODELS_TO_EVALUATE, MODALITY_MAPPING
from .data_loader import load_samples
from .api import call_model, call_model_with_images
from .modality_utils import (
    generate_modality_subsets,
    check_question_answered,
    check_all_questions_answered_for_document,
)
from .retrieval import answer_with_modality_subset
from .pipeline import process_document_rq3, main_async

__all__ = [
    "MODELS_TO_EVALUATE",
    "MODALITY_MAPPING",
    "load_samples",
    "call_model",
    "call_model_with_images",
    "generate_modality_subsets",
    "check_question_answered",
    "check_all_questions_answered_for_document",
    "answer_with_modality_subset",
    "process_document_rq3",
    "main_async",
]
