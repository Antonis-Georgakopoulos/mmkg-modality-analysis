"""
Modality contribution analysis for LongDocURL benchmark (WITH IMAGES).

Adapted from mmlongbench modality_contribution_with_images_prod_code_going_backwards.
"""

from .pipeline import main_async, process_document_rq3
from .data_loader import load_samples_jsonl
from .config import MODELS_TO_EVALUATE, MODALITY_MAPPING

__all__ = [
    "main_async",
    "process_document_rq3",
    "load_samples_jsonl",
    "MODELS_TO_EVALUATE",
    "MODALITY_MAPPING",
]
