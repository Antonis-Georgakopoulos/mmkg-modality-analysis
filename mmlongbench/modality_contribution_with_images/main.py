#!/usr/bin/env python
"""
CLI entry point for modality contribution analysis (WITH IMAGES).

Usage:
    python mmlongbench/modality_contribution_with_images/main.py --api-key YOUR_KEY
    
Or as module:
    python -m mmlongbench.modality_contribution_with_images --api-key YOUR_KEY
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add project root to path for direct script execution
# This allows both `python main.py` and `python -m mmlongbench.modality_contribution_with_images`
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ["ORT_LOG_LEVEL"] = "3"  # Suppress ONNX Runtime warnings/errors

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

from mmlongbench.modality_contribution_with_images.pipeline import main_async


def main():
    parser = argparse.ArgumentParser(
        description="RQ3: Modality Minimalism Evaluation (WITH IMAGES)"
    )
    parser.add_argument(
        "--samples",
        default="./mmlongbench/data/samples.json",
        help="Path to samples.json"
    )
    parser.add_argument(
        "--documents",
        default="./mmlongbench/data/documents",
        help="Directory containing PDF documents"
    )
    parser.add_argument(
        "--processed-docs-dir",
        default="./processed_documents",
        help="Base directory for processed documents (each doc will have output/ and rag_storage/ subdirs)"
    )
    parser.add_argument(
        "--results-dir",
        default="./results_vlm",
        help="Directory for evaluation results (separate file per model)"
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API"
    )
    parser.add_argument(
        "--parser",
        default="mineru",
        choices=["mineru", "docling"],
        help="Parser to use"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file"
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["cpu", "cuda", "cuda:0", "cuda:1", "mps"],
        help="Device for MinerU parsing (default: cuda)"
    )
    parser.add_argument(
        "--backend",
        default="hybrid-auto-engine",
        choices=["pipeline", "vlm-auto-engine", "hybrid-auto-engine"],
        help="MinerU backend. 'hybrid-auto-engine' (default) combines VLM accuracy with pipeline speed. Use 'pipeline' for faster CPU-only processing."
    )
    parser.add_argument(
        "--max-async",
        type=int,
        default=8,
        help="Max concurrent LLM calls (default: 8, increase based on API rate limits)"
    )
    parser.add_argument(
        "--embedding-max-async",
        type=int,
        default=16,
        help="Max concurrent embedding calls (default: 16)"
    )
    parser.add_argument(
        "--embedding-batch-num",
        type=int,
        default=32,
        help="Batch size for embeddings (default: 32)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("Modality Contribution Analysis (WITH IMAGES)")
    print("="*80 + "\n")
    
    main()
