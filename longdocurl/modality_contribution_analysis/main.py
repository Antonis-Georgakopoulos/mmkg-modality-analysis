#!/usr/bin/env python
"""
CLI entry point for modality contribution analysis - LongDocURL benchmark.

Usage:
    python longdocurl/modality_contribution/main.py --api-key YOUR_KEY
    
Or as module:
    python -m longdocurl.modality_contribution --api-key YOUR_KEY
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add project root to path for direct script execution
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ["ORT_LOG_LEVEL"] = "3"  # Suppress ONNX Runtime warnings/errors

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

from longdocurl.modality_contribution.pipeline import main_async


def main():
    parser = argparse.ArgumentParser(
        description="RQ3: Modality Minimalism Evaluation - LongDocURL Benchmark (WITH IMAGES)"
    )
    parser.add_argument(
        "--samples",
        default="./longdocurl/LongDocURL_public_cleaned.jsonl",
        help="Path to cleaned LongDocURL JSONL file"
    )
    parser.add_argument(
        "--documents",
        default="./longdocurl/pdfs",
        help="Directory containing PDF documents (organized as pdfs/{prefix}/{doc_no}.pdf)"
    )
    parser.add_argument(
        "--processed-docs-dir",
        default="./processed_documents_longdocurl",
        help="Base directory for processed documents (each doc will have output/ and rag_storage/ subdirs)"
    )
    parser.add_argument(
        "--results-dir",
        default="./results_longdocurl",
        help="Directory for evaluation results (separate file per model)"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY"),
        help="OpenAI API key"
    )
    parser.add_argument(
        "--base-url",
        # default="https://llmproxy.uva.nl",
        default="https://api.openai.com/v1",
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
        default="cuda",
        choices=["cpu", "cuda", "cuda:0", "cuda:1", "mps"],
        help="Device for MinerU parsing (default: mps for Mac)"
    )
    parser.add_argument(
        "--backend",
        default="hybrid-auto-engine",
        choices=["pipeline", "vlm-auto-engine", "hybrid-auto-engine"],
        help="MinerU backend. 'hybrid-auto-engine' (default) combines VLM accuracy with pipeline speed."
    )
    parser.add_argument(
        "--max-async",
        type=int,
        default=8,
        help="Max concurrent LLM calls (default: 8)"
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
    parser.add_argument(
        "--max-parallel-insert",
        type=int,
        default=8,
        help="Max concurrent multimodal processing (image descriptions). Default: 8"
    )
    parser.add_argument(
        "--doc-start",
        type=int,
        default=None,
        help="Start document index (1-based, inclusive). Processing goes BACKWARDS from doc-start to doc-end."
    )
    parser.add_argument(
        "--doc-end",
        type=int,
        default=None,
        help="End document index (1-based, inclusive). Processing goes BACKWARDS from doc-start to doc-end."
    )
    
    args = parser.parse_args()
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("Modality Contribution Analysis - LongDocURL Benchmark (WITH IMAGES)")
    print("="*80 + "\n")
    
    main()
