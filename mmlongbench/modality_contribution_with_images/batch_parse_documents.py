#!/usr/bin/env python
"""
Batch Document Parsing for Modality Contribution Pipeline (TRUE PARALLELISM)

This script pre-parses all documents in parallel using MinerU with ThreadPoolExecutor.
This provides TRUE parallelism for the MinerU parsing step.

Run this BEFORE running the evaluation pipeline to speed up processing.
The pipeline will then skip MinerU parsing and only do KG construction.

Usage:
    python mmlongbench/modality_contribution_with_images/batch_parse_documents.py \
        --samples mmlongbench/data/samples.json \
        --documents mmlongbench/data/documents \
        --output processed_documents \
        --workers 2
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from raganything.parser import MineruParser
from lightrag.utils import logger


def get_document_ids_from_samples(samples_path: str) -> Set[str]:
    """Extract unique document IDs from samples file."""
    with open(samples_path, 'r') as f:
        samples = json.load(f)
    
    doc_ids = set()
    for sample in samples:
        doc_id = sample.get('doc_id', '').replace('\n', '').replace('\r', '').strip()
        if doc_id:
            doc_ids.add(doc_id)
    
    return doc_ids


def check_mineru_output_exists(doc_id: str, output_dir: str) -> bool:
    """
    Check if MinerU output already exists for a document.
    Only checks for content_list.json, NOT rag_storage (that's done by pipeline).
    """
    doc_name = doc_id.replace('.pdf', '')
    doc_output_dir = os.path.join(output_dir, doc_name, 'output')
    
    if not os.path.exists(doc_output_dir):
        return False
    
    # Check for content_list.json (key MinerU output file)
    for root, dirs, files in os.walk(doc_output_dir):
        for f in files:
            if f.endswith('_content_list.json'):
                return True
    
    return False


def parse_single_document(
    doc_id: str,
    documents_dir: str,
    output_dir: str,
    device: str,
    backend: str,
) -> Tuple[bool, str, Optional[str]]:
    """
    Parse a single document with MinerU (synchronous, for ThreadPoolExecutor).
    
    This ONLY does MinerU parsing, NOT KG construction.
    The pipeline will do KG construction later.
    
    Returns:
        Tuple of (success, doc_id, error_message)
    """
    doc_name = doc_id.replace('.pdf', '')
    doc_base_dir = os.path.join(output_dir, doc_name)
    doc_output_dir = os.path.join(doc_base_dir, 'output')
    doc_path = os.path.join(documents_dir, doc_id)
    
    if not os.path.exists(doc_path):
        return False, doc_id, f"Document not found: {doc_path}"
    
    # Create output directory
    os.makedirs(doc_output_dir, exist_ok=True)
    
    try:
        # Use MineruParser directly (synchronous)
        parser = MineruParser()
        
        logger.info(f"Parsing {doc_id} with MinerU...")
        
        # Parse the PDF - this is the CPU/GPU-bound work
        content_list = parser.parse_pdf(
            pdf_path=doc_path,
            output_dir=doc_output_dir,
            method="auto",
            device=device,
            backend=backend,
        )
        
        logger.info(f"✅ Parsed {doc_id}: {len(content_list)} content blocks")
        return True, doc_id, None
        
    except Exception as e:
        logger.error(f"❌ Failed to parse {doc_id}: {e}")
        return False, doc_id, str(e)


def batch_parse_documents(
    samples_path: str,
    documents_dir: str,
    output_dir: str,
    max_workers: int = 2,
    device: str = "cuda",
    backend: str = "hybrid-auto-engine",
    limit: int = None,
    skip_existing: bool = True,
) -> Dict:
    """
    Parse all documents from samples file with TRUE PARALLELISM using ThreadPoolExecutor.
    
    This ONLY does MinerU parsing, NOT KG construction.
    Run the evaluation pipeline afterwards to do KG construction.
    
    Args:
        samples_path: Path to samples.json file
        documents_dir: Directory containing PDF documents
        output_dir: Output directory for processed documents
        max_workers: Number of parallel workers (default: 2)
        device: Device for MinerU ("cuda", "mps", or "cpu")
        backend: MinerU backend ("pipeline", "vlm", or "hybrid-auto-engine")
        limit: Limit number of documents to process (for testing)
        skip_existing: Skip documents that already have MinerU output
    
    Returns:
        Dict with processing statistics
    """
    start_time = time.time()
    
    # Get all document IDs from samples
    doc_ids = get_document_ids_from_samples(samples_path)
    logger.info(f"Found {len(doc_ids)} unique documents in samples")
    
    # Apply limit if specified
    if limit:
        doc_ids = set(list(doc_ids)[:limit])
        logger.info(f"Limited to {limit} documents for testing")
    
    # Filter documents
    docs_to_process = []
    already_parsed = []
    not_found = []
    
    for doc_id in doc_ids:
        doc_path = os.path.join(documents_dir, doc_id)
        
        if not os.path.exists(doc_path):
            not_found.append(doc_id)
            logger.warning(f"PDF not found, skipping: {doc_path}")
            continue
        
        if skip_existing and check_mineru_output_exists(doc_id, output_dir):
            already_parsed.append(doc_id)
        else:
            docs_to_process.append(doc_id)
    
    logger.info(f"Already parsed (skipped): {len(already_parsed)} documents")
    logger.info(f"Not found: {len(not_found)} documents")
    logger.info(f"To parse: {len(docs_to_process)} documents")
    
    if not docs_to_process:
        logger.info("All documents already parsed!")
        return {
            "total_documents": len(doc_ids),
            "already_parsed": len(already_parsed),
            "newly_parsed": 0,
            "failed": 0,
            "not_found": len(not_found),
            "processing_time": 0,
        }
    
    # Process documents with TRUE PARALLELISM using ThreadPoolExecutor
    successful = []
    failed = []
    errors = {}
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting TRUE PARALLEL parsing with {max_workers} workers")
    logger.info(f"Device: {device}, Backend: {backend}")
    logger.info(f"{'='*60}\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_doc = {
            executor.submit(
                parse_single_document,
                doc_id,
                documents_dir,
                output_dir,
                device,
                backend,
            ): doc_id
            for doc_id in docs_to_process
        }
        
        # Process with progress bar
        with tqdm(total=len(docs_to_process), desc="Parsing documents (parallel)", unit="doc") as pbar:
            for future in as_completed(future_to_doc):
                doc_id = future_to_doc[future]
                try:
                    success, doc_id, error_msg = future.result()
                    if success:
                        successful.append(doc_id)
                    else:
                        failed.append(doc_id)
                        errors[doc_id] = error_msg
                except Exception as e:
                    failed.append(doc_id)
                    errors[doc_id] = str(e)
                    logger.error(f"Exception for {doc_id}: {e}")
                
                pbar.update(1)
                pbar.set_postfix({"ok": len(successful), "fail": len(failed)})
    
    processing_time = time.time() - start_time
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH MINERU PARSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total documents: {len(doc_ids)}")
    logger.info(f"Already parsed (skipped): {len(already_parsed)}")
    logger.info(f"Newly parsed: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info("\n⚠️  NOTE: This only did MinerU parsing.")
    logger.info("   Run the evaluation pipeline to do KG construction.")
    
    if failed:
        logger.warning("\nFailed documents:")
        for doc_id in failed:
            logger.warning(f"  - {doc_id}: {errors.get(doc_id, 'Unknown error')}")
    
    return {
        "total_documents": len(doc_ids),
        "already_parsed": len(already_parsed),
        "newly_parsed": len(successful),
        "failed": len(failed),
        "not_found": len(not_found),
        "failed_docs": failed,
        "errors": errors,
        "processing_time": processing_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch parse documents with MinerU (TRUE PARALLELISM)"
    )
    parser.add_argument(
        "--samples", "-s",
        default="mmlongbench/data/samples.json",
        help="Path to samples.json file"
    )
    parser.add_argument(
        "--documents", "-d",
        default="mmlongbench/data/documents",
        help="Directory containing PDF documents"
    )
    parser.add_argument(
        "--output", "-o",
        default="processed_documents",
        help="Output directory for processed documents"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=10,
        help="Number of parallel workers (default: 2)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for MinerU (cuda, mps, or cpu)"
    )
    parser.add_argument(
        "--backend",
        default="hybrid-auto-engine",
        help="MinerU backend (pipeline, vlm, or hybrid-auto-engine)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents (for testing)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip documents with existing MinerU output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Validate inputs
    if not os.path.exists(args.samples):
        logger.error(f"Samples file not found: {args.samples}")
        return 1
    
    if not os.path.exists(args.documents):
        logger.error(f"Documents directory not found: {args.documents}")
        return 1
    
    # Run batch parsing (synchronous with ThreadPoolExecutor)
    result = batch_parse_documents(
        samples_path=args.samples,
        documents_dir=args.documents,
        output_dir=args.output,
        max_workers=args.workers,
        device=args.device,
        backend=args.backend,
        limit=args.limit,
        skip_existing=not args.no_skip,
    )
    
    # Return exit code
    if result["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
