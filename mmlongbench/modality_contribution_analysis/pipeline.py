"""
Main pipeline for document processing and modality contribution analysis.
"""

import os
import gc
import json
import asyncio
from typing import Dict, List

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from openai import RateLimitError
from lightrag.utils import EmbeddingFunc, logger
from lightrag.kg.shared_storage import finalize_share_data

from raganything import RAGAnything, RAGAnythingConfig
from raganything.utils import separate_content, insert_text_content

from mmlongbench.eval.eval_score import eval_score

from .config import MODELS_TO_EVALUATE
from .modality_utils import (
    generate_modality_subsets,
    check_question_answered,
    check_all_questions_answered_for_document,
)
from .retrieval import answer_with_modality_subset


def find_content_list_json(output_dir: str) -> str | None:
    """
    Find the content_list.json file in the MinerU output directory.
    Returns the path to the file, or None if not found.
    """
    if not os.path.exists(output_dir):
        return None
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('_content_list.json'):
                return os.path.join(root, f)
    return None


async def process_document(doc_id: str, questions: List[Dict], args, results_by_model: Dict[str, List], results_logprobs: Dict[str, Dict]) -> List[Dict]:
    """
    Process document: test all modality subsets for each question.
    This version includes actual images in the prompt for vision models.
    
    Args:
        doc_id: Document identifier
        questions: List of questions for this document
        args: Command line arguments
        results_by_model: Dictionary of existing results grouped by model (for checkpointing)
    """
    # Check if all questions are already answered before processing document
    if check_all_questions_answered_for_document(doc_id, questions, results_by_model):
        logger.info(f"All questions for {doc_id} are already answered. Skipping document processing entirely!")
        return []
    
    # Setup directories with new structure: processed_documents/{doc_name}/[output,rag_storage]
    doc_name = doc_id.replace('.pdf', '')
    doc_base_dir = os.path.join(args.processed_docs_dir, doc_name)
    working_dir = os.path.join(doc_base_dir, 'rag_storage')
    output_dir = os.path.join(doc_base_dir, 'output')
    
    # Check document processing state
    # - fully_processed: Both MinerU output AND rag_storage exist (skip everything, just query)
    # - mineru_only: MinerU output exists but no rag_storage (skip MinerU, do KG construction)
    # - not_processed: Nothing exists (run full pipeline)
    fully_processed = False
    mineru_only = False
    
    # Check for MinerU content_list.json output file
    content_list_exists = False
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith('_content_list.json'):
                    content_list_exists = True
                    break
            if content_list_exists:
                break
    
    # Check if rag_storage has content
    rag_storage_files = os.listdir(working_dir) if os.path.exists(working_dir) else []
    
    if content_list_exists and len(rag_storage_files) > 0:
        # Both MinerU and KG exist - fully processed
        fully_processed = True
        logger.info(f"Document fully processed! Using existing data from {doc_base_dir}")
    elif content_list_exists:
        # MinerU output exists but no KG - need to do KG construction only
        mineru_only = True
        logger.info(f"MinerU output exists, will do KG construction only for {doc_name}")
    
    # Only check for PDF if document has no MinerU output yet
    doc_path = os.path.join(args.documents, doc_id)
    if not fully_processed and not mineru_only and not os.path.exists(doc_path):
        logger.error(f"Document not found and no processed data exists: {doc_path}")
        return []
    
    # Create directories if needed
    if not fully_processed:
        os.makedirs(working_dir, exist_ok=True)
    if not fully_processed and not mineru_only:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load extraction prompt
    extraction_prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'eval',
        'prompt_for_answer_extraction.md'
    )
    with open(extraction_prompt_path, 'r') as f:
        extraction_prompt = f.read()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing document: {doc_id}")
    logger.info(f"Number of questions for this document: {len(questions)}")
    logger.info(f"{'='*80}")
    
    try:
        # Create RAG configuration
        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser=args.parser,
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=False,
        )
        
        # Create LLM functions
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=args.api_key,
                base_url=args.base_url,
                **kwargs,
            )
        
        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if messages:
                return openai_complete_if_cache(
                    "gpt-5.1",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    **kwargs,
                )
            elif image_data:
                return openai_complete_if_cache(
                    "gpt-5.1",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                                },
                            ],
                        } if image_data else {"role": "user", "content": prompt},
                    ],
                    api_key=args.api_key,
                    base_url=args.base_url,
                    **kwargs,
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
        
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        
        def _embedding_func_with_logging(texts):
            return openai_embed(
                texts,
                model=embedding_model,
                api_key=args.api_key,
                base_url=args.base_url,
            )
        
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=_embedding_func_with_logging,
        )
        
        # Initialize RAGAnything with speedup settings
        logger.info("Initializing RAGAnything...")
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
            lightrag_kwargs={
                "llm_model_max_async": args.max_async,
                "embedding_func_max_async": args.embedding_max_async,
                "embedding_batch_num": args.embedding_batch_num,
                "max_parallel_insert": args.max_parallel_insert,
                "default_llm_timeout": int(os.getenv("LLM_TIMEOUT", "180")),
            }
        )
        
        # Only run full processing if document hasn't been processed yet
        # - fully_processed: skip everything (just query)
        # - mineru_only: skip MinerU, do KG construction from existing content_list.json
        # - neither: run full pipeline
        if not fully_processed:
            if mineru_only:
                # Load existing content_list.json and do KG construction only
                logger.info("MinerU output exists - loading content_list.json for KG construction only...")
                content_list_path = find_content_list_json(output_dir)
                if content_list_path:
                    with open(content_list_path, 'r') as f:
                        content_list = json.load(f)
                    logger.info(f"Loaded {len(content_list)} content blocks from {content_list_path}")
                    
                    # Initialize LightRAG
                    await rag._ensure_lightrag_initialized()
                    
                    # Generate doc_id from content (same as RAGAnything does)
                    content_doc_id = rag._generate_content_based_doc_id(content_list)
                    logger.info(f"Generated doc_id: {content_doc_id}")
                    
                    # Separate text and multimodal content
                    text_content, multimodal_items = separate_content(content_list)
                    
                    # Fix image paths: make them absolute relative to content_list.json location
                    content_list_dir = os.path.dirname(content_list_path)
                    for item in multimodal_items:
                        if 'img_path' in item and not os.path.isabs(item['img_path']):
                            item['img_path'] = os.path.join(content_list_dir, item['img_path'])
                    
                    # Set content source for context extraction
                    if hasattr(rag, "set_content_source_for_context") and multimodal_items:
                        rag.set_content_source_for_context(content_list, rag.config.content_format)
                    
                    # Insert text content into KG
                    if text_content.strip():
                        file_name = os.path.basename(doc_path)
                        await insert_text_content(rag.lightrag, input=text_content, file_paths=file_name, ids=content_doc_id)
                    
                    # Process multimodal content
                    if multimodal_items:
                        await rag._process_multimodal_content(multimodal_items, doc_path, content_doc_id)
                    
                    # Track evidence for knowledge graph triples (required for retrieval by modality)
                    if rag.config.enable_evidence_tracking:
                        logger.info("Tracking evidence for KG triples...")
                        await rag._track_evidence_for_document(content_doc_id)
                    
                    logger.info("KG construction from existing MinerU output complete!")
                else:
                    logger.error(f"MinerU output directory exists but no content_list.json found in {output_dir}")
                    return []
            else:
                logger.info("Processing document with RAGAnything (running MinerU + KG)...")
                
                # Retry logic for memory errors during document processing
                max_retries = 3
                retry_count = 0
                processing_successful = False
                current_backend = args.backend  # Start with the configured backend
                
                while retry_count < max_retries and not processing_successful:
                    try:
                        await rag.process_document_complete(
                            file_path=doc_path,
                            output_dir=output_dir,
                            parse_method="auto",
                            device=args.device,
                            backend=current_backend,
                        )
                        processing_successful = True
                        logger.info("Document processing complete!")
                    
                    except MemoryError as e:
                        retry_count += 1
                        logger.error(f"MemoryError during document processing (attempt {retry_count}/{max_retries}): {e}")
                        
                        if retry_count < max_retries:
                            logger.info("Cleaning up and retrying in 5 seconds...")
                            # Clean up memory
                            gc.collect()
                            await asyncio.sleep(5)
                        else:
                            logger.error(f"Failed to process document after {max_retries} attempts due to memory errors")
                            raise
                            
                    except Exception as e:
                        error_str = str(e).lower()
                        
                        # Check if it's a memory-related error
                        if 'memory' in error_str or 'out of memory' in error_str or 'oom' in error_str:
                            retry_count += 1
                            logger.error(f"Possible memory error during document processing (attempt {retry_count}/{max_retries}): {e}")
                            
                            if retry_count < max_retries:
                                logger.info("Cleaning up and retrying in 5 seconds...")
                                gc.collect()
                                await asyncio.sleep(5)
                            else:
                                logger.error(f"Failed to process document after {max_retries} attempts")
                                raise
                        # Check if it's a backend-related error - fallback to pipeline
                        elif current_backend != "pipeline":
                            logger.warning(f"Backend '{current_backend}' failed for document: {e}")
                            logger.info("Falling back to 'pipeline' backend...")
                            current_backend = "pipeline"
                            retry_count += 1
                            gc.collect()
                            await asyncio.sleep(2)
                        else:
                            # Already using pipeline and still failing, re-raise
                            raise
        else:
            # Document already processed, just initialize LightRAG to load existing data
            logger.info("Skipping MinerU parsing - loading existing processed data...")
            await rag._ensure_lightrag_initialized()
        
        # RQ3: Test all modality subsets for each question
        results = []
        
        for i, question_data in enumerate(questions, 1):
            question = question_data['question']
            ground_truth = question_data['answer']
            answer_format = question_data['answer_format']
            gold_modalities = question_data['gold_modality_types']
            
            question_id = f"{doc_id}_{i}"
            
            logger.info(f"\n{'─'*60}")
            logger.info(f"Question {i}/{len(questions)} [{question_id}]")
            logger.info(f"  {question[:80]}...")
            logger.info(f"  Gold: {gold_modalities}")
            
            # Generate ALL subset experiments (including empty set)
            all_experiments = generate_modality_subsets(gold_modalities)
            
            # For questions with no gold modalities (e.g., "Not answerable"), 
            # test with all available modalities as a single subset
            if not all_experiments:
                all_experiments = [(3, frozenset(['text', 'image', 'table']), 'normal')]
            
            # Test each experiment with each model
            for subset_size, modality_subset, experiment_type in all_experiments:
                subset_list = sorted(list(modality_subset))
                
                # Test with each model
                for model_name in MODELS_TO_EVALUATE:
                    subset_tuple = tuple(sorted(subset_list))
                    
                    # Check if this question with this modality subset has already been answered
                    if check_question_answered(question_id, model_name, subset_tuple, results_by_model):
                        continue
                    
                    # Determine keep_alive for Ollama models:
                    # - Last question: unload model after prompting (keep_alive=0)
                    # - Not last question: keep model loaded (keep_alive=-1)
                    is_last_question = (i == len(questions))
                    keep_alive = 0 if is_last_question else -1
                    
                    logger.info(f"  {model_name}")
                    logger.info(f"    Subset: {subset_list}")
                    
                    raw_response, extracted_result, prediction, logprobs, retrieval_metadata, timing_metadata = await answer_with_modality_subset(
                        rag.lightrag,
                        question,
                        modality_subset,
                        extraction_prompt,
                        model_name=model_name,
                        api_key=args.api_key,
                        base_url=args.base_url,
                        keep_alive=keep_alive
                    )
                    
                    # Evaluate
                    score = eval_score(ground_truth, prediction, answer_format)
                    
                    # Store result with detailed metadata for analysis
                    result = {
                        # Identifiers
                        'question_id': question_id,
                        'doc_id': doc_id,
                        'doc_type': question_data.get('doc_type', ''),
                        'model': model_name,
                        
                        # Question information
                        'question_text': question_data.get('question', ''),
                        'answer': question_data.get('answer', ''),
                        'answer_format': question_data.get('answer_format', ''),
                        
                        # Modality configuration
                        'gold_modalities': tuple(sorted(gold_modalities)),  # e.g. ('image', 'table')
                        'subset_modalities': subset_tuple,                   # e.g. ('image',)
                        'subset_size': subset_size,                          # 1, 2, 3...
                        'experiment_type': experiment_type,                  # 'normal' or 'noise'
                        
                        # Model responses (no logprobs - stored separately)
                        'raw_response': raw_response,
                        'extracted_result': extracted_result,
                        'prediction': prediction,
                        'score': score,
                        
                        # Retrieved context metadata (including images_sent)
                        'retrieved_chunk_ids': retrieval_metadata.get('chunk_ids', []),
                        'retrieved_chunks_modalities': retrieval_metadata.get('chunks_modalities', {}),
                        'images_sent': retrieval_metadata.get('images_sent', 0),
                        
                        # Timing and token metrics
                        'input_tokens': timing_metadata.get('input_tokens', 0),
                        'output_tokens': timing_metadata.get('output_tokens', 0),
                        'inference_time_ms': timing_metadata.get('inference_time_ms', 0),
                        'retrieval_time_ms': timing_metadata.get('retrieval_time_ms', 0),
                        'total_time_ms': timing_metadata.get('retrieval_time_ms', 0) + timing_metadata.get('inference_time_ms', 0),
                        
                        # Additional metadata
                        'evidence_sources': question_data.get('evidence_sources', '[]'),
                        'evidence_pages': question_data.get('evidence_pages', '[]'),
                    }
                    
                    # Store logprobs separately with question_id + subset as composite key
                    if logprobs:
                        # Create composite key: question_id + sorted subset modalities
                        subset_key = f"{question_id}_{'_'.join(sorted(subset_tuple)) if subset_tuple else 'empty'}"
                        logprobs_entry = {
                            'question_id': question_id,
                            'subset_modalities': list(subset_tuple),
                            'model': model_name,
                            'logprobs': logprobs
                        }
                        results_logprobs[model_name][subset_key] = logprobs_entry
                    
                    results.append(result)
                    
                    # Log metrics
                    chunk_count = len(retrieval_metadata.get('chunk_ids', []))
                    images_sent = retrieval_metadata.get('images_sent', 0)
                    total_time = timing_metadata.get('retrieval_time_ms', 0) + timing_metadata.get('inference_time_ms', 0)
                    logger.info(f"    Chunks: {chunk_count} | Images: {images_sent} | Tokens: {timing_metadata.get('input_tokens', 0)} in / {timing_metadata.get('output_tokens', 0)} out")
                    logger.info(f"    Time: {total_time:.0f}ms (retrieval: {timing_metadata.get('retrieval_time_ms', 0):.0f}ms, inference: {timing_metadata.get('inference_time_ms', 0):.0f}ms)")
                    logger.info(f"    Score: {score:.3f}")
            
            # Save checkpoint after each question (not after all questions)
            # This prevents data loss if script is interrupted
            if results:
                # Add new results to the global results_by_model dictionary
                for result in results:
                    model = result['model']
                    # Check if this result is new (not already in results_by_model)
                    if result not in results_by_model[model]:
                        results_by_model[model].append(result)
                
                # Save checkpoint for each model with ALL accumulated results
                # Uses range suffix so parallel jobs write to separate files
                range_suffix = getattr(args, '_range_suffix', '')
                for model in MODELS_TO_EVALUATE:
                    # Save main results (without logprobs)
                    model_file = os.path.join(args.results_dir, f"{model.replace(':', '_')}_results_vlm{range_suffix}.json")
                    with open(model_file, 'w') as f:
                        json.dump(results_by_model[model], f, indent=2)
                    
                    # Save logprobs separately
                    logprobs_file = os.path.join(args.results_dir, f"{model.replace(':', '_')}_logprobs{range_suffix}.json")
                    with open(logprobs_file, 'w') as f:
                        json.dump(results_logprobs[model], f, indent=2)
                
        
        return results
        
    except RateLimitError as e:
        logger.error(f"RATE LIMIT ERROR - Stopping pipeline completely: {e}")
        raise  # Re-raise to stop the entire pipeline
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
        return []
    
    finally:
        # CRITICAL: Reset LightRAG's global shared storage state between documents.
        # Without this, _shared_dicts and _init_flags persist across LightRAG instances
        # in the same process, causing cross-document data contamination in rag_storage.
        finalize_share_data()


async def main_async(args):

    # Create results directory
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Compute range suffix early so checkpoint loading uses the correct files
    doc_start = getattr(args, 'doc_start', None)
    doc_end = getattr(args, 'doc_end', None)
    if doc_start is not None or doc_end is not None:
        range_suffix = f"_doc{doc_start}_to_{doc_end}"
    else:
        range_suffix = ""
    args._range_suffix = range_suffix
    
    # Always load existing results per model (automatic checkpointing)
    # Uses range_suffix so each parallel job reads/writes its own files
    results_by_model = {model: [] for model in MODELS_TO_EVALUATE}
    results_logprobs = {model: {} for model in MODELS_TO_EVALUATE}  # Dict of {question_id_subset: logprobs_entry}
    
    # Always check for existing results to avoid re-computation
    for model in MODELS_TO_EVALUATE:
        # Load main results
        model_file = os.path.join(results_dir, f"{model.replace(':', '_')}_results_vlm{range_suffix}.json")
        if os.path.exists(model_file):
            try:
                with open(model_file, 'r') as f:
                    loaded_results = json.load(f)
                
                # Normalize question_id and doc_id in loaded results to remove newlines
                for result in loaded_results:
                    if 'question_id' in result:
                        result['question_id'] = result['question_id'].replace('\n', '').replace('\r', '').strip()
                    if 'doc_id' in result:
                        result['doc_id'] = result['doc_id'].replace('\n', '').replace('\r', '').strip()
                
                results_by_model[model] = loaded_results
                logger.info(f"Loaded {len(results_by_model[model])} existing results for {model}")
            except Exception as e:
                logger.warning(f"Could not load existing results for {model}: {e}")
        
        # Load logprobs
        logprobs_file = os.path.join(results_dir, f"{model.replace(':', '_')}_logprobs{range_suffix}.json")
        if os.path.exists(logprobs_file):
            try:
                with open(logprobs_file, 'r') as f:
                    loaded_logprobs = json.load(f)
                
                # Normalize composite keys (question_id_subset) in logprobs to remove newlines
                normalized_logprobs = {}
                for subset_key, logprob_data in loaded_logprobs.items():
                    normalized_key = subset_key.replace('\n', '').replace('\r', '').strip()
                    # Also normalize question_id inside the entry if present
                    if 'question_id' in logprob_data:
                        logprob_data['question_id'] = logprob_data['question_id'].replace('\n', '').replace('\r', '').strip()
                    normalized_logprobs[normalized_key] = logprob_data
                
                results_logprobs[model] = normalized_logprobs
                logger.info(f"Loaded {len(results_logprobs[model])} existing logprobs for {model}")
            except Exception as e:
                logger.warning(f"Could not load existing logprobs for {model}: {e}")
    
    # Import here to avoid circular imports
    from .data_loader import load_samples
    
    # Load samples
    doc_questions = load_samples(args.samples)
    total_all_docs = len(doc_questions)
    
    # Document order for forwards processing (doc 1 to last)
    doc_items = list(doc_questions.items())
    
    # Slice by --doc-start / --doc-end (1-based, inclusive)
    # E.g. --doc-start 1 --doc-end 50 processes docs 1, 2, ..., 50
    if doc_start is not None or doc_end is not None:
        # Convert 1-based indices to 0-based slice indices
        # --doc-start 1 --doc-end 50 → slice [0:50]
        slice_start = (doc_start - 1) if doc_start else 0
        slice_end = doc_end if doc_end else total_all_docs
        
        doc_items = doc_items[slice_start:slice_end]
        logger.info(f"Processing documents {doc_start} -> {doc_end}, {len(doc_items)} documents")
    else:
        logger.info(f"Processing ALL {len(doc_items)} documents")
    
    if args.limit:
        doc_items = doc_items[:args.limit]
        logger.info(f"Limited to {args.limit} documents for testing")
    
    # Process each document (forwards)
    total_docs = len(doc_items)
    
    for doc_idx, (doc_id, questions) in enumerate(doc_items, 1):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"Document {doc_idx}/{total_docs}: {doc_id})")
        logger.info(f"{'#'*80}")
        
        _ = await process_document(doc_id, questions, args, results_by_model, results_logprobs)
        
        # Results are already added to results_by_model inside process_document
        logger.info(f"\nDocument {doc_idx}/{total_docs} completed. Results saved to {results_dir}/")
    
    # Summary stats
    logger.info(f"\n\n{'='*80}")
    logger.info("MODALITY CONTRIBUTIONS ANALYSIS")
    logger.info(f"{'='*80}")
    
    for model in MODELS_TO_EVALUATE:
        model_results = results_by_model[model]
        if not model_results:
            logger.warning(f"No results found for model: {model}")
            continue
        logger.info(f"{model}: {len(model_results)} results saved")
    
    logger.info(f"\nResults saved to: {results_dir}/")
    logger.info(f"{'='*80}\n")
