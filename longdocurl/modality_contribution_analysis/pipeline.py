"""
Main pipeline for document processing and modality contribution analysis - LongDocURL benchmark.
"""

import os
import gc
import json
import asyncio
from pathlib import Path
from typing import Dict, List

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from openai import RateLimitError
from lightrag.utils import EmbeddingFunc, logger
from lightrag.kg.shared_storage import finalize_share_data

from raganything import RAGAnything, RAGAnythingConfig
from raganything.utils import separate_content, insert_text_content

from longdocurl.eval import eval_score

from .config import MODELS_TO_EVALUATE
from .modality_utils import (
    generate_modality_subsets,
    check_question_answered,
    check_all_questions_answered_for_document,
)
from .retrieval import answer_with_modality_subset
from .data_loader import find_pdf_path


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


async def process_document_rq3(doc_no: str, questions: List[Dict], args, results_by_model: Dict[str, List], results_logprobs: Dict[str, Dict]) -> List[Dict]:
    """
    Process document for RQ3: test all modality subsets for each question.
    This version includes actual images in the prompt for vision models.
    
    Args:
        doc_no: Document number (e.g., "4026369")
        questions: List of questions for this document
        args: Command line arguments
        results_by_model: Dictionary of existing results grouped by model (for checkpointing)
        results_logprobs: Dictionary for storing logprobs
    """
    # Check if ALL questions are already answered before processing document
    if check_all_questions_answered_for_document(doc_no, questions, results_by_model):
        logger.info(f"✅ ALL questions for {doc_no} are already answered. Skipping document processing entirely!")
        return []
    
    # Setup directories with new structure: processed_documents/{doc_no}/[output,rag_storage]
    doc_base_dir = os.path.join(args.processed_docs_dir, doc_no)
    working_dir = os.path.join(doc_base_dir, 'rag_storage')
    output_dir = os.path.join(doc_base_dir, 'output')
    
    # Check document processing state
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
        fully_processed = True
        logger.info(f"📂 Document fully processed! Using existing data from {doc_base_dir}")
    elif content_list_exists:
        mineru_only = True
        logger.info(f"📄 MinerU output exists, will do KG construction only for {doc_no}")
    
    # Find PDF path for this document
    doc_path = find_pdf_path(doc_no, args.documents)
    if not fully_processed and not mineru_only and not doc_path:
        logger.error(f"PDF not found for doc_no: {doc_no}")
        return []
    
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
    logger.info(f"Processing (WITH IMAGES): {doc_no}")
    logger.info(f"Questions: {len(questions)}")
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
            logger.info(f"🔗 [LLM CALL - Entity/Relation Extraction] Using base_url: {args.base_url}")
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
            logger.info(f"🔗 [VISION CALL - Document Processing] Using base_url: {args.base_url}")
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
                api_key=os.getenv("PERSONAL_EMBEDDING_OPENAI_API_KEY"),
                base_url=os.getenv("EMBEDDING_BINDING"),
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
                        file_name = os.path.basename(doc_path) if doc_path else f"{doc_no}.pdf"
                        await insert_text_content(rag.lightrag, input=text_content, file_paths=file_name, ids=content_doc_id)
                    
                    # Process multimodal content
                    if multimodal_items:
                        await rag._process_multimodal_content(multimodal_items, doc_path, content_doc_id)
                    
                    # Track evidence for knowledge graph triples
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
                current_backend = args.backend
                
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
                            gc.collect()
                            await asyncio.sleep(5)
                        else:
                            logger.error(f"Failed to process document after {max_retries} attempts due to memory errors")
                            raise
                            
                    except Exception as e:
                        error_str = str(e).lower()
                        
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
                        elif current_backend != "pipeline":
                            logger.warning(f"Backend '{current_backend}' failed for document: {e}")
                            logger.info("Falling back to 'pipeline' backend...")
                            current_backend = "pipeline"
                            retry_count += 1
                            gc.collect()
                            await asyncio.sleep(2)
                        else:
                            raise
        else:
            # Document already processed, just initialize LightRAG to load existing data
            logger.info("📂 Skipping MinerU parsing - loading existing processed data...")
            await rag._ensure_lightrag_initialized()
        
        # RQ3: Test all modality subsets for each question
        results = []
        
        for i, question_data in enumerate(questions, 1):
            question = question_data['question']
            ground_truth = question_data['answer']
            answer_format = question_data['answer_format']
            gold_modalities = question_data['gold_modality_types']
            base_question_id = question_data['question_id']  # Original question_id from dataset
            
            logger.info(f"\n{'─'*60}")
            logger.info(f"Question {i}/{len(questions)} [{base_question_id}]")
            logger.info(f"  {question[:80]}...")
            logger.info(f"  Gold: {gold_modalities}")
            
            # Generate ALL subset experiments (including empty set)
            all_experiments = generate_modality_subsets(gold_modalities)
            
            # For questions with no gold modalities, test with all available modalities
            if not all_experiments:
                all_experiments = [(3, frozenset(['text', 'image', 'table']), 'normal')]
            
            # Test each experiment with each model
            for subset_idx, (subset_size, modality_subset, experiment_type) in enumerate(all_experiments, 1):
                subset_list = sorted(list(modality_subset))
                
                # Create unique question_id per modality subset
                # e.g., "free_gpt4o_4026369_60_70_12_1" for first subset, "_2" for second, etc.
                question_id = f"{base_question_id}_{subset_idx}"
                
                # Test with each model
                for model_name in MODELS_TO_EVALUATE:
                    subset_tuple = tuple(sorted(subset_list))
                    
                    # Check if this question with this modality subset has already been answered
                    if check_question_answered(question_id, model_name, subset_tuple, results_by_model):
                        continue
                    
                    # Determine keep_alive for Ollama models
                    is_last_question = (i == len(questions))
                    keep_alive = 0 if is_last_question else -1
                    
                    logger.info(f"  🖼️ {model_name} (WITH IMAGES)")
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
                        'doc_no': doc_no,
                        'model': model_name,
                        
                        # Question information
                        'question_text': question,
                        'answer': ground_truth,
                        'answer_format': answer_format,
                        'question_type': question_data.get('question_type', ''),
                        'task_tag': question_data.get('task_tag', ''),
                        
                        # Modality configuration
                        'gold_modalities': tuple(sorted(gold_modalities)),
                        'subset_modalities': subset_tuple,
                        'subset_size': subset_size,
                        'experiment_type': experiment_type,
                        
                        # Model responses
                        'raw_response': raw_response,
                        'extracted_result': extracted_result,
                        'prediction': prediction,
                        'score': score,
                        
                        # Retrieved context metadata
                        'retrieved_chunk_ids': retrieval_metadata.get('chunk_ids', []),
                        'retrieved_chunks_modalities': retrieval_metadata.get('chunks_modalities', {}),
                        'images_sent': retrieval_metadata.get('images_sent', 0),
                        
                        # Timing and token metrics
                        'input_tokens': timing_metadata.get('input_tokens', 0),
                        'output_tokens': timing_metadata.get('output_tokens', 0),
                        'inference_time_ms': timing_metadata.get('inference_time_ms', 0),
                        'retrieval_time_ms': timing_metadata.get('retrieval_time_ms', 0),
                        'total_time_ms': timing_metadata.get('retrieval_time_ms', 0) + timing_metadata.get('inference_time_ms', 0),
                        
                        # Additional metadata from LongDocURL
                        'evidence_sources': question_data.get('evidence_sources', []),
                        'evidence_pages': question_data.get('evidence_pages', []),
                    }
                    
                    # Store logprobs separately
                    if logprobs:
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
            
            # Save checkpoint after each question
            if results:
                for result in results:
                    model = result['model']
                    if result not in results_by_model[model]:
                        results_by_model[model].append(result)
                
                # Save checkpoint for each model
                range_suffix = getattr(args, '_range_suffix', '')
                for model in MODELS_TO_EVALUATE:
                    # Save main results
                    model_file = os.path.join(args.results_dir, f"{model.replace(':', '_')}_results_vlm{range_suffix}.json")
                    with open(model_file, 'w') as f:
                        json.dump(results_by_model[model], f, indent=2)
                    
                    # Save logprobs separately
                    logprobs_file = os.path.join(args.results_dir, f"{model.replace(':', '_')}_logprobs{range_suffix}.json")
                    with open(logprobs_file, 'w') as f:
                        json.dump(results_logprobs[model], f, indent=2)
        
        return results
        
    except RateLimitError as e:
        logger.error(f"🛑 RATE LIMIT ERROR - Stopping pipeline completely: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing document {doc_no}: {e}", exc_info=True)
        return []
    
    finally:
        finalize_share_data()


async def main_async(args):
    """Main async function for RQ3 evaluation (WITH IMAGES)"""
    # Create results directory
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Compute range suffix
    doc_start = getattr(args, 'doc_start', None)
    doc_end = getattr(args, 'doc_end', None)
    if doc_start is not None or doc_end is not None:
        range_suffix = f"_doc{doc_start}_to_{doc_end}"
    else:
        range_suffix = ""
    args._range_suffix = range_suffix
    
    # Load existing results per model (automatic checkpointing)
    results_by_model = {model: [] for model in MODELS_TO_EVALUATE}
    results_logprobs = {model: {} for model in MODELS_TO_EVALUATE}
    
    for model in MODELS_TO_EVALUATE:
        # Load main results
        model_file = os.path.join(results_dir, f"{model.replace(':', '_')}_results_vlm{range_suffix}.json")
        if os.path.exists(model_file):
            try:
                with open(model_file, 'r') as f:
                    loaded_results = json.load(f)
                results_by_model[model] = loaded_results
                logger.info(f"📂 Loaded {len(results_by_model[model])} existing results for {model}")
            except Exception as e:
                logger.warning(f"Could not load existing results for {model}: {e}")
        
        # Load logprobs
        logprobs_file = os.path.join(results_dir, f"{model.replace(':', '_')}_logprobs{range_suffix}.json")
        if os.path.exists(logprobs_file):
            try:
                with open(logprobs_file, 'r') as f:
                    results_logprobs[model] = json.load(f)
                logger.info(f"📂 Loaded {len(results_logprobs[model])} existing logprobs for {model}")
            except Exception as e:
                logger.warning(f"Could not load existing logprobs for {model}: {e}")
    
    # Load samples from JSONL
    from .data_loader import load_samples_jsonl
    
    doc_questions = load_samples_jsonl(args.samples)
    
    # Build canonical document list from PDFs on disk (stable regardless of JSONL changes)
    all_pdf_doc_nos = sorted(
        [p.stem for p in Path(args.documents).glob("*/*.pdf")],
        key=lambda x: int(x),
    )
    total_all_docs = len(all_pdf_doc_nos)
    logger.info(f"📂 Found {total_all_docs} PDFs on disk in {args.documents}")
    
    # Reverse for backwards processing
    all_pdf_doc_nos.reverse()
    
    # Slice by --doc-start / --doc-end
    if doc_start is not None or doc_end is not None:
        rev_start = total_all_docs - doc_start if doc_start else 0
        rev_end = (total_all_docs - doc_end + 1) if doc_end else total_all_docs
        all_pdf_doc_nos = all_pdf_doc_nos[rev_start:rev_end]
        logger.info(f"📋 Processing documents {doc_start} -> {doc_end} (BACKWARDS), {len(all_pdf_doc_nos)} documents")
    else:
        logger.info(f"📋 Processing ALL {len(all_pdf_doc_nos)} documents (BACKWARDS)")
    
    # Pair each doc_no with its questions from the JSONL (empty list if no questions)
    doc_items = [(doc_no, doc_questions.get(doc_no, [])) for doc_no in all_pdf_doc_nos]
    
    if args.limit:
        doc_items = doc_items[:args.limit]
        logger.info(f"Limited to {args.limit} documents for testing")
    
    # Process each document
    total_docs = len(doc_items)
    
    for doc_idx, (doc_no, questions) in enumerate(doc_items, 1):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"Document {doc_idx}/{total_docs}: {doc_no} (WITH IMAGES, BACKWARDS)")
        logger.info(f"{'#'*80}")
        
        _ = await process_document_rq3(doc_no, questions, args, results_by_model, results_logprobs)
        
        logger.info(f"\n✅ Document {doc_idx}/{total_docs} completed. Results saved to {results_dir}/")
    
    # Summary stats
    logger.info(f"\n\n{'='*80}")
    logger.info("MODALITY CONTRIBUTIONS ANALYSIS (WITH IMAGES, BACKWARDS)")
    logger.info(f"{'='*80}")
    
    for model in MODELS_TO_EVALUATE:
        model_results = results_by_model[model]
        if not model_results:
            logger.warning(f"No results found for model: {model}")
            continue
        logger.info(f"{model}: {len(model_results)} results saved")
    
    logger.info(f"\nResults saved to: {results_dir}/")
    logger.info(f"{'='*80}\n")
