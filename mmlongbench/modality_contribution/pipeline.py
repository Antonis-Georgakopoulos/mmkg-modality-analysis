"""
Main pipeline for document processing and modality contribution analysis.
"""

import os
import gc
import json
import asyncio
from typing import Dict, List

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger

from raganything import RAGAnything, RAGAnythingConfig

from mmlongbench.eval.eval_score import eval_score

from .config import MODELS_TO_EVALUATE
from .modality_utils import (
    generate_modality_subsets,
    check_question_answered,
    check_all_questions_answered_for_document,
)
from .retrieval import answer_with_modality_subset


async def process_document_rq3(doc_id: str, questions: List[Dict], args, results_by_model: Dict[str, List], results_logprobs: Dict[str, Dict]) -> List[Dict]:
    """
    Process document for RQ3: test all modality subsets for each question.
    
    Args:
        doc_id: Document identifier
        questions: List of questions for this document
        args: Command line arguments
        results_by_model: Dictionary of existing results grouped by model (for checkpointing)
    """
    # Check if ALL questions are already answered before processing document
    if check_all_questions_answered_for_document(doc_id, questions, results_by_model):
        logger.info(f"✅ ALL questions for {doc_id} are already answered. Skipping document processing entirely!")
        return []
    
    # Setup directories with new structure: processed_documents/{doc_name}/[output,rag_storage]
    doc_name = doc_id.replace('.pdf', '')
    doc_base_dir = os.path.join(args.processed_docs_dir, doc_name)
    working_dir = os.path.join(doc_base_dir, 'rag_storage')
    output_dir = os.path.join(doc_base_dir, 'output')
    
    # Check if document has already been processed (MinerU output exists)
    already_processed = False
    if os.path.exists(working_dir) and os.path.exists(output_dir):
        # Check if rag_storage has content (key indicator files)
        rag_storage_files = os.listdir(working_dir) if os.path.exists(working_dir) else []
        
        # Check for MinerU content_list.json output file
        content_list_exists = False
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith('_content_list.json'):
                    content_list_exists = True
                    break
            if content_list_exists:
                break
        
        # If rag_storage has files AND content_list.json exists, skip MinerU
        if len(rag_storage_files) > 0 and content_list_exists:
            already_processed = True
            logger.info(f"📂 Document already processed! Using existing data from {doc_base_dir}")
    
    # Only check for PDF if document is NOT already processed
    doc_path = os.path.join(args.documents, doc_id)
    if not already_processed and not os.path.exists(doc_path):
        logger.error(f"Document not found and no processed data exists: {doc_path}")
        return []
    
    if not already_processed:
        os.makedirs(working_dir, exist_ok=True)
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
    logger.info(f"Processing: {doc_id}")
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
                    "gpt-4o",
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
                    "gpt-4o",
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
        
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model=embedding_model,
                api_key=args.api_key,
                base_url=args.base_url,
            ),
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
            }
        )
        
        # Only run MinerU parsing if document hasn't been processed yet
        if not already_processed:
            logger.info("Processing document with RAGAnything (running MinerU)...")
            
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
            logger.info("📂 Skipping MinerU parsing - loading existing processed data...")
            await rag._ensure_lightrag_initialized()
        
        # RQ3: Test all modality subsets for each question
        results = []
        
        for i, question_data in enumerate(questions, 1):
            question = question_data['question']
            ground_truth = question_data['answer']
            answer_format = question_data['answer_format']
            gold_modalities = question_data['gold_modality_types']
            
            logger.info(f"\n[{i}/{len(questions)}] Question: {question[:80]}...")
            logger.info(f"  Gold modalities: {gold_modalities}")
            
            # Generate ALL subset experiments (including empty set)
            all_experiments = generate_modality_subsets(gold_modalities)
            
            # For questions with no gold modalities (e.g., "Not answerable"), 
            # test with all available modalities as a single subset
            if not all_experiments:
                logger.info("  No gold modalities - testing with all available modalities")
                all_experiments = [(3, frozenset(['text', 'image', 'table']), 'normal')]
            else:
                logger.info(f"  Testing {len(all_experiments)} subset experiments (including empty set)")
            
            # Test each experiment with each model
            for subset_size, modality_subset, experiment_type in all_experiments:
                subset_list = sorted(list(modality_subset))
                logger.info(f"    Testing subset {subset_list} ({experiment_type})...")
                
                # Test with each model
                for model_name in MODELS_TO_EVALUATE:
                    # Generate unique question ID
                    question_id = f"{doc_id}_{i}"
                    subset_tuple = tuple(sorted(subset_list))
                    
                    # Check if this question with this modality subset has already been answered
                    if check_question_answered(question_id, model_name, subset_tuple, results_by_model):
                        logger.info(f"      Model: {model_name} - SKIPPED (already answered)")
                        continue
                    
                    logger.info(f"      Model: {model_name}")
                    
                    raw_response, extracted_result, prediction, logprobs, retrieval_metadata = await answer_with_modality_subset(
                        rag.lightrag,
                        question,
                        modality_subset,
                        extraction_prompt,
                        model_name=model_name,
                        api_key=args.api_key,
                        base_url=args.base_url
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
                        
                        # Retrieved context metadata
                        'retrieved_chunk_ids': retrieval_metadata.get('chunk_ids', []),
                        'retrieved_chunks_modalities': retrieval_metadata.get('chunks_modalities', {}),
                        
                        # Additional metadata
                        'evidence_sources': question_data.get('evidence_sources', '[]'),
                        'evidence_pages': question_data.get('evidence_pages', '[]'),
                    }
                    
                    # Store logprobs separately with question_id + subset as composite key
                    logger.debug(f"  📊 Logprobs received: {logprobs is not None}, type: {type(logprobs)}")
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
                        logger.info(f"  📊 Saved logprobs for {model_name}: {len(logprobs)} tokens")
                    
                    results.append(result)
                    
                    logger.info(f"        Score: {score:.3f}")
            
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
                for model in MODELS_TO_EVALUATE:
                    # Save main results (without logprobs)
                    model_file = os.path.join(args.results_dir, f"{model.replace(':', '_')}_results.json")
                    with open(model_file, 'w') as f:
                        json.dump(results_by_model[model], f, indent=2)
                    
                    # Save logprobs separately
                    logprobs_file = os.path.join(args.results_dir, f"{model.replace(':', '_')}_logprobs.json")
                    with open(logprobs_file, 'w') as f:
                        json.dump(results_logprobs[model], f, indent=2)
                
                logger.info(f"  💾 Checkpoint saved after question {i}/{len(questions)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
        return []
    
    finally:
        # NOTE: We don't cleanup processed documents anymore since we want to reuse them
        # If you need to reprocess a document, manually delete its folder from processed_documents/
        pass


async def main_async(args):
    """Main async function for RQ3 evaluation"""
    # Create results directory
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Always load existing results per model (automatic checkpointing)
    results_by_model = {model: [] for model in MODELS_TO_EVALUATE}
    results_logprobs = {model: {} for model in MODELS_TO_EVALUATE}  # Dict of {question_id_subset: logprobs_entry}
    
    # Always check for existing results to avoid re-computation
    for model in MODELS_TO_EVALUATE:
        # Load main results
        model_file = os.path.join(results_dir, f"{model.replace(':', '_')}_results.json")
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
                logger.info(f"📂 Loaded {len(results_by_model[model])} existing results for {model}")
            except Exception as e:
                logger.warning(f"Could not load existing results for {model}: {e}")
        
        # Load logprobs
        logprobs_file = os.path.join(results_dir, f"{model.replace(':', '_')}_logprobs.json")
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
                logger.info(f"📂 Loaded {len(results_logprobs[model])} existing logprobs for {model}")
            except Exception as e:
                logger.warning(f"Could not load existing logprobs for {model}: {e}")
    
    # Import here to avoid circular imports
    from .data_loader import load_samples
    
    # Load samples
    doc_questions = load_samples(args.samples)
    
    if args.limit:
        doc_questions = dict(list(doc_questions.items())[:args.limit])
        logger.info(f"Limited to {args.limit} documents for testing")
    
    # Process each document
    total_docs = len(doc_questions)
    
    for doc_idx, (doc_id, questions) in enumerate(doc_questions.items(), 1):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"Document {doc_idx}/{total_docs}: {doc_id}")
        logger.info(f"{'#'*80}")
        
        _ = await process_document_rq3(doc_id, questions, args, results_by_model, results_logprobs)
        
        # Results are already added to results_by_model inside process_document_rq3
        logger.info(f"\n✅ Document {doc_idx}/{total_docs} completed. Results saved to {results_dir}/")
    
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
