#!/usr/bin/env python
"""
1. For each question with gold evidence_sources (e.g., ['Chart', 'Table']):
   - Test all subsets: ['Chart'], ['Table'], ['Chart', 'Table']
   - Measure accuracy for each subset
   - Track which subset sizes reach accuracy threshold
2. Aggregate across questions to find modality minimalism patterns
3. Answer: "Can we answer with just 1 modality? When do we need 2? 3?"

Key Difference from run_modality_eval.py:
- That script: Tests ONLY single-modality questions with exact modality match
- This script: Tests ALL questions with ALL modality subsets to find minimum sufficient set
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set
from itertools import combinations

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger
from raganything import RAGAnything, RAGAnythingConfig
from mmlongbench.eval.eval_score import eval_score
from mmlongbench.eval.extract_answer import extract_answer
from lightrag.base import QueryParam
from lightrag.operate import kg_query
from lightrag.prompt import PROMPTS
from dataclasses import asdict

from ollama import chat
from ollama import ChatResponse

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)


# Models to evaluate
MODELS_TO_EVALUATE = [
    "gpt-4o-mini",
    "gemma3",
    "qwen3",
    "deepseek-r1"
]

# Map evidence_sources to modality types
MODALITY_MAPPING = {
    'Chart': 'image',
    'Figure': 'image',
    'Table': 'table',
    'Pure-text (Plain-text)': 'text',
    'Generalized-text (Layout)': 'text',
}


def load_samples(samples_path: str) -> Dict[str, List[Dict]]:
    """Load samples and group by document (including 'Not answerable' questions)"""
    with open(samples_path, 'r') as f:
        all_samples = json.load(f)
    
    doc_questions = defaultdict(list)
    
    for sample in all_samples:
        # Parse evidence_sources
        evidence_sources = eval(sample.get('evidence_sources', '[]'))
        
        # Map to modality types
        modality_types = []
        for source in evidence_sources:
            if source in MODALITY_MAPPING:
                modality_types.append(MODALITY_MAPPING[source])
            else:
                logger.warning(f"Unknown modality: {source}")
        
        # For questions without modality types (e.g., "Not answerable"), use empty list
        if not modality_types:
            modality_types = []
            evidence_sources = []
        
        # Store with both names and types
        sample['evidence_sources_list'] = evidence_sources
        sample['gold_modality_types'] = list(set(modality_types))  # unique types
        sample['gold_modality_names'] = evidence_sources
        doc_questions[sample['doc_id']].append(sample)
    
    total_questions = sum(len(q) for q in doc_questions.values())
    not_answerable_count = sum(1 for samples in doc_questions.values() for s in samples if s.get('answer') == 'Not answerable')
    logger.info(f"Loaded {total_questions} questions from {len(doc_questions)} documents")
    logger.info(f"Including {not_answerable_count} 'Not answerable' questions")
    
    return doc_questions


async def call_model(model_name: str, question: str, system_prompt: str, api_key: str = None, base_url: str = None) -> tuple:
    """
    Call a model (either OpenAI or Ollama) with the given question and system prompt.
    
    Args:
        model_name: Name of the model (e.g., "gpt-4o-mini", "gemma2", etc.)
        question: The user question
        system_prompt: The system prompt with context
        api_key: OpenAI API key (only for OpenAI models)
        base_url: Base URL for API (only for OpenAI models)
    
    Returns:
        Tuple of (response_text: str, logprobs: list | None)
    """
    try:
        if model_name.startswith("gpt-"):
            # OpenAI model - openai_complete_if_cache is async, await it directly
            result = await openai_complete_if_cache(
                model_name,
                question,
                system_prompt=system_prompt,
                history_messages=[],
                api_key=api_key,
                base_url=base_url,
                logprobs=True,
                top_logprobs=1  # Minimum value to get logprobs (valid range: 1-20)
            )
            # Handle both tuple (response, logprobs) and string returns
            if isinstance(result, tuple):
                response, logprobs = result
                return (response if isinstance(response, str) else str(response)), logprobs
            else:
                # Streaming or other case - no logprobs
                return (result if isinstance(result, str) else str(result)), None
        else:
            # Ollama model
            response: ChatResponse = await asyncio.to_thread(
                chat,
                model=model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt,
                    },
                    {
                        'role': 'user',
                        'content': question,
                    },
                ],
                logprobs=True
            )
            
            # Extract logprobs from Ollama response
            # NOTE: logprobs are on response object, NOT response.message
            logprobs_data = None
            
            if hasattr(response, 'logprobs') and response.logprobs:
                # Convert logprobs to serializable format
                try:
                    logprobs_data = [
                        {
                            'token': lp.get('token', ''),
                            'logprob': lp.get('logprob', 0.0),
                            'bytes': lp.get('bytes', []),
                            'top_logprobs': [
                                {
                                    'token': tlp.get('token', ''),
                                    'logprob': tlp.get('logprob', 0.0),
                                    'bytes': tlp.get('bytes', [])
                                }
                                for tlp in lp.get('top_logprobs', [])
                            ] if 'top_logprobs' in lp else []
                        }
                        for lp in response.logprobs
                    ]
                except Exception as e:
                    logger.error(f"Error converting logprobs: {e}")
                    logprobs_data = None
            
            return response.message.content, logprobs_data
    except Exception as e:
        logger.error(f"Error calling model {model_name}: {e}")
        return f"ERROR calling {model_name}: {str(e)}", None


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
    
    for size in range(1, len(unique_modalities) + 1):
        for subset in combinations(unique_modalities, size):
            subsets.append((size, frozenset(subset), 'normal'))
    
    return sorted(subsets, key=lambda x: x[0])  # Sort by size


def generate_noise_experiments(gold_modalities: List[str]) -> List[tuple]:
    """
    Generate noise experiments by adding ONE extra modality at a time.
    Returns list of tuples: (subset_size, frozenset_of_modalities, experiment_type)
    
    Example: ['text'] -> [
        (2, {'text', 'image'}, 'noise'),
        (2, {'text', 'table'}, 'noise')
    ]
    
    Example: ['text', 'image'] -> [
        (3, {'text', 'image', 'table'}, 'noise')
    ]
    """
    ALL_MODALITIES = ['text', 'image', 'table']
    gold_set = set(gold_modalities)
    noise_modalities = [m for m in ALL_MODALITIES if m not in gold_set]
    
    noise_experiments = []
    
    # Add +1 modality at a time
    for noise_mod in noise_modalities:
        combined = gold_set | {noise_mod}
        noise_experiments.append((len(combined), frozenset(combined), 'noise'))
    
    return noise_experiments


async def answer_with_modality_subset(
    lightrag,
    question: str,
    modality_subset: Set[str],
    extraction_prompt: str,
    model_name: str,
    api_key: str = None,
    base_url: str = None
) -> tuple:
    """
    Answer a question using ONLY the specified modality subset.
    Returns: (raw_response, extracted_result, prediction, logprobs, retrieval_metadata)
    where retrieval_metadata = {"chunk_ids": [...], "chunks_modalities": {"text": 1, "image": 2, ...}}
    """
    try:
        # Step 1: Query KG
        query_param = QueryParam(
            mode="hybrid",
            only_need_context=True,
            top_k=40,
            chunk_top_k=20,
            response_type="Multiple Paragraphs",
            stream=False,
            enable_rerank=True,
            max_entity_tokens=6000,
            max_relation_tokens=8000,
            max_total_tokens=30000,
        )
        
        global_config = asdict(lightrag)
        
        kg_result = await kg_query(
            question,
            lightrag.chunk_entity_relation_graph,
            lightrag.entities_vdb,
            lightrag.relationships_vdb,
            lightrag.text_chunks,
            query_param,
            global_config,
            hashing_kv=lightrag.llm_response_cache,
            chunks_vdb=lightrag.chunks_vdb,
        )
        
        # Step 2: Filter by modality subset using OR operator
        raw_data = kg_result.raw_data or {}
        data_section = raw_data.get("data", {})
        kg_relationships = data_section.get("relationships", [])
        
        filtered_relationships = []
        
        for kg_rel in kg_relationships:
            try:
                src_id = kg_rel.get("src_id", "")
                tgt_id = kg_rel.get("tgt_id", "")
                
                if not src_id or not tgt_id:
                    continue
                
                edge = await lightrag.chunk_entity_relation_graph.get_edge(src_id, tgt_id)
                if not edge:
                    continue
                
                # Check if relationship matches ANY modality in subset (OR operator)
                edge_modalities_str = edge.get("modality", "")
                if not edge_modalities_str:
                    continue
                
                edge_modalities = set(m.strip().lower() for m in edge_modalities_str.split(",") if m.strip())
                
                # Subset operator: include relationships whose modalities are fully contained in the test subset
                # Example: testing ['table','text'] includes pure 'table', pure 'text', and 'table,text' relationships
                # But testing ['table'] only includes pure 'table' relationships, not 'table,text'
                if set(edge_modalities).issubset(set(modality_subset)):
                    filtered_relationships.append({
                        "src_entity": src_id,
                        "tgt_entity": tgt_id,
                        "relation": kg_rel.get("keywords", ""),
                        "content": kg_rel.get("content"),
                        "edge": edge,
                    })
            except Exception as e:
                logger.debug(f"Error filtering relationship: {e}")
                continue
        
        # Initialize retrieval metadata
        retrieval_metadata = {"chunk_ids": [], "chunks_modalities": {}}
        
        if not filtered_relationships:
            raw_response = f"No information found in modalities {modality_subset} for this question."
        else:
            # Step 3: Build context and generate answer
            entity_ids_filtered = set()
            chunk_ids_filtered = set()
            chunk_to_modality = {}  # Map chunk_id -> modality from the edge it came from
            
            for rel in filtered_relationships:
                entity_ids_filtered.add(rel["src_entity"].lower())
                entity_ids_filtered.add(rel["tgt_entity"].lower())
                
                # Get modality of this edge
                edge_modalities_str = rel["edge"].get("modality", "")
                edge_modalities = set(m.strip().lower() for m in edge_modalities_str.split(",") if m.strip())
                
                evidence_summary_json = rel["edge"].get("evidence_summary_json")
                if evidence_summary_json:
                    try:
                        evidence_summary = json.loads(evidence_summary_json)
                        for src in evidence_summary.get("sources", []):
                            chunk_id = src.get("chunk_id")
                            if chunk_id:
                                chunk_ids_filtered.add(chunk_id)
                                # Track modality for this chunk (use first modality if multiple)
                                if chunk_id not in chunk_to_modality and edge_modalities:
                                    chunk_to_modality[chunk_id] = next(iter(edge_modalities))
                    except:
                        pass
            
            kg_entities_all = data_section.get("entities", [])
            kg_entities = [
                e for e in kg_entities_all
                if e.get("entity_name", "").lower() in entity_ids_filtered
            ]
            
            kg_chunks = []
            retrieved_chunk_ids = []
            retrieved_chunks_modalities = {}
            
            if chunk_ids_filtered:
                try:
                    chunk_ids_list = list(chunk_ids_filtered)
                    chunk_data_list = await lightrag.text_chunks.get_by_ids(chunk_ids_list)
                    for chunk_id, chunk_data in zip(chunk_ids_list, chunk_data_list):
                        if chunk_data and "content" in chunk_data:
                            kg_chunks.append({
                                "chunk_id": chunk_id,
                                "content": chunk_data["content"],
                            })
                            
                            # Track retrieved chunk IDs
                            retrieved_chunk_ids.append(chunk_id)
                            
                            # Track modalities for each chunk from the edge it came from
                            chunk_modality = chunk_to_modality.get(chunk_id, "text")
                            retrieved_chunks_modalities[chunk_modality] = retrieved_chunks_modalities.get(chunk_modality, 0) + 1
                except Exception as e:
                    logger.debug(f"Could not retrieve chunks: {e}")
            
            # Build context
            entities_str = "\n".join(
                json.dumps({
                    "entity": e.get("entity_name", ""),
                    "type": e.get("entity_type", ""),
                    "description": e.get("description", "")
                }, ensure_ascii=False)
                for e in kg_entities
            )
            
            relations_str = "\n".join(
                json.dumps({
                    "src_id": r["src_entity"],
                    "tgt_id": r["tgt_entity"],
                    "keywords": r["relation"],
                    "description": r["content"] or ""
                }, ensure_ascii=False)
                for r in filtered_relationships
            )
            
            chunks_str = "\n".join(
                json.dumps({"content": c.get("content", "")}, ensure_ascii=False)
                for c in kg_chunks
            )
            
            references = data_section.get("references", [])
            reference_list_str = "\n".join(
                f"[{ref['reference_id']}] {ref['file_path']}"
                for ref in references
                if ref.get("reference_id")
            )
            
            kg_context = PROMPTS["kg_query_context"].format(
                entities_str=entities_str,
                relations_str=relations_str,
                text_chunks_str=chunks_str,
                reference_list_str=reference_list_str,
            )
            
            sys_prompt = PROMPTS["rag_response"].format(
                response_type="Multiple Paragraphs",
                user_prompt="n/a",
                context_data=kg_context,
            )
            
            # Store retrieval metadata
            retrieval_metadata = {
                "chunk_ids": retrieved_chunk_ids,
                "chunks_modalities": retrieved_chunks_modalities
            }
            
            # Call the specified model (OpenAI or Ollama)
            raw_response, logprobs = await call_model(
                model_name=model_name,
                question=question,
                system_prompt=sys_prompt,
                api_key=api_key,
                base_url=base_url
            )
        
        if not raw_response:
            raw_response = "No answer found"
            logprobs = None
        
        # Extract answer using judge model
        extracted_result = extract_answer(
            question,
            raw_response,
            extraction_prompt,
            model_name="gpt-4o"
        )
        
        # Parse extracted answer
        try:
            prediction = extracted_result.split("Answer format:")[0].split("Extracted answer:")[1].strip()
        except:
            prediction = raw_response
        
        return raw_response, extracted_result, prediction, logprobs, retrieval_metadata
        
    except Exception as e:
        logger.error(f"Error in answer_with_modality_subset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"ERROR: {str(e)}", "Failed", f"ERROR: {str(e)}", None, {"chunk_ids": [], "chunks_modalities": {}}


async def process_document_rq3(doc_id: str, questions: List[Dict], args, results_by_model: Dict[str, List], results_logprobs: Dict[str, Dict]) -> List[Dict]:
    """
    Process document for RQ3: test all modality subsets for each question.
    
    Args:
        doc_id: Document identifier
        questions: List of questions for this document
        args: Command line arguments
        results_by_model: Dictionary of existing results grouped by model (for checkpointing)
    """
    doc_path = os.path.join(args.documents, doc_id)
    if not os.path.exists(doc_path):
        logger.error(f"Document not found: {doc_path}")
        return []
    
    # Load extraction prompt
    extraction_prompt_path = os.path.join(
        os.path.dirname(__file__),
        'eval',
        'prompt_for_answer_extraction.md'
    )
    with open(extraction_prompt_path, 'r') as f:
        extraction_prompt = f.read()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {doc_id}")
    logger.info(f"Questions: {len(questions)}")
    logger.info(f"{'='*80}")
    
    # Setup directories with new structure: processed_documents/{doc_name}/[output,rag_storage]
    doc_name = doc_id.replace('.pdf', '')
    doc_base_dir = os.path.join(args.processed_docs_dir, doc_name)
    working_dir = os.path.join(doc_base_dir, 'rag_storage')
    output_dir = os.path.join(doc_base_dir, 'output')
    
    # Check if document has already been processed
    already_processed = False
    if os.path.exists(working_dir) and os.path.exists(output_dir):
        # Check if rag_storage has content (key indicator files)
        rag_storage_files = os.listdir(working_dir) if os.path.exists(working_dir) else []
        output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
        
        # If both directories have files, assume already processed
        if len(rag_storage_files) > 0 and len(output_files) > 0:
            already_processed = True
            logger.info(f"📂 Document already processed! Using existing data from {doc_base_dir}")
    
    if not already_processed:
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create RAG configuration
        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser=args.parser,
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
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
        
        # Initialize RAGAnything
        logger.info("Initializing RAGAnything...")
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )
        
        # Always call process_document_complete to initialize lightrag
        # It will load existing data if already processed
        if not already_processed:
            logger.info("Processing document with RAGAnything...")
        else:
            logger.info("Loading existing processed data...")
        
        await rag.process_document_complete(
            file_path=doc_path,
            output_dir=output_dir,
            parse_method="auto"
        )
        
        if not already_processed:
            logger.info("Document processing complete!")
        else:
            logger.info("Existing data loaded!")
        
        # RQ3: Test all modality subsets for each question
        results = []
        
        for i, question_data in enumerate(questions, 1):
            question = question_data['question']
            ground_truth = question_data['answer']
            answer_format = question_data['answer_format']
            gold_modalities = question_data['gold_modality_types']
            
            logger.info(f"\n[{i}/{len(questions)}] Question: {question[:80]}...")
            logger.info(f"  Gold modalities: {gold_modalities}")
            
            # Generate normal (subset) experiments
            modality_subsets = generate_modality_subsets(gold_modalities)
            
            # Generate noise experiments (add extra modalities)
            noise_experiments = generate_noise_experiments(gold_modalities)
            
            # Combine both types of experiments
            all_experiments = modality_subsets + noise_experiments
            
            # For questions with no gold modalities (e.g., "Not answerable"), 
            # test with all available modalities as a single subset
            if not all_experiments:
                logger.info("  No gold modalities - testing with all available modalities")
                all_experiments = [(3, frozenset(['text', 'image', 'table']), 'normal')]
            else:
                logger.info(f"  Testing {len(modality_subsets)} normal experiments + {len(noise_experiments)} noise experiments")
            
            # Test each experiment with each model
            for subset_size, modality_subset, experiment_type in all_experiments:
                subset_list = sorted(list(modality_subset))
                logger.info(f"    Testing subset {subset_list} ({experiment_type})...")
                
                # Test with each model
                for model_name in MODELS_TO_EVALUATE:
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
                    question_id = f"{doc_id}_{i}"  # Unique identifier per question
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
                        'subset_modalities': tuple(sorted(subset_list)),     # e.g. ('image',)
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
                    
                    # Store logprobs separately
                    if logprobs:
                        logprobs_entry = {
                            'model': model_name,
                            'logprobs': logprobs
                        }
                        results_logprobs[model_name][question_id] = logprobs_entry
                    
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
    
    # Load existing results per model if resuming
    results_by_model = {model: [] for model in MODELS_TO_EVALUATE}
    results_logprobs = {model: {} for model in MODELS_TO_EVALUATE}  # Dict of {question_id: logprobs}
    
    if args.resume:
        for model in MODELS_TO_EVALUATE:
            # Load main results
            model_file = os.path.join(results_dir, f"{model.replace(':', '_')}_results.json")
            if os.path.exists(model_file):
                try:
                    with open(model_file, 'r') as f:
                        results_by_model[model] = json.load(f)
                    logger.info(f"Resuming {model}: found {len(results_by_model[model])} existing results")
                except Exception as e:
                    logger.warning(f"Could not load existing results for {model}: {e}")
            
            # Load logprobs
            logprobs_file = os.path.join(results_dir, f"{model.replace(':', '_')}_logprobs.json")
            if os.path.exists(logprobs_file):
                try:
                    with open(logprobs_file, 'r') as f:
                        results_logprobs[model] = json.load(f)
                    logger.info(f"Resuming {model}: found {len(results_logprobs[model])} existing logprobs")
                except Exception as e:
                    logger.warning(f"Could not load existing logprobs for {model}: {e}")
    
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


def main():
    parser = argparse.ArgumentParser(
        description="RQ3: Modality Minimalism Evaluation"
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
        default="./results",
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
    
    args = parser.parse_args()
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("Modality Contribution Analysis")
    print("="*80 + "\n")
    
    main()