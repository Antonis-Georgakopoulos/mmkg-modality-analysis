"""
Retrieval logic for answering questions with specific modality subsets (WITH IMAGES).

This version extracts image paths from chunks and sends actual images to vision models.
"""

import json
import re
import time
from typing import Set, List, Dict, Optional

import numpy as np

from lightrag.utils import logger
from lightrag.prompt import PROMPTS

from mmlongbench.eval.extract_answer import extract_answer

from .api import call_model_with_images, call_model_with_vlm_messages, encode_image_to_base64, get_image_mime_type
from .config import GROUPED_MODALITIES, IMAGE_MODALITIES


def expand_modality_subset(modality_subset: Set[str]) -> Set[str]:
    """
    Expand group names (layout, plain_text) to their internal modalities.
    Individual modalities (image, table) pass through unchanged.
    
    Example: {'image', 'layout'} -> {'image', 'header', 'footer', 'page_number', 'page_footnote'}
    """
    expanded = set()
    for mod in modality_subset:
        mod_lower = mod.lower()
        if mod_lower in GROUPED_MODALITIES:
            # Expand group to internal modalities
            expanded.update(GROUPED_MODALITIES[mod_lower])
        else:
            # Keep individual modality as-is
            expanded.add(mod_lower)
    return expanded


def extract_image_path_from_content(content: str) -> Optional[str]:
    """
    Extract image path from chunk content.
    
    The content contains "Image Path: /path/to/image.jpg" format.
    
    Args:
        content: Chunk content string
        
    Returns:
        Image path if found, None otherwise
    """
    # Pattern to match "Image Path: /path/to/image.ext"
    pattern = r"Image Path:\s*([^\r\n]+\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def process_chunks_with_vlm_markers(
    chunk_ids: List[str],
    chunk_contents: Dict[str, str],
    chunk_to_modality: Dict[str, str]
) -> tuple:
    """
    Process chunks and add VLM markers for images (matching raganything approach).
    
    This matches the approach in raganything/query.py:_process_image_paths_for_vlm
    where images are interleaved with text using [VLM_IMAGE_N] markers.
    
    Args:
        chunk_ids: List of chunk IDs to process
        chunk_contents: Dict mapping chunk_id -> content
        chunk_to_modality: Dict mapping chunk_id -> modality
        
    Returns:
        Tuple of (processed_contents: Dict[str, str], images: List[Dict])
        where processed_contents has VLM markers inserted
    """
    images = []
    processed_contents = {}
    seen_paths = set()  # Avoid duplicate images
    image_counter = 0
    
    # Pattern to match image paths (same as raganything)
    image_path_pattern = r"Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))"
    
    for chunk_id in chunk_ids:
        content = chunk_contents.get(chunk_id, "")
        if not content:
            processed_contents[chunk_id] = content
            continue
        
        modality = chunk_to_modality.get(chunk_id, "")
        
        # Only process image paths for chunks with image-capable modalities
        if modality not in IMAGE_MODALITIES:
            processed_contents[chunk_id] = content
            continue
        
        # Find and process image paths in this chunk's content
        def replace_image_path(match):
            nonlocal image_counter
            
            image_path = match.group(1).strip()
            
            # Skip duplicates
            if image_path in seen_paths:
                return match.group(0)  # Keep original, already have this image
            
            # Validate and encode image
            base64_data = encode_image_to_base64(image_path)
            if base64_data:
                image_counter += 1
                seen_paths.add(image_path)
                
                images.append({
                    "base64": base64_data,
                    "mime_type": get_image_mime_type(image_path),
                    "chunk_id": chunk_id,
                    "path": image_path,
                    "marker_num": image_counter
                })
                
                logger.info(f"📸 Found image #{image_counter} for chunk {chunk_id}: {image_path}")
                
                # Return original path + VLM marker (matching raganything approach)
                return f"Image Path: {image_path}\n[VLM_IMAGE_{image_counter}]"
            else:
                logger.warning(f"⚠️ Could not encode image: {image_path}")
                return match.group(0)  # Keep original
        
        # Process content with regex replacement
        processed_content = re.sub(image_path_pattern, replace_image_path, content)
        processed_contents[chunk_id] = processed_content
    
    return processed_contents, images


def build_vlm_messages_with_images(
    system_prompt: str,
    question: str,
    images: List[Dict[str, str]]
) -> List[Dict]:
    """
    Build VLM message format with interleaved images (matching raganything approach).
    
    This matches raganything/query.py:_build_vlm_messages_with_images
    
    Args:
        system_prompt: System prompt containing context with [VLM_IMAGE_N] markers
        question: User question
        images: List of image dicts with base64 data
        
    Returns:
        List of message dicts for OpenAI API
    """
    if not images:
        # Pure text mode
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    
    # Build multimodal content by splitting at image markers
    content_parts = []
    
    # Split the system prompt at image markers
    text_parts = system_prompt.split("[VLM_IMAGE_")
    
    for i, text_part in enumerate(text_parts):
        if i == 0:
            # First text part (before any image marker)
            if text_part.strip():
                content_parts.append({"type": "text", "text": text_part})
        else:
            # Find marker number and insert corresponding image
            marker_match = re.match(r"(\d+)\](.*)", text_part, re.DOTALL)
            if marker_match:
                image_num = int(marker_match.group(1)) - 1  # Convert to 0-based index
                remaining_text = marker_match.group(2)
                
                # Insert corresponding image
                if 0 <= image_num < len(images):
                    img = images[image_num]
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img['mime_type']};base64,{img['base64']}",
                            "detail": "high"
                        }
                    })
                
                # Insert remaining text
                if remaining_text.strip():
                    content_parts.append({"type": "text", "text": remaining_text})
    
    # Add user question at the end
    content_parts.append({
        "type": "text",
        "text": f"\n\nUser Question: {question}\n\nPlease answer based on the context and images provided."
    })
    
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that can analyze both text and image content to provide comprehensive answers."
        },
        {"role": "user", "content": content_parts}
    ]


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
    This version includes actual images in the prompt for vision models.
    
    Returns: (raw_response, extracted_result, prediction, logprobs, retrieval_metadata, timing_metadata)
    where retrieval_metadata = {"chunk_ids": [...], "chunks_modalities": {"text": 1, "image": 2, ...}, "images_sent": int}
    and timing_metadata = {"retrieval_time_ms": ..., "inference_time_ms": ..., "input_tokens": ..., "output_tokens": ...}
    """
    try:
        retrieval_start = time.perf_counter()
        
        # Step 1: Get ALL edges from graph and filter by modality FIRST
        # This is the correct order: filter by modality -> compute similarity -> top_k
        all_edges = await lightrag.chunk_entity_relation_graph.get_all_edges()
        
        # Step 2: Filter by modality FIRST
        # Include edges that have AT LEAST ONE of the target modalities
        # (i.e., edges with ONLY the modality, OR that modality PLUS others)
        # Expand group names (layout, plain_text) to internal modalities
        modality_subset_lower = expand_modality_subset(modality_subset)
        modality_filtered_edges = []
        for edge in all_edges:
            edge_modalities_str = edge.get("modality", "")
            if not edge_modalities_str:
                continue
            
            edge_modalities = set(m.strip().lower() for m in edge_modalities_str.split(",") if m.strip())
            
            # Intersection: include edge if it has AT LEAST ONE of the target modalities
            # e.g., target={table} matches edges with {table} or {table,text}
            if edge_modalities.intersection(modality_subset_lower):
                modality_filtered_edges.append(edge)
        
        
        # Step 3: Extract ALL chunks from modality-filtered edges
        # Then perform similarity search on chunk content (not edge metadata)
        FINAL_TOP_K = 20
        chunk_to_edges = {}  # chunk_id -> list of edges that reference it
        chunk_to_modality = {}  # chunk_id -> modality
        
        for edge in modality_filtered_edges:
            evidence_summary_json = edge.get("evidence_summary_json")
            if not evidence_summary_json:
                continue
            try:
                evidence_summary = json.loads(evidence_summary_json)
                
                for src in evidence_summary.get("sources", []):
                    chunk_id = src.get("chunk_id")
                    # Get the modality for THIS specific chunk/evidence
                    chunk_modality = src.get("modality", "").lower().strip()
                    
                    if chunk_id and chunk_modality:
                        # Only include chunks whose modality is in the filter set
                        if chunk_modality in modality_subset_lower:
                            if chunk_id not in chunk_to_edges:
                                chunk_to_edges[chunk_id] = []
                            chunk_to_edges[chunk_id].append(edge)
                            # Track modality for this chunk
                            chunk_to_modality[chunk_id] = chunk_modality
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        
        # Step 4: Get chunk contents and compute similarity
        filtered_relationships = []
        top_chunk_ids = []
        chunk_contents = {}
        
        if chunk_to_edges:
            chunk_ids_list = list(chunk_to_edges.keys())
            chunk_data_list = await lightrag.text_chunks.get_by_ids(chunk_ids_list)
            
            # Build chunk_id -> content mapping
            for chunk_id, chunk_data in zip(chunk_ids_list, chunk_data_list):
                if chunk_data and "content" in chunk_data:
                    chunk_contents[chunk_id] = chunk_data["content"]
            
            
            if chunk_contents:
                # Compute query embedding (only embedding we need to compute!)
                query_embedding = await lightrag.embedding_func.func([question])
                query_embedding = query_embedding[0]
                query_vec = np.array(query_embedding)
                
                # Get pre-computed chunk vectors from VDB (no re-computation needed)
                chunk_ids_with_content = list(chunk_contents.keys())
                chunk_vectors = await lightrag.chunks_vdb.get_vectors_by_ids(chunk_ids_with_content)
                
                
                # Compute cosine similarities using pre-computed vectors
                similarities = []
                for chunk_id in chunk_ids_with_content:
                    if chunk_id not in chunk_vectors:
                        continue
                    chunk_vec = np.array(chunk_vectors[chunk_id])
                    similarity = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-10)
                    similarities.append((similarity, chunk_id))
                
                # Sort by similarity (descending) and take top_k chunks
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_chunks = similarities[:FINAL_TOP_K]
                top_chunk_ids = [cid for _, cid in top_chunks]
                
                # ===== DETAILED LOGGING: Retrieved chunks =====
                logger.info("=" * 60)
                logger.info("📋 RETRIEVED CHUNKS FOR QUESTION:")
                logger.info(f"   Q: {question[:100]}...")
                logger.info(f"   Total chunks retrieved: {len(top_chunk_ids)}")
                
                # Group chunks by modality
                chunks_by_modality = {}
                for chunk_id in top_chunk_ids:
                    modality = chunk_to_modality.get(chunk_id, "unknown")
                    if modality not in chunks_by_modality:
                        chunks_by_modality[modality] = []
                    chunks_by_modality[modality].append(chunk_id)
                
                logger.info("   Chunks by modality:")
                for mod, cids in sorted(chunks_by_modality.items()):
                    logger.info(f"      - {mod}: {len(cids)} chunks")
                
                # Log image/table chunks specifically
                image_table_chunks = [cid for cid in top_chunk_ids 
                                     if chunk_to_modality.get(cid, "") in IMAGE_MODALITIES]
                if image_table_chunks:
                    logger.info(f"   🖼️ Image/Table chunks ({len(image_table_chunks)}):")
                    for cid in image_table_chunks:
                        mod = chunk_to_modality.get(cid, "unknown")
                        content_preview = chunk_contents.get(cid, "")[:150].replace("\n", " ")
                        logger.info(f"      - [{mod}] {cid}: {content_preview}...")
                else:
                    logger.info("   ⚠️ No image/table chunks in retrieved set")
                logger.info("=" * 60)
                
                # Build filtered_relationships from edges that reference top chunks
                seen_edges = set()
                for chunk_id in top_chunk_ids:
                    for edge in chunk_to_edges.get(chunk_id, []):
                        edge_key = (edge.get("source", ""), edge.get("target", ""))
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            filtered_relationships.append({
                                "src_entity": edge.get("source", ""),
                                "tgt_entity": edge.get("target", ""),
                                "relation": edge.get("keywords", ""),
                                "content": edge.get("description", ""),
                                "edge": edge,
                            })
        
        # Initialize retrieval metadata
        retrieval_metadata = {"chunk_ids": [], "chunks_modalities": {}, "images_sent": 0}
        
        if not filtered_relationships:
            # No context from RAG - call LLM with empty context to test parametric memory
            kg_context = PROMPTS["kg_query_context"].format(
                entities_str="",
                relations_str="",
                text_chunks_str="",
                reference_list_str="",
            )
            
            sys_prompt = PROMPTS["rag_response"].format(
                response_type="Multiple Paragraphs",
                user_prompt="n/a",
                context_data=kg_context,
            )
            
            
            raw_response, logprobs, input_tokens, output_tokens, inference_time_ms = await call_model_with_images(
                model_name=model_name,
                question=question,
                system_prompt=sys_prompt,
                images=[],
                api_key=api_key,
                base_url=base_url
            )
        else:
            # Step 5: Build context using top_chunk_ids (already computed via similarity)
            # Get entities from filtered relationships
            entity_ids_filtered = set()
            for rel in filtered_relationships:
                entity_ids_filtered.add(rel["src_entity"].lower())
                entity_ids_filtered.add(rel["tgt_entity"].lower())
            
            # Get entity data directly from the graph
            kg_entities = []
            entity_names_list = list(entity_ids_filtered)
            if entity_names_list:
                nodes_dict = await lightrag.chunk_entity_relation_graph.get_nodes_batch(entity_names_list)
                for entity_name in entity_names_list:
                    node = nodes_dict.get(entity_name)
                    if node:
                        kg_entities.append({"entity_name": entity_name, **node})
            
            # ===== Process chunks with VLM markers (matching raganything approach) =====
            processed_contents, images_to_send = process_chunks_with_vlm_markers(
                top_chunk_ids,
                chunk_contents,
                chunk_to_modality
            )
            
            # ===== DETAILED LOGGING: Images extracted =====
            logger.info("=" * 60)
            logger.info("🖼️ IMAGES EXTRACTED FROM CHUNKS:")
            if images_to_send:
                logger.info(f"   Total images: {len(images_to_send)}")
                for i, img in enumerate(images_to_send, 1):
                    logger.info(f"   Image #{i}:")
                    logger.info(f"      - Chunk ID: {img['chunk_id']}")
                    logger.info(f"      - Path: {img['path']}")
                    logger.info(f"      - MIME type: {img['mime_type']}")
                    logger.info(f"      - VLM marker: [VLM_IMAGE_{img['marker_num']}]")
            else:
                logger.info("   ⚠️ No images extracted (no image/table chunks or no valid image paths)")
            logger.info("=" * 60)
            
            # Build kg_chunks using processed contents (with VLM markers)
            kg_chunks = []
            retrieved_chunk_ids = top_chunk_ids
            retrieved_chunks_modalities = {}
            
            for chunk_id in top_chunk_ids:
                content = processed_contents.get(chunk_id, chunk_contents.get(chunk_id, None))
                if content:
                    kg_chunks.append({
                        "chunk_id": chunk_id,
                        "content": content,
                    })
                    # Track modalities
                    chunk_modality = chunk_to_modality.get(chunk_id, "text")
                    retrieved_chunks_modalities[chunk_modality] = retrieved_chunks_modalities.get(chunk_modality, 0) + 1
            
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
            
            # Use processed content (with VLM markers) for chunks
            chunks_str = "\n".join(
                json.dumps({"content": c.get("content", "")}, ensure_ascii=False)
                for c in kg_chunks
            )
            
            # References are not available without kg_query, set empty
            reference_list_str = ""
            
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
            
            # ===== DETAILED LOGGING: Final prompt summary =====
            logger.info("=" * 60)
            logger.info("📝 PROMPT SUMMARY:")
            logger.info(f"   Question: {question[:80]}...")
            logger.info(f"   Entities in context: {len(kg_entities)}")
            logger.info(f"   Relations in context: {len(filtered_relationships)}")
            logger.info(f"   Chunks in context: {len(kg_chunks)}")
            logger.info(f"   Images attached: {len(images_to_send)}")
            logger.info(f"   System prompt length: {len(sys_prompt)} chars")
            
            # Log a snippet of the context showing VLM markers if present
            if images_to_send:
                # Find and log where VLM markers appear in the prompt
                vlm_markers_in_prompt = [f"[VLM_IMAGE_{i}]" for i in range(1, len(images_to_send) + 1)]
                markers_found = [m for m in vlm_markers_in_prompt if m in sys_prompt]
                logger.info(f"   VLM markers in prompt: {markers_found}")
            logger.info("=" * 60)
            
            # Store retrieval metadata
            retrieval_metadata = {
                "chunk_ids": retrieved_chunk_ids,
                "chunks_modalities": retrieved_chunks_modalities,
                "images_sent": len(images_to_send)
            }
            
            # Log full prompt for debugging (save to file since it can be very long)
            prompt_log_path = "./results_vlm/debug_full_prompt.txt"
            try:
                import os
                os.makedirs(os.path.dirname(prompt_log_path), exist_ok=True)
                with open(prompt_log_path, "w", encoding="utf-8") as f:
                    f.write("=" * 80 + "\n")
                    f.write("🔍 FULL PROMPT SENT TO VLM (WITH IMAGES - INTERLEAVED)\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Question: {question}\n\n")
                    f.write(f"Number of entities: {len(kg_entities)}\n")
                    f.write(f"Number of relations: {len(filtered_relationships)}\n")
                    f.write(f"Number of chunks: {len(kg_chunks)}\n")
                    f.write(f"Number of images: {len(images_to_send)}\n")
                    if images_to_send:
                        f.write(f"Image paths: {[img['path'] for img in images_to_send]}\n")
                    f.write("\n" + "-" * 40 + " SYSTEM PROMPT START " + "-" * 40 + "\n")
                    f.write(sys_prompt)
                    f.write("\n" + "-" * 40 + " SYSTEM PROMPT END " + "-" * 40 + "\n")
                    f.write("=" * 80 + "\n")
                logger.info(f"📝 Full prompt saved to: {prompt_log_path}")
            except Exception as e:
                logger.warning(f"Could not save debug prompt: {e}")
            
            # Build VLM messages with interleaved images (matching raganything approach)
            vlm_messages = build_vlm_messages_with_images(sys_prompt, question, images_to_send)
            
            # Call the vision model with properly formatted messages
            raw_response, logprobs, input_tokens, output_tokens, inference_time_ms = await call_model_with_vlm_messages(
                model_name=model_name,
                messages=vlm_messages,
                api_key=api_key,
                base_url=base_url
            )
        
        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000 - inference_time_ms
        
        if not raw_response:
            raw_response = "No answer found"
            logprobs = None
        
        # Build timing metadata
        timing_metadata = {
            "retrieval_time_ms": retrieval_time_ms,
            "inference_time_ms": inference_time_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        
        # Extract answer using judge model
        extracted_result = extract_answer(
            question,
            raw_response,
            extraction_prompt,
            model_name="gpt-4o",
            api_key=api_key
        )
        
        # Parse extracted answer
        try:
            prediction = extracted_result.split("Answer format:")[0].split("Extracted answer:")[1].strip()
        except (IndexError, AttributeError):
            prediction = raw_response
        
        return raw_response, extracted_result, prediction, logprobs, retrieval_metadata, timing_metadata
        
    except Exception as e:
        logger.error(f"Error in answer_with_modality_subset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        empty_timing = {"retrieval_time_ms": 0, "inference_time_ms": 0, "input_tokens": 0, "output_tokens": 0}
        return f"ERROR: {str(e)}", "Failed", f"ERROR: {str(e)}", None, {"chunk_ids": [], "chunks_modalities": {}, "images_sent": 0}, empty_timing
