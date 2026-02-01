"""
Retrieval logic for answering questions with specific modality subsets.
"""

import json
from typing import Set

import numpy as np

from lightrag.utils import logger
from lightrag.prompt import PROMPTS

from mmlongbench.eval.extract_answer import extract_answer

from .api import call_model
from .config import GROUPED_MODALITIES


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
        # Step 1: Get ALL edges from graph and filter by modality FIRST
        # This is the correct order: filter by modality -> compute similarity -> top_k
        all_edges = await lightrag.chunk_entity_relation_graph.get_all_edges()
        logger.info(f"  📊 Total edges in graph: {len(all_edges)}")
        
        # Step 2: Filter by modality FIRST
        # Include edges that have AT LEAST ONE of the target modalities
        # (i.e., edges with ONLY the modality, OR that modality PLUS others)
        # Expand group names (layout, plain_text) to internal modalities
        modality_subset_lower = expand_modality_subset(modality_subset)
        logger.info(f"  🔄 Expanded modalities: {modality_subset} -> {modality_subset_lower}")
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
        
        logger.info(f"  🔍 After modality filter ({modality_subset}): {len(modality_filtered_edges)} edges (intersection logic)")
        
        # DEBUG: Log sample of filtered edges with their modalities
        if modality_filtered_edges:
            logger.info(f"  📋 Sample filtered edges (first 5):")
            for i, edge in enumerate(modality_filtered_edges[:5]):
                edge_mod = edge.get("modality", "")
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                logger.info(f"     [{i+1}] {src} -> {tgt} | edge_modality: {edge_mod}")
        
        # Step 3: Extract ALL chunks from modality-filtered edges
        # Then perform similarity search on chunk content (not edge metadata)
        FINAL_TOP_K = 40
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
        
        logger.info(f"  📄 Extracted {len(chunk_to_edges)} unique chunks from filtered edges")
        
        # DEBUG: Log chunk modality distribution
        modality_counts = {}
        for cid, mod in chunk_to_modality.items():
            modality_counts[mod] = modality_counts.get(mod, 0) + 1
        logger.info(f"  📊 Chunk modality distribution: {modality_counts}")
        logger.info(f"  🎯 Target modalities: {modality_subset_lower}")
        
        # DEBUG: Log sample of chunks with their modalities
        if chunk_to_modality:
            logger.info(f"  📋 Sample chunks kept (first 10):")
            for i, (cid, mod) in enumerate(list(chunk_to_modality.items())[:10]):
                logger.info(f"     [{i+1}] chunk_id: {cid[:30]}... | modality: {mod}")
        
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
            
            logger.info(f"  📝 Retrieved content for {len(chunk_contents)} chunks")
            
            if chunk_contents:
                # Compute query embedding (only embedding we need to compute!)
                query_embedding = await lightrag.embedding_func.func([question])
                query_embedding = query_embedding[0]
                query_vec = np.array(query_embedding)
                
                # Get pre-computed chunk vectors from VDB (no re-computation needed)
                chunk_ids_with_content = list(chunk_contents.keys())
                chunk_vectors = await lightrag.chunks_vdb.get_vectors_by_ids(chunk_ids_with_content)
                
                logger.info(f"  🔢 Retrieved {len(chunk_vectors)} pre-computed vectors from VDB")
                
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
                
                logger.info(f"  ✅ After similarity ranking: top {len(top_chunks)} chunks (top_k={FINAL_TOP_K})")
                
                # DEBUG: Log final top chunks and their modalities
                final_modality_counts = {}
                logger.info(f"  📋 Final top chunks modalities:")
                for i, (sim, cid) in enumerate(top_chunks[:10]):
                    mod = chunk_to_modality.get(cid, "unknown")
                    final_modality_counts[mod] = final_modality_counts.get(mod, 0) + 1
                    logger.info(f"     [{i+1}] sim={sim:.4f} | chunk_id: {cid[:30]}... | modality: {mod}")
                
                # Count all top chunks modalities
                for _, cid in top_chunks:
                    mod = chunk_to_modality.get(cid, "unknown")
                    if mod not in final_modality_counts:
                        final_modality_counts[mod] = 0
                # Recount properly
                final_modality_counts = {}
                for _, cid in top_chunks:
                    mod = chunk_to_modality.get(cid, "unknown")
                    final_modality_counts[mod] = final_modality_counts.get(mod, 0) + 1
                logger.info(f"  ✅ FINAL chunks modality distribution: {final_modality_counts}")
                
                # Verify all chunks match target modalities
                mismatched = [cid for _, cid in top_chunks if chunk_to_modality.get(cid, "").lower() not in modality_subset_lower]
                if mismatched:
                    logger.warning(f"  ⚠️ MISMATCH: {len(mismatched)} chunks have modality not in target {modality_subset_lower}!")
                else:
                    logger.info(f"  ✅ VERIFIED: All {len(top_chunks)} chunks match target modalities {modality_subset_lower}")
                
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
        retrieval_metadata = {"chunk_ids": [], "chunks_modalities": {}}
        
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
            
            logger.info(f"  🔍 No context for modalities {modality_subset} - testing parametric memory")
            
            raw_response, logprobs = await call_model(
                model_name=model_name,
                question=question,
                system_prompt=sys_prompt,
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
            
            # Build kg_chunks from top_chunk_ids (already retrieved during similarity search)
            kg_chunks = []
            retrieved_chunk_ids = top_chunk_ids
            retrieved_chunks_modalities = {}
            
            for chunk_id in top_chunk_ids:
                content = chunk_contents.get(chunk_id, None)
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
            
            # DEBUG: Log context details
            logger.info(f"  🔍 Context: {len(kg_entities)} entities, {len(filtered_relationships)} relations, {len(kg_chunks)} chunks")
            if kg_chunks:
                logger.info(f"  📄 First chunk preview: {kg_chunks[0].get('content', '')[:150]}...")
            
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
            model_name="gpt-4o",
            api_key=api_key
        )
        logger.info(f"  📝 Extracted result: {extracted_result[:100] if extracted_result else 'None'}...")
        
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
