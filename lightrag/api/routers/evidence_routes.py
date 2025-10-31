"""Evidence-based modality query routes for LightRAG API.

This module provides advanced query capabilities that filter results based on
evidence modalities (text, image, table, mathematical formulas) stored in the
knowledge graph relationships.
"""

import json
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Set
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.base import QueryParam
from lightrag.operate import kg_query, extract_keywords_only

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Supported evidence modality types."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"  # mathematical formulas
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class LogicalOperator(str, Enum):
    """Logical operators for combining modality filters."""
    AND = "AND"  # All requested modalities must be present
    OR = "OR"    # At least one requested modality must be present
    ONLY = "ONLY"  # Exactly the requested modalities (no more, no less)


class SearchMode(str, Enum):
    """Search mode for evidence query."""
    KEYWORD = "keyword"  # Direct keyword matching (fast, no LLM)
    LLM = "llm"  # Use LLM to extract keywords from natural language question


class ModalityQueryRequest(BaseModel):
    """Request model for evidence-based modality search."""

    search_mode: SearchMode = Field(
        default=SearchMode.KEYWORD,
        description="Search mode: 'keyword' for direct matching, 'llm' for natural language understanding",
    )

    query: str = Field(
        ...,
        description="For keyword mode: space-separated keywords. For LLM mode: natural language question.",
        examples=["dog chase", "What is the relationship between the dog and duck?"],
    )

    question: Optional[str] = Field(
        None,
        description="(Optional) Natural language question for answer generation when generate_answer=True",
        examples=["Why does the dog chase the duck?"],
    )

    generate_answer: bool = Field(
        default=False,
        description="Whether to generate a natural language answer using LLM based on filtered evidence",
    )
    
    modalities: List[ModalityType] = Field(
        default=[ModalityType.TEXT],
        description="List of modalities to filter by",
        examples=[["text", "image"], ["table", "equation"]]
    )
    
    modality_operator: LogicalOperator = Field(
        default=LogicalOperator.OR,
        description="Logical operator for combining modalities: AND requires all modalities, OR requires at least one"
    )
    
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = Field(
        default="mix",
        description="Query mode: local, global, hybrid, naive, or mix (mix is recommended for most comprehensive results)"
    )
    
    min_evidence_count: int = Field(
        default=1,
        ge=1,
        description="Minimum number of evidence pieces required for a relationship to be included"
    )
    
    include_evidence_details: bool = Field(
        default=True,
        description="Include detailed evidence information (source chunks, modalities, document references) in the response"
    )
    
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of relationship results to return"
    )
    
    sort_by: Literal["relevance", "evidence_strength"] = Field(
        default="relevance",
        description="Sort relationships by query relevance (similarity) or evidence strength (support count)"
    )
    
    # Additional parameters for LLM mode (kg_query)
    entity_top_k: int = Field(
        default=40,
        ge=1,
        le=100,
        description="Number of top entities to retrieve in LLM mode (matches system DEFAULT_TOP_K)"
    )
    
    chunk_top_k: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of top chunks to retrieve in LLM mode (matches system DEFAULT_CHUNK_TOP_K)"
    )
    
    response_type: str = Field(
        default="Multiple Paragraphs",
        description="Response format preference for LLM answer generation"
    )


class EvidenceDetail(BaseModel):
    """Detailed evidence information for a relationship."""
    chunk_id: str
    file_path: Optional[str] = None
    modality: str
    doc_id: Optional[str] = None
    timestamp: Optional[int] = None
    content: Optional[str] = Field(None, description="The actual content of the chunk")


class RelationshipWithEvidence(BaseModel):
    """Relationship with evidence metadata."""
    src_entity: str
    tgt_entity: str
    relation: str
    content: Optional[str] = None
    fact_key: Optional[str] = None
    evidence_total_count: int = 0
    evidence_doc_count: int = 0
    modalities: List[str] = Field(default_factory=list, description="List of modalities supporting this relationship")
    evidence_details: Optional[List[EvidenceDetail]] = None


class ModalityQueryResponse(BaseModel):
    """Response model for modality-based queries."""
    query: str
    search_mode: str = Field(description="Search mode used (keyword or llm)")
    modality_filter: List[str]
    modality_operator: str
    sort_by: str = Field(description="Sorting method used (relevance or evidence_strength)")
    total_relationships_found: int
    relationships: List[RelationshipWithEvidence]
    extracted_keywords: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Keywords extracted by LLM when search_mode=llm"
    )
    generated_answer: Optional[str] = Field(
        None,
        description="Natural language answer generated by LLM when generate_answer=True"
    )
    total_time_seconds: float = Field(
        description="Total processing time in seconds"
    )
    timing_breakdown: Optional[Dict[str, float]] = Field(
        None,
        description="Detailed timing breakdown for each processing stage (in seconds)"
    )


def create_evidence_routes(rag, api_key: Optional[str] = None):
    """Create evidence-based query routes.
    
    Args:
        rag: LightRAG instance
        api_key: Optional API key for authentication
    
    Returns:
        APIRouter with evidence query endpoints
    """
    
    router = APIRouter(tags=["evidence"], prefix="/evidence")
    
    @router.post(
        "/query",
        response_model=ModalityQueryResponse,
        summary="Query with modality filtering",
        description="""Search the knowledge graph with advanced modality-based filtering.
        
        This endpoint allows you to:
        - Filter relationships by evidence modalities (text, image, table, equation)
        - Combine modality filters with AND/OR/ONLY logic
        - Retrieve detailed evidence metadata
        - Get answers based only on evidence from specific modalities
        
        Modality Operators:
        - OR: Relationship has at least one of the selected modalities
        - AND: Relationship has all of the selected modalities (can have others too)
        - ONLY: Relationship has exactly the selected modalities (no more, no less)
        
        Example use cases:
        - Find relationships supported by images AND tables
        - Get relationships with ONLY text evidence (no images/tables)
        - Search only mathematical formulas and equations
        """,
    )
    async def query_with_modality_filter(
        request: ModalityQueryRequest,
        auth: Any = Depends(get_combined_auth_dependency(api_key)),
    ) -> ModalityQueryResponse:
        """Query knowledge graph with modality-based evidence filtering."""
        # Start timing (outside try block so it's always accessible)
        start_time = time.time()
        timings = {}
        
        try:
            logger.info(
                f"[EVIDENCE_QUERY] Mode: {request.search_mode.value} | RAG Mode: {request.mode} | "
                f"Query: '{request.query}' | Modalities: {request.modalities} | Operator: {request.modality_operator.value}"
            )
            
            modality_set = set(m.value for m in request.modalities)
            extracted_keywords_dict = None
            generated_answer = None
            relationships_with_evidence = []
            
            # ========== LLM MODE: Use full kg_query pipeline (same as Retrieval page) ==========
            if request.search_mode == SearchMode.LLM:
                logger.info("[EVIDENCE_QUERY] ===== LLM MODE: Using full RAG pipeline (kg_query) =====")
                
                # Build QueryParam for kg_query (match all parameters from original query)
                # Set only_need_context=True to get context without generating answer yet
                query_param = QueryParam(
                    mode=request.mode,
                    only_need_context=True,  # Get context only, we'll generate answer after filtering
                    top_k=request.entity_top_k,
                    chunk_top_k=request.chunk_top_k,
                    response_type=request.response_type,
                    stream=False,  # Never stream in this endpoint
                    enable_rerank=True,  # Enable reranking for better relevance
                    # Use system defaults for token limits (critical for complete retrieval!)
                    max_entity_tokens=6000,
                    max_relation_tokens=8000,
                    max_total_tokens=30000,
                )
                
                global_config = asdict(rag)
                
                try:
                    # Call the FULL RAG pipeline (same as WebUI Retrieval)
                    logger.info(f"[EVIDENCE_QUERY] Calling kg_query with mode={request.mode}, top_k={request.entity_top_k}")
                    t_retrieval_start = time.time()
                    
                    kg_result = await kg_query(
                        request.query,
                        rag.chunk_entity_relation_graph,
                        rag.entities_vdb,
                        rag.relationships_vdb,
                        rag.text_chunks,
                        query_param,
                        global_config,
                        hashing_kv=rag.llm_response_cache,  # Cache is fine - we generate our own answer from filtered context
                        chunks_vdb=rag.chunks_vdb,
                    )
                    
                    timings['kg_query_retrieval'] = time.time() - t_retrieval_start
                    logger.info(f"[EVIDENCE_QUERY] kg_query completed in {timings['kg_query_retrieval']:.2f}s. Got context without answer generation.")
                    logger.info(f"[EVIDENCE_QUERY] DEBUG: kg_result.content length: {len(kg_result.content) if kg_result.content else 0} chars")
                    logger.info(f"[EVIDENCE_QUERY] DEBUG: kg_result.content preview: {kg_result.content[:100] if kg_result.content else 'None'}...")
                    
                    # Extract keywords from metadata
                    metadata = kg_result.metadata
                    if metadata and "keywords" in metadata:
                        kw = metadata["keywords"]
                        extracted_keywords_dict = {
                            "high_level": kw.get("high_level", []),
                            "low_level": kw.get("low_level", [])
                        }
                        logger.info(f"[EVIDENCE_QUERY] Extracted keywords: {extracted_keywords_dict}")
                    
                    # Get the ACTUAL relationships that kg_query used (from raw_data)
                    logger.info("[EVIDENCE_QUERY] Filtering relationships from kg_query by modality")
                    t_filter_start = time.time()
                    
                    raw_data = kg_result.raw_data or {}
                    data_section = raw_data.get("data", {})
                    kg_relationships = data_section.get("relationships", [])
                    logger.info(f"[EVIDENCE_QUERY] kg_query used {len(kg_relationships)} relationships")
                    
                    # Now filter those relationships by modality
                    # Need to get modality info from graph edges
                    for kg_rel in kg_relationships:
                        try:
                            # Extract src/tgt from kg_query relationship format
                            src_id = kg_rel.get("src_id", "")
                            tgt_id = kg_rel.get("tgt_id", "")
                            keywords = kg_rel.get("keywords", "")
                            
                            if not src_id or not tgt_id:
                                continue
                            
                            # Get the edge from graph to check modalities
                            edge = await rag.chunk_entity_relation_graph.get_edge(src_id, tgt_id)
                            if not edge:
                                continue
                            
                            # Check modality filtering
                            edge_modalities_str = edge.get("modality", "")
                            if not edge_modalities_str:
                                continue
                            
                            edge_modalities = set(m.strip().lower() for m in edge_modalities_str.split(",") if m.strip())
                            
                            # Apply modality operator logic
                            matches = False
                            if request.modality_operator == LogicalOperator.AND:
                                matches = modality_set.issubset(edge_modalities)
                            elif request.modality_operator == LogicalOperator.OR:
                                matches = bool(modality_set.intersection(edge_modalities))
                            elif request.modality_operator == LogicalOperator.ONLY:
                                matches = modality_set == edge_modalities
                            
                            if not matches:
                                continue
                            
                            # Check minimum evidence count
                            evidence_count = edge.get("evidence_total_count", 0)
                            if evidence_count < request.min_evidence_count:
                                continue
                            
                            # Build relationship object (includes details from kg_query)
                            evidence_details = None
                            if request.include_evidence_details:
                                evidence_summary_json = edge.get("evidence_summary_json")
                                if evidence_summary_json:
                                    try:
                                        evidence_summary = json.loads(evidence_summary_json)
                                        # Create evidence details with chunk_ids
                                        evidence_details = []
                                        chunk_ids_to_fetch = []
                                        for src in evidence_summary.get("sources", []):
                                            chunk_id = src.get("chunk_id", "")
                                            evidence_details.append(EvidenceDetail(
                                                chunk_id=chunk_id,
                                                file_path=src.get("file_path"),
                                                modality=src.get("modality", "unknown"),
                                                doc_id=src.get("doc_id"),
                                                timestamp=src.get("timestamp"),
                                                content=None  # Will be populated below
                                            ))
                                            if chunk_id:
                                                chunk_ids_to_fetch.append(chunk_id)
                                        
                                        # Fetch chunk content for all evidence sources
                                        if chunk_ids_to_fetch:
                                            try:
                                                chunk_data_list = await rag.text_chunks.get_by_ids(chunk_ids_to_fetch)
                                                chunk_content_map = {
                                                    cid: cdata.get("content") if cdata else None
                                                    for cid, cdata in zip(chunk_ids_to_fetch, chunk_data_list)
                                                }
                                                # Populate content in evidence_details
                                                for evidence in evidence_details:
                                                    if evidence.chunk_id in chunk_content_map:
                                                        evidence.content = chunk_content_map[evidence.chunk_id]
                                            except Exception as e:
                                                logger.debug(f"Could not fetch chunk content: {e}")
                                    except json.JSONDecodeError:
                                        pass
                            
                            relationship = RelationshipWithEvidence(
                                src_entity=src_id,
                                tgt_entity=tgt_id,
                                relation=keywords,
                                content=kg_rel.get("content"),
                                fact_key=edge.get("fact_key"),
                                evidence_total_count=evidence_count,
                                evidence_doc_count=edge.get("evidence_doc_count", 0),
                                modalities=list(edge_modalities),
                                evidence_details=evidence_details,
                            )
                            relationships_with_evidence.append(relationship)
                            
                        except Exception as e:
                            logger.debug(f"[EVIDENCE_QUERY] Error processing relationship: {e}")
                            continue
                    
                    logger.info(
                        f"[EVIDENCE_QUERY] LLM mode: {len(relationships_with_evidence)} relationships "
                        f"after modality filtering (from {len(kg_relationships)} kg_query relationships)"
                    )
                    
                    # Sort based on user preference
                    if request.sort_by == "evidence_strength":
                        relationships_with_evidence.sort(
                            key=lambda r: r.evidence_total_count,
                            reverse=True
                        )
                        logger.info(f"[EVIDENCE_QUERY] Sorted by evidence strength (support count)")
                    else:
                        # Preserve kg_query's relevance ranking (already sorted by similarity)
                        logger.info(f"[EVIDENCE_QUERY] Preserving relevance order from kg_query (similarity-based)")
                    
                    # Limit to top_k
                    relationships_with_evidence = relationships_with_evidence[:request.top_k]
                    logger.info(f"[EVIDENCE_QUERY] Limited to top {len(relationships_with_evidence)} relationships (top_k={request.top_k})")
                    
                    timings['modality_filtering_and_sorting'] = time.time() - t_filter_start
                    
                    # Generate answer from filtered relationships (if requested)
                    logger.info(f"[EVIDENCE_QUERY] Checking answer generation: generate_answer={request.generate_answer}, filtered_relations={len(relationships_with_evidence)}")
                    if request.generate_answer and relationships_with_evidence:
                        logger.info(f"[EVIDENCE_QUERY] ===== Generating answer from {len(relationships_with_evidence)} modality-filtered relationships =====")
                        try:
                            # Step 1: Extract entity IDs from filtered relationships
                            entity_ids_in_filtered = set()
                            fact_keys = set()
                            for rel in relationships_with_evidence:
                                entity_ids_in_filtered.add(rel.src_entity.lower())
                                entity_ids_in_filtered.add(rel.tgt_entity.lower())
                                if rel.fact_key:
                                    fact_keys.add(rel.fact_key)
                            
                            logger.info(f"[EVIDENCE_QUERY] Extracted {len(entity_ids_in_filtered)} unique entities from filtered relationships")
                            
                            # Step 2: Filter entities to only those in filtered relationships
                            kg_entities_all = data_section.get("entities", [])
                            kg_entities = [
                                e for e in kg_entities_all
                                if e.get("entity_name", "").lower() in entity_ids_in_filtered
                            ]
                            logger.info(f"[EVIDENCE_QUERY] Filtered entities: {len(kg_entities)} (from {len(kg_entities_all)} total)")
                            
                            # Step 3: Get modality-specific chunks from relationships' evidence
                            # Extract chunk_ids and count how many relationships reference each chunk
                            chunk_reference_count = {}
                            
                            for rel in relationships_with_evidence:
                                if rel.evidence_details:
                                    for evidence in rel.evidence_details:
                                        # evidence.modality already matches our filter (from relationship filtering)
                                        if evidence.chunk_id:
                                            chunk_reference_count[evidence.chunk_id] = chunk_reference_count.get(evidence.chunk_id, 0) + 1
                            
                            logger.info(f"[EVIDENCE_QUERY] Extracted {len(chunk_reference_count)} unique chunk_ids from relationship evidence")
                            
                            # Step 4: Retrieve chunk content from text_chunks storage
                            kg_chunks = []
                            if chunk_reference_count:
                                try:
                                    # Use rag.text_chunks directly (already initialized)
                                    chunk_ids_list = list(chunk_reference_count.keys())
                                    chunk_data_list = await rag.text_chunks.get_by_ids(chunk_ids_list)
                                    
                                    # Convert to the format expected by prompt builder, with reference count
                                    for chunk_id, chunk_data in zip(chunk_ids_list, chunk_data_list):
                                        if chunk_data and "content" in chunk_data:
                                            kg_chunks.append({
                                                "chunk_id": chunk_id,
                                                "content": chunk_data["content"],
                                                "tokens": chunk_data.get("tokens", 0),
                                                "full_doc_id": chunk_data.get("full_doc_id", chunk_id),
                                                "reference_count": chunk_reference_count[chunk_id],  # How many relationships reference this chunk
                                            })
                                    
                                    # Sort chunks by reference count (most referenced first)
                                    # These are the chunks most important to the answer
                                    kg_chunks.sort(key=lambda c: c["reference_count"], reverse=True)
                                    
                                    logger.info(f"[EVIDENCE_QUERY] Retrieved {len(kg_chunks)} modality-specific chunks from storage (sorted by relevance)")
                                
                                except Exception as e:
                                    logger.warning(f"[EVIDENCE_QUERY] Could not retrieve chunks: {e}")
                                    kg_chunks = []
                            
                            if not kg_chunks:
                                logger.info("[EVIDENCE_QUERY] No modality-specific chunks found, answer will be based on entities and relationships only")
                            
                            # Build context string from filtered relationships + entities + chunks
                            # Use same format as kg_query
                            from lightrag.prompt import PROMPTS
                            
                            entities_str = "\n".join(
                                json.dumps({"entity": e.get("entity_name", ""), "type": e.get("entity_type", ""), "description": e.get("description", "")}, ensure_ascii=False)
                                for e in kg_entities
                            )
                            
                            relations_str = "\n".join(
                                json.dumps({"src_id": r.src_entity, "tgt_id": r.tgt_entity, "keywords": r.relation, "description": r.content or ""}, ensure_ascii=False)
                                for r in relationships_with_evidence
                            )
                            
                            chunks_str = "\n".join(
                                json.dumps({"content": c.get("content", "")}, ensure_ascii=False)
                                for c in kg_chunks
                            )
                            
                            # Build reference list
                            references = data_section.get("references", [])
                            reference_list_str = "\n".join(
                                f"[{ref['reference_id']}] {ref['file_path']}"
                                for ref in references
                                if ref.get("reference_id")
                            )
                            
                            # Build knowledge graph context using same template
                            kg_context = PROMPTS["kg_query_context"].format(
                                entities_str=entities_str,
                                relations_str=relations_str,
                                text_chunks_str=chunks_str,
                                reference_list_str=reference_list_str,
                            )
                            
                            # Build system prompt using same template
                            sys_prompt = PROMPTS["rag_response"].format(
                                response_type=request.response_type,
                                user_prompt="n/a",
                                context_data=kg_context,
                            )
                            
                            logger.info(f"[EVIDENCE_QUERY] Calling LLM with {len(kg_entities)} filtered entities, {len(relationships_with_evidence)} filtered relations, {len(kg_chunks)} modality-specific chunks")
                            
                            # Call LLM with filtered context
                            t_llm_start = time.time()
                            response = await rag.llm_model_func(
                                request.query,
                                system_prompt=sys_prompt,
                            )
                            timings['llm_answer_generation'] = time.time() - t_llm_start
                            
                            generated_answer = response if isinstance(response, str) else str(response)
                            logger.info(f"[EVIDENCE_QUERY] Answer generated in {timings['llm_answer_generation']:.2f}s, length: {len(generated_answer)} chars")
                            
                        except Exception as e:
                            logger.error(f"[EVIDENCE_QUERY] Answer generation failed: {e}", exc_info=True)
                            generated_answer = None
                    elif request.generate_answer and not relationships_with_evidence:
                        generated_answer = f"No information found for the query in the selected modalities: {', '.join(m.value for m in request.modalities)}."
                        logger.info("[EVIDENCE_QUERY] No relationships found after filtering - returning no-info message")
                    
                except Exception as e:
                    logger.error(f"[EVIDENCE_QUERY] kg_query failed: {e}", exc_info=True)
                    # Fall back to simple filtering if kg_query fails
                    relationships_with_evidence = []
                
                # LLM mode complete - return with answer generated from filtered relationships
                total_time = time.time() - start_time
                
                return ModalityQueryResponse(
                    query=request.query,
                    search_mode=request.search_mode.value,
                    modality_filter=[m.value for m in request.modalities],
                    modality_operator=request.modality_operator.value,
                    sort_by=request.sort_by,
                    total_relationships_found=len(relationships_with_evidence),
                    relationships=relationships_with_evidence,
                    extracted_keywords=extracted_keywords_dict,
                    generated_answer=generated_answer,
                    total_time_seconds=total_time,
                    timing_breakdown=timings,
                )
            
            # ========== KEYWORD MODE: Fast direct filtering ==========
            else:
                logger.info("[EVIDENCE_QUERY] ===== KEYWORD MODE: Fast direct filtering =====")
                t_keyword_start = time.time()
                
                query_terms = [term.lower().strip() for term in request.query.split() if term.strip()]
                relationships_with_evidence = []
                
                # KEYWORD MODE: Get all edges and filter
                try:
                    all_edges = await rag.chunk_entity_relation_graph.get_all_edges()
                    logger.info(f"[EVIDENCE_QUERY] Retrieved {len(all_edges)} total edges from graph")
                except Exception as e:
                    logger.error(f"[EVIDENCE_QUERY] Failed to retrieve edges: {e}")
                    all_edges = []
            
            # Filter edges based on modality criteria
            modality_set = set(m.value for m in request.modalities)
            
            for edge in all_edges:
                try:
                    # Extract source and target from edge dict
                    src_id = edge.get("source", "")
                    tgt_id = edge.get("target", "")
                    
                    if not src_id or not tgt_id:
                        continue
                    
                    # Extract modality information from edge
                    edge_modalities_str = edge.get("modality", "")
                    if not edge_modalities_str:
                        continue
                    
                    # Parse modalities (comma-separated string)
                    edge_modalities = set(m.strip().lower() for m in edge_modalities_str.split(",") if m.strip())
                    
                    # Apply modality filter based on operator
                    matches = False
                    if request.modality_operator == LogicalOperator.AND:
                        # All requested modalities must be present (can have additional ones)
                        matches = modality_set.issubset(edge_modalities)
                    elif request.modality_operator == LogicalOperator.OR:
                        # At least one requested modality must be present
                        matches = bool(modality_set.intersection(edge_modalities))
                    elif request.modality_operator == LogicalOperator.ONLY:
                        # Exactly the requested modalities (no more, no less)
                        matches = modality_set == edge_modalities
                    
                    if not matches:
                        continue
                    
                    # Check minimum evidence count
                    evidence_count = edge.get("evidence_total_count", 0)
                    if evidence_count < request.min_evidence_count:
                        continue
                    
                    # Apply query filtering (search in entities, relations, and content)
                    if query_terms:
                        # Combine searchable fields
                        searchable_text = " ".join([
                            src_id.lower(),
                            tgt_id.lower(),
                            edge.get("keywords", "").lower(),
                            edge.get("content", "").lower()
                        ])
                        
                        # Check if any query term matches
                        if not any(term in searchable_text for term in query_terms):
                            continue
                    
                    # Parse evidence details if requested
                    evidence_details = None
                    if request.include_evidence_details:
                        evidence_summary_json = edge.get("evidence_summary_json")
                        if evidence_summary_json:
                            try:
                                evidence_summary = json.loads(evidence_summary_json)
                                # Create evidence details with chunk_ids
                                evidence_details = []
                                chunk_ids_to_fetch = []
                                for src in evidence_summary.get("sources", []):
                                    chunk_id = src.get("chunk_id", "")
                                    evidence_details.append(EvidenceDetail(
                                        chunk_id=chunk_id,
                                        file_path=src.get("file_path"),
                                        modality=src.get("modality", "unknown"),
                                        doc_id=src.get("doc_id"),
                                        timestamp=src.get("timestamp"),
                                        content=None  # Will be populated below
                                    ))
                                    if chunk_id:
                                        chunk_ids_to_fetch.append(chunk_id)
                                
                                # Fetch chunk content for all evidence sources
                                if chunk_ids_to_fetch:
                                    try:
                                        chunk_data_list = await rag.text_chunks.get_by_ids(chunk_ids_to_fetch)
                                        chunk_content_map = {
                                            cid: cdata.get("content") if cdata else None
                                            for cid, cdata in zip(chunk_ids_to_fetch, chunk_data_list)
                                        }
                                        # Populate content in evidence_details
                                        for evidence in evidence_details:
                                            if evidence.chunk_id in chunk_content_map:
                                                evidence.content = chunk_content_map[evidence.chunk_id]
                                    except Exception as e:
                                        logger.debug(f"Could not fetch chunk content: {e}")
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse evidence_summary_json for edge {src_id}->{tgt_id}")
                    
                    relationship = RelationshipWithEvidence(
                        src_entity=src_id,
                        tgt_entity=tgt_id,
                        relation=edge.get("keywords", ""),
                        content=edge.get("content"),
                        fact_key=edge.get("fact_key"),
                        evidence_total_count=evidence_count,
                        evidence_doc_count=edge.get("evidence_doc_count", 0),
                        modalities=list(edge_modalities),
                        evidence_details=evidence_details,
                    )
                    
                    relationships_with_evidence.append(relationship)
                    
                except Exception as e:
                    logger.debug(f"[EVIDENCE_QUERY] Error processing edge: {e}")
                    continue
            
            # Sort based on user preference
            if request.sort_by == "evidence_strength":
                relationships_with_evidence.sort(
                    key=lambda r: r.evidence_total_count,
                    reverse=True
                )
                logger.info(f"[EVIDENCE_QUERY] Sorted by evidence strength (support count)")
            else:
                # Preserve match order (order relationships were found by text matching)
                logger.info(f"[EVIDENCE_QUERY] Preserving match order from keyword search")
            
            # Limit to top_k
            relationships_with_evidence = relationships_with_evidence[:request.top_k]
            
            logger.info(
                f"[EVIDENCE_QUERY] Found {len(relationships_with_evidence)} relationships "
                f"matching modality filter"
            )
            
            timings['keyword_filtering_and_sorting'] = time.time() - t_keyword_start
            
            # Step 3: Generate answer if requested
            generated_answer = None
            logger.info(f"[EVIDENCE_QUERY] Checking answer generation: generate_answer={request.generate_answer}, relationships_count={len(relationships_with_evidence)}")
            
            if request.generate_answer and relationships_with_evidence:
                logger.info("[EVIDENCE_QUERY] Starting answer generation from filtered evidence")
                try:
                    # Build context from filtered relationships
                    entities_context = []
                    relations_context = []
                    
                    for rel in relationships_with_evidence:
                        relations_context.append({
                            "src_id": rel.src_entity,
                            "tgt_id": rel.tgt_entity,
                            "keywords": rel.relation,
                            "content": rel.content or "",
                            "modalities": ", ".join(rel.modalities),
                            "evidence_count": rel.evidence_total_count
                        })
                        entities_context.append({"entity_name": rel.src_entity})
                        entities_context.append({"entity_name": rel.tgt_entity})
                    
                    # Remove duplicate entities
                    seen_entities = set()
                    unique_entities = []
                    for ent in entities_context:
                        if ent["entity_name"] not in seen_entities:
                            unique_entities.append(ent)
                            seen_entities.add(ent["entity_name"])
                    
                    # Use the question field if provided, otherwise use the original query
                    question_for_llm = request.question or request.query
                    
                    # Build prompt for answer generation
                    operator_str = f" {request.modality_operator.value} "
                    modalities_str = operator_str.join([m.value for m in request.modalities])
                    modality_info = f" (filtered by {modalities_str} modalities)"
                    
                    context_text = f"""Based on the following knowledge graph relationships{modality_info}:\n\n"""
                    context_text += "Relationships:\n"
                    for i, rel in enumerate(relations_context[:10], 1):  # Limit to top 10 for context
                        context_text += f"{i}. {rel['src_id']} -{rel['keywords']}-> {rel['tgt_id']}"
                        context_text += f" (Evidence: {rel['evidence_count']}, Modalities: {rel['modalities']})\n"
                        if rel['content']:
                            context_text += f"   Description: {rel['content'][:200]}...\n"
                    
                    prompt = f"{context_text}\n\nQuestion: {question_for_llm}\n\nAnswer:"
                    logger.info(f"[EVIDENCE_QUERY] Prompt length: {len(prompt)} chars")
                    
                    # Call LLM
                    llm_func = rag.llm_model_func
                    logger.info(f"[EVIDENCE_QUERY] Calling LLM function: {llm_func}")
                    t_llm_start = time.time()
                    generated_answer = await llm_func(prompt)
                    timings['llm_answer_generation'] = time.time() - t_llm_start
                    logger.info(f"[EVIDENCE_QUERY] Answer generated in {timings['llm_answer_generation']:.2f}s, length: {len(generated_answer) if generated_answer else 0} chars")
                    
                except Exception as e:
                    logger.error(f"[EVIDENCE_QUERY] Answer generation failed: {e}", exc_info=True)
                    generated_answer = None
            
            total_time = time.time() - start_time
            
            return ModalityQueryResponse(
                query=request.query,
                search_mode=request.search_mode.value,
                modality_filter=[m.value for m in request.modalities],
                modality_operator=request.modality_operator.value,
                sort_by=request.sort_by,
                total_relationships_found=len(relationships_with_evidence),
                relationships=relationships_with_evidence,
                extracted_keywords=extracted_keywords_dict,
                generated_answer=generated_answer,
                total_time_seconds=total_time,
                timing_breakdown=timings,
            )
            
        except Exception as e:
            logger.error(f"[EVIDENCE_QUERY] Query failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Evidence query failed: {str(e)}"
            )
    
    @router.get(
        "/modalities/stats",
        summary="Get modality statistics",
        description="""Get statistics about modality distribution in the knowledge graph.
        
        Returns counts and percentages of relationships by modality type.
        """,
    )
    async def get_modality_stats(
        auth: Any = Depends(get_combined_auth_dependency(api_key)),
    ) -> Dict[str, Any]:
        """Get statistics about evidence modalities in the knowledge graph."""
        try:
            all_edges = await rag.chunk_entity_relation_graph.get_all_edges()
            
            modality_counts = {}
            total_with_evidence = 0
            total_edges = len(all_edges)
            
            for edge in all_edges:
                modalities_str = edge.get("modality", "")
                if modalities_str:
                    total_with_evidence += 1
                    for modality in modalities_str.split(","):
                        modality = modality.strip().lower()
                        if modality:
                            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            
            # Calculate percentages
            modality_stats = {
                mod: {
                    "count": count,
                    "percentage": round(count / total_with_evidence * 100, 2) if total_with_evidence > 0 else 0
                }
                for mod, count in modality_counts.items()
            }
            
            return {
                "total_edges": total_edges,
                "edges_with_evidence": total_with_evidence,
                "edges_without_evidence": total_edges - total_with_evidence,
                "modality_distribution": modality_stats,
            }
            
        except Exception as e:
            logger.error(f"Failed to get modality stats: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get modality statistics: {str(e)}"
            )
    
    return router