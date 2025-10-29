"""Evidence-based modality query routes for LightRAG API.

This module provides advanced query capabilities that filter results based on
evidence modalities (text, image, table, mathematical formulas) stored in the
knowledge graph relationships.
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional, Set
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from lightrag.api.utils_api import get_combined_auth_dependency

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
    AND = "AND"
    OR = "OR"


class ModalityQueryRequest(BaseModel):
    """Request model for evidence-based modality queries."""
    
    query: str = Field(
        min_length=3,
        description="The search query text",
        examples=["What is the relationship between the dog and duck?"]
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
    
    mode: Literal["local", "global", "hybrid", "naive"] = Field(
        default="hybrid",
        description="Query mode for graph traversal"
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
        description="Maximum number of results to return"
    )


class EvidenceDetail(BaseModel):
    """Detailed evidence information for a relationship."""
    chunk_id: str
    file_path: Optional[str] = None
    modality: str
    doc_id: Optional[str] = None
    timestamp: Optional[int] = None


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
    modality_filter: List[str]
    modality_operator: str
    total_relationships_found: int
    relationships: List[RelationshipWithEvidence]
    answer: Optional[str] = Field(None, description="Generated answer based on filtered evidence")


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
        - Combine modality filters with AND/OR logic
        - Retrieve detailed evidence metadata
        - Get answers based only on evidence from specific modalities
        
        Example use cases:
        - Find relationships supported by images AND tables
        - Get all text-based evidence for a query
        - Search only mathematical formulas and equations
        """,
    )
    async def query_with_modality_filter(
        request: ModalityQueryRequest,
        auth: Any = Depends(get_combined_auth_dependency(api_key)),
    ) -> ModalityQueryResponse:
        """Query knowledge graph with modality-based evidence filtering."""
        try:
            logger.info(
                f"[EVIDENCE_QUERY] Query: '{request.query}' | Modalities: {request.modalities} | "
                f"Operator: {request.modality_operator}"
            )
            
            # Get all relationships from the graph
            relationships_with_evidence = []
            
            # Access the chunk_entity_relation_graph to get edges with evidence
            try:
                all_edges = await rag.chunk_entity_relation_graph.get_all_edges()
                logger.info(f"[EVIDENCE_QUERY] Retrieved {len(all_edges)} total edges from graph")
            except Exception as e:
                logger.error(f"[EVIDENCE_QUERY] Failed to retrieve edges: {e}")
                all_edges = []
            
            # Filter edges based on modality criteria
            modality_set = set(m.value for m in request.modalities)
            
            # Prepare query terms for filtering (case-insensitive)
            query_terms = [term.lower().strip() for term in request.query.split() if term.strip()]
            
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
                        # All requested modalities must be present
                        matches = modality_set.issubset(edge_modalities)
                    else:  # OR
                        # At least one requested modality must be present
                        matches = bool(modality_set.intersection(edge_modalities))
                    
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
                                evidence_details = [
                                    EvidenceDetail(
                                        chunk_id=src.get("chunk_id", ""),
                                        file_path=src.get("file_path"),
                                        modality=src.get("modality", "unknown"),
                                        doc_id=src.get("doc_id"),
                                        timestamp=src.get("timestamp"),
                                    )
                                    for src in evidence_summary.get("sources", [])
                                ]
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
            
            # Sort by evidence count (descending) and limit to top_k
            relationships_with_evidence.sort(
                key=lambda r: r.evidence_total_count,
                reverse=True
            )
            relationships_with_evidence = relationships_with_evidence[:request.top_k]
            
            logger.info(
                f"[EVIDENCE_QUERY] Found {len(relationships_with_evidence)} relationships "
                f"matching modality filter"
            )
            
            return ModalityQueryResponse(
                query=request.query,
                modality_filter=[m.value for m in request.modalities],
                modality_operator=request.modality_operator.value,
                total_relationships_found=len(relationships_with_evidence),
                relationships=relationships_with_evidence,
                answer=None,  # Can be extended to generate answers
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