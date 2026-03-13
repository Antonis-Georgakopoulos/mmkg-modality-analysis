"""
Minimal Evidence Tracking for Knowledge Graph Triples

This file contains ONLY the essential code to:
1. Track evidence for each triple (src, relation, tgt)
2. Store evidence metadata (chunk_id, file_path, modality, doc_id)
3. Enrich graph edges with evidence attributes

Usage:
    from evidence_minimal import track_evidence_for_document
    
    # After processing a document with LightRAG:
    await track_evidence_for_document(lightrag, doc_id)
"""

import time
import json
import os
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from lightrag.utils import compute_mdhash_id, logger
from lightrag.constants import GRAPH_FIELD_SEP


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class EvidenceItem:
    """Single piece of evidence for a triple."""
    chunk_id: str
    file_path: Optional[str] = None
    modality: Optional[str] = None  # 'text', 'image', 'table', 'equation'
    doc_id: Optional[str] = None
    timestamp: int = int(time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "modality": self.modality,
            "doc_id": self.doc_id,
            "timestamp": self.timestamp,
        }


def _triple_key(src_id: str, tgt_id: str, relation_keywords: Optional[str]) -> str:
    """Create unique key for a triple."""
    key = f"{src_id}|{relation_keywords or ''}|{tgt_id}"
    return compute_mdhash_id(key, prefix="tri-")


def _fact_key(src_id: str, tgt_id: str, relation_keywords: Optional[str]) -> str:
    """Create fact identifier for a relationship."""
    key = f"{src_id}|{relation_keywords or ''}|{tgt_id}"
    return compute_mdhash_id(key, prefix="fact-")


# ============================================================================
# EVIDENCE TRACKER
# ============================================================================

class EvidenceTracker:
    """Tracks and stores evidence for triples."""
    
    def __init__(self, storage):
        self._storage = storage
    
    @classmethod
    async def create_for_lightrag(cls, lightrag: Any) -> "EvidenceTracker":
        """Create tracker using LightRAG's storage."""
        storage = lightrag.key_string_value_json_storage_cls(
            namespace="triple_evidence",
            workspace=lightrag.workspace,
            global_config=lightrag.__dict__,
            embedding_func=None,
        )
        await storage.initialize()
        return cls(storage)
    
    async def record(
        self,
        src_id: str,
        tgt_id: str,
        chunk_id: str,
        file_path: Optional[str],
        modality: Optional[str],
        doc_id: Optional[str],
        relation_keywords: Optional[str] = None,
    ) -> None:
        """Record evidence for a triple. Deduplicates by chunk_id."""
        key = _triple_key(src_id, tgt_id, relation_keywords)
        existing = await self._storage.get_by_id(key) or {}
        
        evidences: List[Dict[str, Any]] = existing.get("evidences", [])
        seen_chunks = {e.get("chunk_id") for e in evidences}
        seen_docs = {e.get("doc_id") for e in evidences if e.get("doc_id")}
        
        # Only add if chunk not already recorded
        if chunk_id not in seen_chunks:
            evidences.append(
                EvidenceItem(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    modality=modality,
                    doc_id=doc_id,
                ).to_dict()
            )
        
        if doc_id:
            seen_docs.add(doc_id)
        
        # Store aggregated evidence
        await self._storage.upsert({
            key: {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "relation_keywords": relation_keywords,
                "evidence_count": len({e["chunk_id"] for e in evidences}),
                "doc_count": len(seen_docs),
                "evidences": evidences,
                "updated_at": int(time.time()),
                "_id": key,
            }
        })
        
        await self._storage.index_done_callback()
    
    async def get(
        self, 
        src_id: str, 
        tgt_id: str, 
        relation_keywords: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch evidence record for a triple."""
        key = _triple_key(src_id, tgt_id, relation_keywords)
        return await self._storage.get_by_id(key)


# ============================================================================
# EVIDENCE SUMMARY BUILDER
# ============================================================================

def build_evidence_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build structured summary from evidence record."""
    if not record:
        return {
            "total_count": 0,
            "doc_count": 0,
            "by_modality": {},
            "modality": "",
            "sources": [],
        }
    
    evidences = record.get("evidences") or []
    
    # Count by modality
    modality_counts: Dict[str, int] = {}
    for ev in evidences:
        mod = (ev.get("modality") or "unknown").lower()
        modality_counts[mod] = modality_counts.get(mod, 0) + 1
    
    modality_csv = ",".join(sorted(modality_counts.keys()))
    
    return {
        "total_count": len(evidences),
        "doc_count": record.get("doc_count", 0),
        "by_modality": modality_counts,
        "modality": modality_csv,
        "sources": [
            {
                "chunk_id": e.get("chunk_id"),
                "file_path": e.get("file_path"),
                "modality": e.get("modality"),
                "doc_id": e.get("doc_id"),
                "timestamp": e.get("timestamp"),
            }
            for e in evidences
        ],
    }


# ============================================================================
# MODALITY INFERENCE
# ============================================================================

async def infer_modality_from_chunk(lightrag: Any, chunk_id: str, debug_log: bool = False) -> str:
    """Infer modality type from chunk content."""
    try:
        chunk = await lightrag.text_chunks.get_by_id(chunk_id)
        if not chunk:
            if debug_log:
                logger.info(f"[DEBUG CHUNK] chunk_id={chunk_id} -> NOT FOUND, defaulting to 'text'")
            return "text"
        
        # DEBUG: Log ALL chunk properties
        if debug_log:
            logger.info(f"\n{'='*80}")
            logger.info(f"[DEBUG CHUNK] chunk_id: {chunk_id}")
            logger.info(f"[DEBUG CHUNK] All keys in chunk: {list(chunk.keys())}")
            for key, value in chunk.items():
                if key == "content":
                    # Truncate content for readability
                    content_preview = str(value)[:300] if value else "<empty>"
                    logger.info(f"[DEBUG CHUNK]   {key}: {content_preview}...")
                else:
                    logger.info(f"[DEBUG CHUNK]   {key}: {value}")
            logger.info(f"{'='*80}")
        
        # Check multimodal flag
        if chunk.get("is_multimodal"):
            modality = chunk.get("original_type", "multimodal")
            if debug_log:
                logger.info(f"[DEBUG CHUNK] -> Detected via is_multimodal=True, original_type={modality}")
            return modality
        
        # Heuristic based on content
        content_text = (chunk.get("content") or "").lower()
        if any(tag in content_text for tag in ["image content analysis:", "image path:"]):
            if debug_log:
                logger.info("[DEBUG CHUNK] -> Detected 'image' via content heuristic")
            return "image"
        elif any(tag in content_text for tag in ["table analysis:", "structure:"]):
            if debug_log:
                logger.info("[DEBUG CHUNK] -> Detected 'table' via content heuristic")
            return "table"
        elif any(tag in content_text for tag in ["mathematical equation analysis:", "equation:"]):
            if debug_log:
                logger.info("[DEBUG CHUNK] -> Detected 'equation' via content heuristic")
            return "equation"
        else:
            if debug_log:
                logger.info("[DEBUG CHUNK] -> Defaulting to 'text' (no multimodal markers found)")
            return "text"
    except Exception as e:
        if debug_log:
            logger.error(f"[DEBUG CHUNK] chunk_id={chunk_id} -> EXCEPTION: {e}")
        return "text"


async def batch_infer_modalities(lightrag: Any, chunk_ids: List[str], debug_first_n: int = 20) -> Dict[str, str]:
    """Batch infer modalities for multiple chunks concurrently.
    
    Args:
        lightrag: LightRAG instance
        chunk_ids: List of chunk IDs to process
        debug_first_n: Number of chunks to log in detail (default 5, set to 0 to disable)
    """
    chunks_logged = 0
    
    async def _infer(cid: str, should_debug: bool) -> tuple[str, str]:
        modality = await infer_modality_from_chunk(lightrag, cid, debug_log=should_debug)
        return (cid, modality)
    
    # Process in batches of 50 to avoid overwhelming the system
    batch_size = 50
    results = {}
    
    for i in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[i:i + batch_size]
        # Debug first N chunks
        batch_results = await asyncio.gather(*[
            _infer(cid, chunks_logged + idx < debug_first_n) 
            for idx, cid in enumerate(batch)
        ])
        for cid, modality in batch_results:
            results[cid] = modality
        chunks_logged += len(batch)
    
    return results


# ============================================================================
# MAIN EVIDENCE TRACKING FUNCTION
# ============================================================================

async def track_evidence_for_document(
    lightrag: Any, 
    doc_id: str,
    ensure_flushed: bool = True
) -> int:
    """
    Track and store evidence for all relationships in a document.
    
    OPTIMIZED VERSION with:
    - Batch modality lookups
    - Batch evidence recording
    - Batch graph/VDB updates
    - Progress logging for large documents
    
    Args:
        lightrag: LightRAG instance
        doc_id: Document ID to process
        ensure_flushed: Flush storages before reading
        
    Returns:
        Number of relationships processed
    """
    if not doc_id:
        return 0
    
    # Flush storages if requested
    if ensure_flushed:
        try:
            await lightrag.doc_status.index_done_callback()
            await lightrag.relationships_vdb.index_done_callback()
        except Exception as e:
            logger.debug(f"Flush failed: {e}")
    
    # Get chunk IDs for this document
    doc_status = await lightrag.doc_status.get_by_id(doc_id)
    if not doc_status:
        logger.debug(f"No doc_status found for {doc_id}")
        return 0
    
    chunk_ids = set(doc_status.get("chunks_list", []) or [])
    if not chunk_ids:
        logger.debug(f"No chunks found for {doc_id}")
        return 0
    
    # Read relationships from JSON file with retry logic
    relationships_path = os.path.join(lightrag.working_dir, "vdb_relationships.json")
    if not os.path.exists(relationships_path):
        logger.warning(f"Relationships file not found: {relationships_path}")
        return 0
    
    items: List[Dict[str, Any]] = []
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(relationships_path, "r", encoding="utf-8") as f:
                rel_json = json.load(f)
            
            # Parse relationships (support multiple JSON formats)
            if isinstance(rel_json, dict):
                maybe = rel_json.get("data")
                if isinstance(maybe, list):
                    items = maybe
                elif isinstance(maybe, dict):
                    items = list(maybe.values())
                else:
                    # Fallback: dict contains records keyed by id
                    items = list(rel_json.values()) if rel_json else []
            elif isinstance(rel_json, list):
                items = rel_json
            else:
                items = []
            break  # Success
        except (json.JSONDecodeError, IOError) as e:
            if attempt < max_retries - 1:
                logger.debug(f"Retry {attempt + 1}/{max_retries} reading relationships: {e}")
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
            else:
                logger.error(f"Failed to read relationships after {max_retries} attempts: {e}")
                items = []
    
    logger.info(f"[EVIDENCE] Doc {doc_id}: {len(chunk_ids)} chunks, {len(items)} relation items")
    if not items:
        logger.warning(f"[EVIDENCE] No relation items found for doc {doc_id}!")
        return 0
    
    # OPTIMIZATION 1: Batch load all modalities upfront
    logger.info(f"[EVIDENCE] Batch loading modalities for {len(chunk_ids)} chunks...")
    modality_cache = await batch_infer_modalities(lightrag, list(chunk_ids))
    logger.info(f"[EVIDENCE] Modalities loaded: {dict(list({m: list(modality_cache.values()).count(m) for m in set(modality_cache.values())}.items())[:5])}")
    
    # Create evidence tracker
    tracker = await EvidenceTracker.create_for_lightrag(lightrag)
    
    # OPTIMIZATION 2: Collect all operations for batch processing
    evidence_records = []  # [(src_id, tgt_id, chunk_id, file_path, modality, keywords)]
    processed = 0
    
    # First pass: collect all evidence records
    for idx, item in enumerate(items):
        try:
            src_id = item.get("src_id")
            tgt_id = item.get("tgt_id")
            source_field = item.get("source_id")
            
            if not src_id or not tgt_id or not source_field:
                continue
            
            # Get chunk IDs (uses GRAPH_FIELD_SEP from LightRAG)
            if isinstance(source_field, str):
                chunk_id_list = [s for s in source_field.split(GRAPH_FIELD_SEP) if s]
            elif isinstance(source_field, list):
                chunk_id_list = [str(s) for s in source_field if s]
            else:
                chunk_id_list = [str(source_field)]
            
            # Only process chunks belonging to this document
            for chunk_id in chunk_id_list:
                if chunk_id not in chunk_ids:
                    continue
                
                modality = modality_cache.get(chunk_id, "text")
                file_path = item.get("file_path")
                keywords = item.get("keywords")
                
                evidence_records.append((
                    src_id, tgt_id, chunk_id, file_path, modality, keywords, item
                ))
                processed += 1
        
        except Exception as e:
            logger.debug(f"Failed to collect evidence record: {e}")
            continue
    
    if processed == 0:
        logger.warning(f"[EVIDENCE] ✗ No relationships to process for doc {doc_id}")
        return 0
    
    logger.info(f"[EVIDENCE] Recording evidence for {processed} relationships (batched)...")
    
    # OPTIMIZATION 3: Batch record all evidence
    record_tasks = []
    for src_id, tgt_id, chunk_id, file_path, modality, keywords, _ in evidence_records:
        record_tasks.append(
            tracker.record(
                src_id=src_id,
                tgt_id=tgt_id,
                chunk_id=chunk_id,
                file_path=file_path,
                modality=modality,
                doc_id=doc_id,
                relation_keywords=keywords,
            )
        )
    
    # Execute in batches to avoid overwhelming the system
    batch_size = 100
    for i in range(0, len(record_tasks), batch_size):
        batch = record_tasks[i:i + batch_size]
        await asyncio.gather(*batch)
        if len(record_tasks) > 200:  # Only log for large docs
            logger.info(f"[EVIDENCE] Progress: {min(i + batch_size, len(record_tasks))}/{len(record_tasks)} records saved")
    
    logger.info(f"[EVIDENCE] Building summaries and updating graph...")
    
    # OPTIMIZATION 4: Batch fetch aggregated evidence and prepare updates
    unique_triples = {}
    for src_id, tgt_id, chunk_id, file_path, modality, keywords, item in evidence_records:
        triple_key = (src_id, tgt_id, keywords)
        if triple_key not in unique_triples:
            unique_triples[triple_key] = (src_id, tgt_id, keywords, item, chunk_id, file_path)
    
    # Fetch all aggregated evidence concurrently
    agg_tasks = [
        tracker.get(src_id, tgt_id, keywords)
        for src_id, tgt_id, keywords, _, _, _ in unique_triples.values()
    ]
    aggregated_results = await asyncio.gather(*agg_tasks)
    
    # Build summaries
    summaries = {}
    for (src_id, tgt_id, keywords, _, _, _), agg in zip(unique_triples.values(), aggregated_results):
        fact = _fact_key(src_id, tgt_id, keywords)
        summary = build_evidence_summary(agg or {})
        summaries[(src_id, tgt_id, keywords)] = (fact, summary)
    
    # OPTIMIZATION 5: Batch update relationships VDB
    rel_updates = {}
    for src_id, tgt_id, keywords, item, chunk_id, file_path in unique_triples.values():
        fact, summary = summaries.get((src_id, tgt_id, keywords), ("", {}))
        relation_id = item.get("__id__") or item.get("_id")
        if relation_id:
            rel_updates[relation_id] = {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "keywords": keywords,
                "content": item.get("content", ""),
                "source_id": chunk_id,
                "file_path": file_path,
                "fact_key": fact,
                "evidence_summary": summary,
            }
    
    if rel_updates:
        await lightrag.relationships_vdb.upsert(rel_updates)
    
    # OPTIMIZATION 6: Batch update graph edges
    edge_fetch_tasks = [
        lightrag.chunk_entity_relation_graph.get_edge(src_id, tgt_id)
        for src_id, tgt_id, keywords, _, _, _ in unique_triples.values()
    ]
    edges = await asyncio.gather(*edge_fetch_tasks)
    
    edge_updates = []
    for (src_id, tgt_id, keywords, _, _, _), edge in zip(unique_triples.values(), edges):
        if edge:
            fact, summary = summaries.get((src_id, tgt_id, keywords), ("", {}))
            new_edge = dict(edge)
            new_edge["fact_key"] = fact
            new_edge["evidence_total_count"] = int(summary.get("total_count", 0))
            new_edge["evidence_doc_count"] = int(summary.get("doc_count", 0))
            new_edge["modality"] = summary.get("modality", "")
            new_edge["evidence_summary_json"] = json.dumps(summary, ensure_ascii=False)
            edge_updates.append((src_id, tgt_id, new_edge))
    
    # Update edges in batches
    for src_id, tgt_id, new_edge in edge_updates:
        await lightrag.chunk_entity_relation_graph.upsert_edge(src_id, tgt_id, new_edge)
    
    # Persist changes
    try:
        await lightrag.relationships_vdb.index_done_callback()
        await lightrag.chunk_entity_relation_graph.index_done_callback()
    except Exception as e:
        logger.warning(f"Failed to persist changes: {e}")
    
    logger.info(f"[EVIDENCE] ✓ Successfully tracked evidence for {processed} relationships in document {doc_id}")
    return processed


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_usage(lightrag, doc_id: str):
    """
    Example: Track evidence after processing a document.
    
    This should be called AFTER:
    - LightRAG has inserted text content
    - Multimodal content has been processed
    """
    # Track evidence for all relationships in the document
    count = await track_evidence_for_document(lightrag, doc_id)
    
    print(f"✓ Tracked evidence for {count} relationships")
    
    # Retrieve evidence for a specific triple
    tracker = await EvidenceTracker.create_for_lightrag(lightrag)
    evidence = await tracker.get(
        src_id="entity1",
        tgt_id="entity2",
        relation_keywords="related_to"
    )
    
    if evidence:
        summary = build_evidence_summary(evidence)
        print(f"Evidence summary: {summary}")
