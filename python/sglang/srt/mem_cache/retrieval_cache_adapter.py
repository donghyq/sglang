from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from sglang.srt.mem_cache.retrieval_cache_planner import (
    RetrievalCacheMissType,
    RetrievalCachePlan,
    RetrievalChunkRef,
    RetrievalConditionedKVPlanner,
    RetrievalRenderKey,
)


@dataclass(frozen=True)
class RetrievalCachePlanningContext:
    render_key: RetrievalRenderKey
    retrieved_chunks: List[RetrievalChunkRef]


@dataclass(frozen=True)
class RetrievalCachePlanSummary:
    total_chunks: int
    hit_chunks: int
    miss_chunks: int
    reusable_token_count: int
    fallback_required: bool
    plan_latency_ms: float
    miss_breakdown: Dict[str, int]


def build_planning_context(
    retrieval_cache_payload: Optional[Dict[str, Any]],
) -> Optional[RetrievalCachePlanningContext]:
    if not retrieval_cache_payload:
        return None

    chunks_payload = retrieval_cache_payload.get("chunks") or []
    retrieved_chunks = []
    for chunk_payload in chunks_payload:
        chunk_id = chunk_payload.get("id")
        if not chunk_id:
            raise ValueError("retrieval_cache chunk is missing required field: id")
        retrieved_chunks.append(
            RetrievalChunkRef(
                chunk_id=chunk_id,
                content_hash=chunk_payload.get("content_hash"),
            )
        )

    render_key = RetrievalRenderKey(
        namespace=retrieval_cache_payload.get("namespace"),
        model_name=retrieval_cache_payload.get("model_name"),
        model_fingerprint=retrieval_cache_payload.get("model_fingerprint"),
        tokenizer_rev=retrieval_cache_payload.get("tokenizer_rev"),
        tokenizer_fingerprint=retrieval_cache_payload.get("tokenizer_fingerprint"),
        template_rev=retrieval_cache_payload.get("template_rev"),
        render_rev=retrieval_cache_payload.get("render_rev"),
        special_token_config=retrieval_cache_payload.get("special_token_config"),
        schema_version=retrieval_cache_payload.get("schema_version"),
        order_sensitive=retrieval_cache_payload.get("order_sensitive", True),
    )
    return RetrievalCachePlanningContext(
        render_key=render_key,
        retrieved_chunks=retrieved_chunks,
    )


def summarize_plan(plan: RetrievalCachePlan) -> RetrievalCachePlanSummary:
    miss_breakdown: Dict[str, int] = {}
    for decision in plan.decisions:
        if decision.miss_type is None:
            continue
        miss_key = str(decision.miss_type.value)
        miss_breakdown[miss_key] = miss_breakdown.get(miss_key, 0) + 1

    return RetrievalCachePlanSummary(
        total_chunks=len(plan.decisions),
        hit_chunks=plan.hit_count,
        miss_chunks=plan.miss_count,
        reusable_token_count=plan.reusable_token_count,
        fallback_required=plan.fallback_required,
        plan_latency_ms=plan.plan_latency_ms,
        miss_breakdown=miss_breakdown,
    )


def plan_from_payload(
    planner: RetrievalConditionedKVPlanner,
    retrieval_cache_payload: Optional[Dict[str, Any]],
) -> Optional[RetrievalCachePlan]:
    try:
        context = build_planning_context(retrieval_cache_payload)
    except Exception:
        return None

    if context is None:
        return None
    try:
        return planner.plan(context.render_key, context.retrieved_chunks)
    except Exception:
        return None
