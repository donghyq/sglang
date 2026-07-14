"""KV Cache status query interface for Dynamo Cache-Aware routing.

Dynamo's Cache-Aware router calls this to check whether the current
SGLang node has reusable KV Cache for a given retrieval payload.
The response is a simple yes/no with miss breakdown — no internal
storage details are exposed.

Fail-closed: any error -> has_cache=False, fallback_required=True.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from sglang.srt.mem_cache.retrieval_cache_adapter import (
    plan_from_payload,
    summarize_plan,
)
from sglang.srt.mem_cache.retrieval_cache_planner import (
    RetrievalConditionedKVPlanner,
)


@dataclass(frozen=True)
class KVCacheStatus:
    """Immutable status snapshot for a single retrieval-payload query.

    Attributes:
        has_cache: True if at least one chunk has a KV cache hit.
        hit_chunk_count: number of chunks with reusable KV cache.
        miss_chunk_count: number of chunks without a reusable KV cache.
        reusable_token_count: total tokens that can be reused from cache.
        fallback_required: True if the caller must fall back to full prefill.
        miss_breakdown: mapping of miss-type string -> count.
        plan_latency_ms: time spent computing the plan, in milliseconds.
    """

    has_cache: bool
    hit_chunk_count: int
    miss_chunk_count: int
    reusable_token_count: int
    fallback_required: bool
    miss_breakdown: Dict[str, int] = field(default_factory=dict)
    plan_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "has_cache": self.has_cache,
            "hit_chunk_count": self.hit_chunk_count,
            "miss_chunk_count": self.miss_chunk_count,
            "reusable_token_count": self.reusable_token_count,
            "fallback_required": self.fallback_required,
            "miss_breakdown": dict(self.miss_breakdown),
            "plan_latency_ms": round(self.plan_latency_ms, 6),
        }


class KVCacheStatusReporter:
    """Query interface wrapping a RetrievalConditionedKVPlanner.

    Dynamo's Cache-Aware router calls ``query()`` with a retrieval payload
    to check whether the current SGLang node has reusable KV Cache.

    The reporter is intentionally thin — it delegates all planning logic
    to ``RetrievalConditionedKVPlanner`` via the adapter layer, and only
    converts the plan into a user-facing status snapshot.

    Thread-safety: the planner uses plain dict lookups internally, which
    are safe for concurrent reads in CPython.  No additional locking is
    needed for the query path.
    """

    def __init__(self, planner: RetrievalConditionedKVPlanner) -> None:
        self._planner = planner

    def query(
        self, retrieval_payload: Optional[Dict[str, Any]]
    ) -> KVCacheStatus:
        """Check KV cache availability for a retrieval payload.

        Fail-closed: any exception or missing payload results in
        ``has_cache=False`` and ``fallback_required=True``.
        """
        try:
            plan = plan_from_payload(self._planner, retrieval_payload)
        except Exception:
            return self._fail_closed()

        if plan is None:
            return self._fail_closed()

        try:
            summary = summarize_plan(plan)
        except Exception:
            return self._fail_closed()

        has_cache = summary.hit_chunks > 0
        # If there is no reusable cache at all, the caller must fall back
        # to full prefill — regardless of whether individual chunks missed
        # or the query was simply empty.
        fallback_required = summary.fallback_required or not has_cache

        return KVCacheStatus(
            has_cache=has_cache,
            hit_chunk_count=summary.hit_chunks,
            miss_chunk_count=summary.miss_chunks,
            reusable_token_count=summary.reusable_token_count,
            fallback_required=fallback_required,
            miss_breakdown=dict(summary.miss_breakdown),
            plan_latency_ms=summary.plan_latency_ms,
        )

    @staticmethod
    def _fail_closed() -> KVCacheStatus:
        """Return a fail-closed status (no cache, fallback required)."""
        return KVCacheStatus(
            has_cache=False,
            hit_chunk_count=0,
            miss_chunk_count=0,
            reusable_token_count=0,
            fallback_required=True,
            miss_breakdown={},
            plan_latency_ms=0.0,
        )


__all__ = [
    "KVCacheStatus",
    "KVCacheStatusReporter",
]
