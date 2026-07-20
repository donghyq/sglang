"""KV Cache status query interface for Dynamo Cache-Aware routing.

Dynamo's Cache-Aware router calls this to check whether the current
SGLang node has reusable KV Cache for a given retrieval payload.
The response is a simple yes/no with miss breakdown — no internal
storage details are exposed.

Fail-closed: any error -> has_cache=False, fallback_required=True.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sglang.srt.mem_cache.retrieval_cache_adapter import (
    plan_from_payload,
    summarize_plan,
)
from sglang.srt.mem_cache.retrieval_cache_planner import (
    RetrievalConditionedKVPlanner,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache


@dataclass(frozen=True)
class KVCacheStatus:
    """Immutable status snapshot for a single retrieval-payload query.

    Attributes:
        has_cache: True only when the local Radix cache verifies a non-empty
            exact token prefix in the requested namespace.
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
    planner_candidate: bool = False
    physical_exact_hit: bool = False
    estimated_kv_bytes: int = 0
    cache_layer: Optional[str] = None
    namespace: Optional[str] = None
    identity_version: Optional[str] = None
    status_timestamp: float = 0.0
    locked: bool = False
    lease_expires_at: float = 0.0
    business_value: Dict[str, Any] = field(default_factory=dict)
    reject_reason: Optional[str] = None

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
            "planner_candidate": self.planner_candidate,
            "physical_exact_hit": self.physical_exact_hit,
            "estimated_kv_bytes": self.estimated_kv_bytes,
            "cache_layer": self.cache_layer,
            "namespace": self.namespace,
            "identity_version": self.identity_version,
            "status_timestamp": self.status_timestamp,
            "locked": self.locked,
            "lease_expires_at": self.lease_expires_at,
            "business_value": dict(self.business_value),
            "reject_reason": self.reject_reason,
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

    def __init__(
        self,
        planner: RetrievalConditionedKVPlanner,
        prefix_cache: Optional["RadixCache"] = None,
        metrics_collector: Optional[Any] = None,
    ) -> None:
        self._planner = planner
        self._prefix_cache = prefix_cache
        self._metrics_collector = metrics_collector or getattr(
            prefix_cache, "metrics_collector", None
        )

    def query(
        self,
        retrieval_payload: Optional[Dict[str, Any]],
        *,
        input_ids: Optional[List[int]] = None,
        extra_key: Optional[str] = None,
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

        planner_candidate = summary.hit_chunks > 0
        physical_exact_hit = False
        physical_tokens = 0
        node = None
        if self._prefix_cache is not None and input_ids:
            try:
                from sglang.srt.mem_cache.radix_cache import RadixKey

                physical_tokens, node = self._prefix_cache.probe_prefix(
                    RadixKey(input_ids, extra_key)
                )
                # A page-aligned or partial prefix is still physically reusable;
                # every reported token was compared exactly by RadixKey.match().
                physical_exact_hit = physical_tokens > 0
            except Exception:
                physical_exact_hit = False

        has_cache = physical_exact_hit
        # If there is no reusable cache at all, the caller must fall back
        # to full prefill — regardless of whether individual chunks missed
        # or the query was simply empty.
        fallback_required = not physical_exact_hit
        try:
            metadata = (
                self._prefix_cache.business_metadata_store.get_for_node(node.id)
                if node is not None and self._prefix_cache is not None
                else None
            )
        except Exception:
            metadata = None

        # Metrics are best effort and must never change status semantics.
        try:
            if self._metrics_collector is not None:
                self._metrics_collector.record_planner_physical_outcome(
                    planner_candidate=planner_candidate,
                    physical_exact_hit=physical_exact_hit,
                )
        except Exception:
            pass

        return KVCacheStatus(
            has_cache=has_cache,
            hit_chunk_count=summary.hit_chunks,
            miss_chunk_count=summary.miss_chunks,
            reusable_token_count=physical_tokens,
            fallback_required=fallback_required,
            miss_breakdown=dict(summary.miss_breakdown),
            plan_latency_ms=summary.plan_latency_ms,
            planner_candidate=planner_candidate,
            physical_exact_hit=physical_exact_hit,
            estimated_kv_bytes=metadata.kv_bytes if metadata else 0,
            cache_layer="device" if physical_exact_hit else None,
            namespace=extra_key if physical_exact_hit else None,
            identity_version=(
                "retrieval:v1"
                if metadata and metadata.retrieval_namespace.startswith("retrieval:v1:")
                else None
            ),
            status_timestamp=time.time(),
            locked=bool(node and node.lock_ref > 0),
            lease_expires_at=metadata.lease_expires_at if metadata else 0.0,
            business_value=(
                {
                    "content_type": metadata.content_type,
                    "lifecycle_state": metadata.lifecycle_state,
                    "recompute_cost": metadata.recompute_cost,
                }
                if metadata
                else {}
            ),
            reject_reason=(
                None
                if physical_exact_hit
                else (
                    "physical_prefix_not_verified"
                    if planner_candidate
                    else "planner_miss"
                )
            ),
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
            status_timestamp=time.time(),
            reject_reason="invalid_or_missing_payload",
        )


__all__ = [
    "KVCacheStatus",
    "KVCacheStatusReporter",
]
