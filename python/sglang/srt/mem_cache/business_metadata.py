from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass
class BusinessMetadata:
    """Unified business attribute schema consumed by eviction policy.

    All upstream signals (request headers, router tags, retriever context,
    adapter hints) are normalized into this single structure before reaching
    the cache layer.  The eviction policy only reads this schema and never
    accepts raw "evict scores" from upstream.

    Core scoring fields:
        hot_bucket_score:           how hot this prefix's business bucket is
        time_window_score:          time-window relevance (e.g. lunch vs night)
        estimated_reload_cost:      cost to recompute this prefix if evicted
        estimated_reuse_prefix_len: expected reuse length (prefix tokens)
        business_complete:          if True, the business turn is finished;
                                    this node is a candidate for early eviction

    Optional classification fields (used for bucket-level metrics and bounded
    weight adjustments; do NOT directly override eviction score):
        biz_type:   business category, e.g. "search", "chat", "rag"
        priority:   integer priority hint (higher = more important)
        sla_class:  SLA tier: "best_effort", "standard", "premium"
        tenant:     tenant identifier for multi-tenant isolation
        trace_tag:  opaque tag for tracing / debugging
    """

    hot_bucket_score: float = 0.0
    time_window_score: float = 0.0
    estimated_reload_cost: float = 0.0
    estimated_reuse_prefix_len: float = 0.0
    business_complete: bool = False
    biz_type: str = "default"
    priority: int = 0
    sla_class: str = "standard"
    tenant: str = "default"
    trace_tag: str = ""


_SLA_WEIGHT_MULTIPLIER: Dict[str, float] = {
    "best_effort": 0.5,
    "standard": 1.0,
    "premium": 1.5,
}


def _sla_multiplier(sla_class: str) -> float:
    return _SLA_WEIGHT_MULTIPLIER.get(sla_class, 1.0)


class BusinessMetadataBuilder:
    """Normalizes heterogeneous upstream business context into BusinessMetadata.

    This is the single formal entry point for metadata injection.  Upstream
    callers (request adapter, router, retriever) pass a dict-like context;
    the builder maps it to the stable schema.  Missing fields degrade
    gracefully to defaults so that LRU/SLRU behaviour is preserved when no
    business information is available.

    Usage::

        builder = BusinessMetadataBuilder()
        metadata = builder.build({
            "biz_type": "rag",
            "sla_class": "premium",
            "hot_bucket_score": 3.0,
            "reload_cost": 5.0,
        })
        cache.set_business_metadata(node, **asdict(metadata))
    """

    _ALIASES: Dict[str, str] = {
        "reload_cost": "estimated_reload_cost",
        "reuse_len": "estimated_reuse_prefix_len",
        "reuse_prefix_len": "estimated_reuse_prefix_len",
        "complete": "business_complete",
        "done": "business_complete",
        "type": "biz_type",
        "biz": "biz_type",
    }

    _VALID_FIELDS = frozenset(
        {
            "hot_bucket_score",
            "time_window_score",
            "estimated_reload_cost",
            "estimated_reuse_prefix_len",
            "business_complete",
            "biz_type",
            "priority",
            "sla_class",
            "tenant",
            "trace_tag",
        }
    )

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _coerce_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _coerce_bool(self, value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            return default
        if value is None:
            return default
        return bool(value)

    def build(self, context: Mapping[str, Any]) -> BusinessMetadata:
        normalized: Dict[str, Any] = {}
        for raw_key, value in context.items():
            key = self._ALIASES.get(raw_key, raw_key)
            if key in self._VALID_FIELDS:
                normalized[key] = value

        if "hot_bucket_score" in normalized:
            normalized["hot_bucket_score"] = self._coerce_float(
                normalized["hot_bucket_score"]
            )
        if "time_window_score" in normalized:
            normalized["time_window_score"] = self._coerce_float(
                normalized["time_window_score"]
            )
        if "estimated_reload_cost" in normalized:
            normalized["estimated_reload_cost"] = self._coerce_float(
                normalized["estimated_reload_cost"]
            )
        if "estimated_reuse_prefix_len" in normalized:
            normalized["estimated_reuse_prefix_len"] = self._coerce_float(
                normalized["estimated_reuse_prefix_len"]
            )
        if "business_complete" in normalized:
            normalized["business_complete"] = self._coerce_bool(
                normalized["business_complete"]
            )
        if "priority" in normalized:
            normalized["priority"] = self._coerce_int(normalized["priority"])
        if "biz_type" in normalized:
            normalized["biz_type"] = str(normalized["biz_type"])
        if "sla_class" in normalized:
            normalized["sla_class"] = str(normalized["sla_class"])
        if "tenant" in normalized:
            normalized["tenant"] = str(normalized["tenant"])
        if "trace_tag" in normalized:
            normalized["trace_tag"] = str(normalized["trace_tag"])

        return BusinessMetadata(**normalized)

    def sla_weight_multiplier(self, sla_class: str) -> float:
        return _sla_multiplier(sla_class)


class BusinessMetadataStore:
    """Thread-safe side metadata store for cache nodes.

    The design is intentionally simple:
    - metadata is keyed by node id
    - callers may optionally attach metadata after node creation
    - absent metadata falls back to zero-value scoring inputs
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._metadata_by_node_id: Dict[int, BusinessMetadata] = {}

    def set_for_node(
        self, node_id: int, metadata: BusinessMetadata
    ) -> BusinessMetadata:
        with self._lock:
            self._metadata_by_node_id[node_id] = metadata
        return metadata

    def get_for_node(self, node_id: int) -> Optional[BusinessMetadata]:
        with self._lock:
            return self._metadata_by_node_id.get(node_id)

    def pop_for_node(self, node_id: int) -> None:
        with self._lock:
            self._metadata_by_node_id.pop(node_id, None)

    def copy_for_node(self, source_node_id: int, target_node_id: int) -> None:
        with self._lock:
            metadata = self._metadata_by_node_id.get(source_node_id)
            if metadata is None:
                return
            self._metadata_by_node_id[target_node_id] = BusinessMetadata(
                hot_bucket_score=metadata.hot_bucket_score,
                time_window_score=metadata.time_window_score,
                estimated_reload_cost=metadata.estimated_reload_cost,
                estimated_reuse_prefix_len=metadata.estimated_reuse_prefix_len,
                business_complete=metadata.business_complete,
                biz_type=metadata.biz_type,
                priority=metadata.priority,
                sla_class=metadata.sla_class,
                tenant=metadata.tenant,
                trace_tag=metadata.trace_tag,
            )
