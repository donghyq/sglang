from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional

_METRIC_WORKFLOW_STAGES = frozenset(
    {
        "unknown",
        "retrieval",
        "planning",
        "tool_call",
        "tool_result",
        "response",
        "completed",
    }
)
_METRIC_CONTENT_TYPES = frozenset(
    {
        "public_prefix",
        "session_prefix",
        "private_tail",
        "tool_schema",
        "retrieval_prefix",
    }
)


def metric_workflow_stage(value: str) -> str:
    """Map untrusted workflow values to a bounded Prometheus label set."""
    return value if value in _METRIC_WORKFLOW_STAGES else "other"


def metric_content_type(value: str) -> str:
    """Map content types to the normalized schema's bounded label set."""
    return value if value in _METRIC_CONTENT_TYPES else "private_tail"


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
    session_id: str = ""
    workflow_id: str = ""
    workflow_stage: str = "unknown"
    lifecycle_state: str = "active"
    retrieval_namespace: str = ""
    content_type: str = "private_tail"
    reusable_tokens: int = 0
    kv_bytes: int = 0
    recompute_cost: float = 0.0
    lease_expires_at: float = 0.0
    reuse_probability: float = 0.0
    prediction_confidence: float = 0.0
    prediction_expires_at: float = 0.0


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
        "workflow": "workflow_id",
        "session": "session_id",
        "stage": "workflow_stage",
        "state": "lifecycle_state",
        "namespace": "retrieval_namespace",
        "reuse_probability_confidence": "prediction_confidence",
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
            "session_id",
            "workflow_id",
            "workflow_stage",
            "lifecycle_state",
            "retrieval_namespace",
            "content_type",
            "reusable_tokens",
            "kv_bytes",
            "recompute_cost",
            "lease_expires_at",
            "reuse_probability",
            "prediction_confidence",
            "prediction_expires_at",
        }
    )

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
            return result if math.isfinite(result) else default
        except (TypeError, ValueError):
            return default

    def _coerce_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(upper, max(lower, value))

    @staticmethod
    def _string(value: Any, default: str = "", max_length: int = 256) -> str:
        if value is None:
            return default
        return str(value)[:max_length]

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
        if not isinstance(context, Mapping):
            return BusinessMetadata()
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
            normalized["priority"] = int(
                self._clamp(self._coerce_int(normalized["priority"]), -100, 100)
            )

        for field_name in (
            "estimated_reload_cost",
            "estimated_reuse_prefix_len",
            "recompute_cost",
        ):
            if field_name in normalized:
                normalized[field_name] = self._clamp(
                    self._coerce_float(normalized[field_name]), 0.0, 1e9
                )
        for field_name in ("reusable_tokens", "kv_bytes"):
            if field_name in normalized:
                normalized[field_name] = int(
                    self._clamp(self._coerce_int(normalized[field_name]), 0, 1 << 50)
                )
        for field_name in (
            "reuse_probability",
            "prediction_confidence",
        ):
            if field_name in normalized:
                normalized[field_name] = self._clamp(
                    self._coerce_float(normalized[field_name]), 0.0, 1.0
                )
        for field_name in (
            "lease_expires_at",
            "prediction_expires_at",
        ):
            if field_name in normalized:
                normalized[field_name] = max(
                    0.0, self._coerce_float(normalized[field_name])
                )

        string_defaults = {
            "biz_type": "default",
            "sla_class": "standard",
            "tenant": "default",
            "trace_tag": "",
            "session_id": "",
            "workflow_id": "",
            "workflow_stage": "unknown",
            "lifecycle_state": "active",
            "retrieval_namespace": "",
            "content_type": "private_tail",
        }
        for field_name, default in string_defaults.items():
            if field_name in normalized:
                normalized[field_name] = self._string(
                    normalized[field_name], default=default
                )

        if normalized.get("sla_class") not in _SLA_WEIGHT_MULTIPLIER:
            normalized["sla_class"] = "standard"
        if normalized.get("lifecycle_state") not in {
            "active",
            "tool_waiting",
            "tool_returned",
            "completed",
            "cancelled",
        }:
            normalized["lifecycle_state"] = "active"
        if normalized.get("content_type") not in {
            "public_prefix",
            "session_prefix",
            "private_tail",
            "tool_schema",
            "retrieval_prefix",
        }:
            normalized["content_type"] = "private_tail"

        if normalized.get("business_complete"):
            normalized.setdefault("lifecycle_state", "completed")

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
            self._metadata_by_node_id[target_node_id] = replace(metadata)

    def clear(self) -> None:
        with self._lock:
            self._metadata_by_node_id.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._metadata_by_node_id)

    def bind_runtime_facts(
        self,
        node_id: int,
        metadata: BusinessMetadata,
        *,
        reusable_tokens: int,
        kv_bytes: Optional[int] = None,
    ) -> BusinessMetadata:
        """Attach normalized request metadata plus cache-observed facts."""
        bound = replace(
            metadata,
            reusable_tokens=max(0, reusable_tokens),
            kv_bytes=(metadata.kv_bytes if kv_bytes is None else max(0, kv_bytes)),
        )
        return self.set_for_node(node_id, bound)
