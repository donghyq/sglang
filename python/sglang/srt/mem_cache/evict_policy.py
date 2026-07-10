from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

from sglang.srt.mem_cache.business_metadata import (
    BusinessMetadataStore,
    _sla_multiplier,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


class BusinessAwareStrategy(EvictionStrategy):
    """Business-aware eviction with bounded score composition.

    Smaller tuple values are evicted earlier.  The first element is the
    keep-score; the second keeps deterministic LRU tie-breaking.

    Design principles:
    - **Bounded business influence**: SLA class provides a multiplier in
      [0.5, 1.5], never enough to fully override recency/frequency.
    - **Graceful degradation**: when metadata is absent the score collapses
      to pure recency + frequency, i.e. LRU-like behaviour.
    - **business_complete penalty**: a finished business turn pushes the node
      toward early eviction, but the penalty is capped.
    """

    def __init__(
        self,
        metadata_store: BusinessMetadataStore,
        recency_weight: float = 1.0,
        frequency_weight: float = 1.0,
        reuse_prefix_weight: float = 1.0,
        reload_cost_weight: float = 1.0,
        hot_bucket_weight: float = 1.0,
        time_window_weight: float = 0.5,
        business_complete_penalty: float = 5.0,
        priority_weight: float = 0.5,
    ):
        self.metadata_store = metadata_store
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.reuse_prefix_weight = reuse_prefix_weight
        self.reload_cost_weight = reload_cost_weight
        self.hot_bucket_weight = hot_bucket_weight
        self.time_window_weight = time_window_weight
        self.business_complete_penalty = business_complete_penalty
        self.priority_weight = priority_weight

    def _compute_keep_score(self, node: "TreeNode") -> float:
        metadata = self.metadata_store.get_for_node(node.id)

        if metadata is None:
            # Graceful degradation: pure recency + frequency (LRU-like).
            return (
                self.recency_weight * node.last_access_time
                + self.frequency_weight * node.hit_count
            )

        # Bounded SLA multiplier in [0.5, 1.5].
        sla_mult = _sla_multiplier(metadata.sla_class)

        keep_score = (
            self.recency_weight * node.last_access_time
            + self.frequency_weight * node.hit_count
            + self.reuse_prefix_weight * metadata.estimated_reuse_prefix_len
            + self.reload_cost_weight * metadata.estimated_reload_cost
            + self.hot_bucket_weight * metadata.hot_bucket_score
            + self.time_window_weight * metadata.time_window_score
            + self.priority_weight * metadata.priority
        )
        keep_score *= sla_mult

        if metadata.business_complete:
            keep_score -= self.business_complete_penalty

        return keep_score

    def explain(self, node: "TreeNode") -> dict:
        metadata = self.metadata_store.get_for_node(node.id)
        keep_score = self._compute_keep_score(node)

        if metadata is None:
            return {
                "node_id": node.id,
                "last_access_time": node.last_access_time,
                "hit_count": node.hit_count,
                "metadata_present": False,
                "keep_score": keep_score,
                "priority": self.get_priority(node),
                "note": "no business metadata; degraded to LRU-like score",
            }

        return {
            "node_id": node.id,
            "last_access_time": node.last_access_time,
            "hit_count": node.hit_count,
            "metadata_present": True,
            "hot_bucket_score": metadata.hot_bucket_score,
            "time_window_score": metadata.time_window_score,
            "estimated_reload_cost": metadata.estimated_reload_cost,
            "estimated_reuse_prefix_len": metadata.estimated_reuse_prefix_len,
            "business_complete": metadata.business_complete,
            "biz_type": metadata.biz_type,
            "priority": metadata.priority,
            "sla_class": metadata.sla_class,
            "tenant": metadata.tenant,
            "trace_tag": metadata.trace_tag,
            "sla_weight_multiplier": _sla_multiplier(metadata.sla_class),
            "keep_score": keep_score,
            "priority": self.get_priority(node),
        }

    def get_priority(self, node: "TreeNode") -> Tuple[float, float]:
        keep_score = self._compute_keep_score(node)
        return (keep_score, node.last_access_time)
