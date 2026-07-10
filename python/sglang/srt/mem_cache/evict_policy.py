from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

from sglang.srt.mem_cache.business_metadata import BusinessMetadataStore

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
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


class BusinessAwareStrategy(EvictionStrategy):
    """Business-aware eviction with a conservative first-version score.

    Smaller tuple values are evicted earlier.
    The first element is the keep-score itself so nodes with a smaller keep-score
    are evicted earlier. The second element keeps deterministic LRU tie-breaking.
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
    ):
        self.metadata_store = metadata_store
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.reuse_prefix_weight = reuse_prefix_weight
        self.reload_cost_weight = reload_cost_weight
        self.hot_bucket_weight = hot_bucket_weight
        self.time_window_weight = time_window_weight
        self.business_complete_penalty = business_complete_penalty

    def _compute_keep_score(self, node: "TreeNode") -> float:
        metadata = self.metadata_store.get_for_node(node.id)
        hot_bucket_score = 0.0
        time_window_score = 0.0
        estimated_reload_cost = 0.0
        estimated_reuse_prefix_len = 0.0
        business_complete = False
        if metadata is not None:
            hot_bucket_score = metadata.hot_bucket_score
            time_window_score = metadata.time_window_score
            estimated_reload_cost = metadata.estimated_reload_cost
            estimated_reuse_prefix_len = metadata.estimated_reuse_prefix_len
            business_complete = metadata.business_complete

        keep_score = (
            self.recency_weight * node.last_access_time
            + self.frequency_weight * node.hit_count
            + self.reuse_prefix_weight * estimated_reuse_prefix_len
            + self.reload_cost_weight * estimated_reload_cost
            + self.hot_bucket_weight * hot_bucket_score
            + self.time_window_weight * time_window_score
        )
        if business_complete:
            keep_score -= self.business_complete_penalty
        return keep_score

    def explain(self, node: "TreeNode") -> dict:
        metadata = self.metadata_store.get_for_node(node.id)
        hot_bucket_score = 0.0
        time_window_score = 0.0
        estimated_reload_cost = 0.0
        estimated_reuse_prefix_len = 0.0
        business_complete = False
        if metadata is not None:
            hot_bucket_score = metadata.hot_bucket_score
            time_window_score = metadata.time_window_score
            estimated_reload_cost = metadata.estimated_reload_cost
            estimated_reuse_prefix_len = metadata.estimated_reuse_prefix_len
            business_complete = metadata.business_complete

        keep_score = self._compute_keep_score(node)
        return {
            "node_id": node.id,
            "last_access_time": node.last_access_time,
            "hit_count": node.hit_count,
            "hot_bucket_score": hot_bucket_score,
            "time_window_score": time_window_score,
            "estimated_reload_cost": estimated_reload_cost,
            "estimated_reuse_prefix_len": estimated_reuse_prefix_len,
            "business_complete": business_complete,
            "keep_score": keep_score,
            "priority": self.get_priority(node),
        }

    def get_priority(self, node: "TreeNode") -> Tuple[float, float]:
        keep_score = self._compute_keep_score(node)
        return (keep_score, node.last_access_time)
