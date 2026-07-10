from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BusinessMetadata:
    hot_bucket_score: float = 0.0
    time_window_score: float = 0.0
    estimated_reload_cost: float = 0.0
    estimated_reuse_prefix_len: float = 0.0
    business_complete: bool = False


class BusinessMetadataStore:
    """Thread-safe side metadata store for cache nodes.

    The first version keeps the design intentionally simple:
    - metadata is keyed by node id
    - callers may optionally attach metadata after node creation
    - absent metadata falls back to zero-value scoring inputs
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._metadata_by_node_id: Dict[int, BusinessMetadata] = {}

    def set_for_node(self, node_id: int, metadata: BusinessMetadata) -> None:
        with self._lock:
            self._metadata_by_node_id[node_id] = metadata

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
            )
