import sys
import time
import types
import unittest
from dataclasses import dataclass
from pathlib import Path


def _bootstrap_local_sglang_import() -> None:
    if "sglang" in sys.modules:
        return

    repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / "python" / "sglang"
    sglang_stub = types.ModuleType("sglang")
    sglang_stub.__path__ = [str(package_root)]
    sys.modules["sglang"] = sglang_stub


_bootstrap_local_sglang_import()

from sglang.srt.mem_cache.business_metadata import (
    BusinessMetadata,
    BusinessMetadataStore,
)
from sglang.srt.mem_cache.evict_policy import BusinessAwareStrategy


@dataclass
class FakeNode:
    id: int
    last_access_time: float
    hit_count: int


class TestBusinessAwareEvictionPolicy(unittest.TestCase):

    def test_business_aware_strategy_prefers_hotter_node(self):
        store = BusinessMetadataStore()
        strategy = BusinessAwareStrategy(metadata_store=store)

        baseline = time.monotonic()
        cold = FakeNode(id=1, last_access_time=baseline, hit_count=1)
        hot = FakeNode(id=2, last_access_time=baseline, hit_count=1)

        store.set_for_node(
            cold.id,
            BusinessMetadata(
                hot_bucket_score=0.0,
                time_window_score=0.0,
                estimated_reload_cost=1.0,
                estimated_reuse_prefix_len=8.0,
            ),
        )
        store.set_for_node(
            hot.id,
            BusinessMetadata(
                hot_bucket_score=3.0,
                time_window_score=2.0,
                estimated_reload_cost=4.0,
                estimated_reuse_prefix_len=32.0,
            ),
        )

        # Smaller priority tuple means evicted earlier.
        self.assertLess(
            strategy.get_priority(cold),
            strategy.get_priority(hot),
        )

    def test_business_complete_penalizes_node(self):
        store = BusinessMetadataStore()
        strategy = BusinessAwareStrategy(metadata_store=store)

        baseline = time.monotonic()
        active = FakeNode(id=10, last_access_time=baseline, hit_count=3)
        completed = FakeNode(id=11, last_access_time=baseline, hit_count=3)

        store.set_for_node(
            active.id,
            BusinessMetadata(
                hot_bucket_score=1.0,
                estimated_reload_cost=2.0,
                estimated_reuse_prefix_len=16.0,
                business_complete=False,
            ),
        )
        store.set_for_node(
            completed.id,
            BusinessMetadata(
                hot_bucket_score=1.0,
                estimated_reload_cost=2.0,
                estimated_reuse_prefix_len=16.0,
                business_complete=True,
            ),
        )

        self.assertLess(
            strategy.get_priority(completed),
            strategy.get_priority(active),
        )

    def test_explain_contains_keep_score_and_metadata(self):
        store = BusinessMetadataStore()
        strategy = BusinessAwareStrategy(metadata_store=store)

        node = FakeNode(id=21, last_access_time=123.0, hit_count=7)
        store.set_for_node(
            node.id,
            BusinessMetadata(
                hot_bucket_score=4.0,
                time_window_score=1.5,
                estimated_reload_cost=6.0,
                estimated_reuse_prefix_len=24.0,
                business_complete=False,
            ),
        )

        explanation = strategy.explain(node)
        self.assertEqual(explanation["node_id"], 21)
        self.assertEqual(explanation["hit_count"], 7)
        self.assertEqual(explanation["hot_bucket_score"], 4.0)
        self.assertEqual(explanation["estimated_reuse_prefix_len"], 24.0)
        self.assertIn("keep_score", explanation)
        self.assertIn("priority", explanation)


if __name__ == "__main__":
    unittest.main()
