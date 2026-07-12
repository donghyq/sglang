"""Unit tests for business-aware radix eviction.

CPU-only tests. No server launch, no model loading.
"""

import sys
import time
import types
import unittest
from dataclasses import dataclass
from pathlib import Path


def _bootstrap_local_sglang_import() -> None:
    if "sglang" in sys.modules:
        return

    repo_root = Path(__file__).resolve().parents[4]
    package_root = repo_root / "python" / "sglang"
    sglang_stub = types.ModuleType("sglang")
    sglang_stub.__path__ = [str(package_root)]
    sys.modules["sglang"] = sglang_stub


_bootstrap_local_sglang_import()

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")

from sglang.srt.mem_cache.business_metadata import (
    BusinessMetadata,
    BusinessMetadataBuilder,
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
        self.assertLess(strategy.get_priority(cold), strategy.get_priority(hot))

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
            strategy.get_priority(completed), strategy.get_priority(active)
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

    def test_feature_flag_off_is_plain_lru(self):
        baseline = time.monotonic()
        old = FakeNode(id=100, last_access_time=baseline, hit_count=1)
        new = FakeNode(id=101, last_access_time=baseline + 10, hit_count=1)
        strategy = BusinessAwareStrategy(metadata_store=BusinessMetadataStore())
        self.assertLess(strategy.get_priority(old), strategy.get_priority(new))

    def test_extreme_values_are_bounded(self):
        store = BusinessMetadataStore()
        strategy = BusinessAwareStrategy(metadata_store=store)
        baseline = time.monotonic()

        normal = FakeNode(id=110, last_access_time=baseline, hit_count=1)
        extreme = FakeNode(id=111, last_access_time=baseline, hit_count=1)

        store.set_for_node(normal.id, BusinessMetadata())
        store.set_for_node(
            extreme.id,
            BusinessMetadata(
                hot_bucket_score=1e30,
                time_window_score=1e30,
                estimated_reload_cost=1e30,
                estimated_reuse_prefix_len=1e30,
                priority=10**9,
            ),
        )
        delta = strategy.get_priority(extreme)[0] - strategy.get_priority(normal)[0]
        self.assertLessEqual(delta, 360.0)

    def test_non_finite_values_fail_closed(self):
        store = BusinessMetadataStore()
        strategy = BusinessAwareStrategy(metadata_store=store)
        baseline = time.monotonic()

        clean = FakeNode(id=120, last_access_time=baseline, hit_count=1)
        dirty = FakeNode(id=121, last_access_time=baseline, hit_count=1)
        store.set_for_node(clean.id, BusinessMetadata())
        store.set_for_node(
            dirty.id,
            BusinessMetadata(
                hot_bucket_score=float("inf"),
                time_window_score=float("nan"),
                estimated_reload_cost=float("-inf"),
            ),
        )
        self.assertEqual(strategy.get_priority(clean), strategy.get_priority(dirty))

    def test_all_high_priority_requests_do_not_break_recency_order(self):
        store = BusinessMetadataStore()
        strategy = BusinessAwareStrategy(metadata_store=store)
        baseline = time.monotonic()
        old = FakeNode(id=130, last_access_time=baseline, hit_count=1)
        new = FakeNode(id=131, last_access_time=baseline + 10, hit_count=1)
        for node in [old, new]:
            store.set_for_node(
                node.id,
                BusinessMetadata(priority=999999, sla_class="premium"),
            )
        self.assertLess(strategy.get_priority(old), strategy.get_priority(new))


class TestBusinessMetadataBuilder(unittest.TestCase):
    def test_alias_mapping(self):
        builder = BusinessMetadataBuilder()
        metadata = builder.build(
            {
                "reload_cost": 5.0,
                "reuse_len": 32.0,
                "complete": True,
                "biz": "rag",
                "sla_class": "premium",
                "priority": 3,
            }
        )
        self.assertEqual(metadata.estimated_reload_cost, 5.0)
        self.assertEqual(metadata.estimated_reuse_prefix_len, 32.0)
        self.assertTrue(metadata.business_complete)
        self.assertEqual(metadata.biz_type, "rag")
        self.assertEqual(metadata.sla_class, "premium")
        self.assertEqual(metadata.priority, 3)

    def test_unknown_keys_ignored(self):
        builder = BusinessMetadataBuilder()
        metadata = builder.build(
            {
                "hot_bucket_score": 2.0,
                "random_field": 999,
                "another_unknown": "test",
            }
        )
        self.assertEqual(metadata.hot_bucket_score, 2.0)
        self.assertEqual(metadata.biz_type, "default")

    def test_empty_context_degrades_to_defaults(self):
        builder = BusinessMetadataBuilder()
        metadata = builder.build({})
        self.assertEqual(metadata.hot_bucket_score, 0.0)
        self.assertFalse(metadata.business_complete)
        self.assertEqual(metadata.sla_class, "standard")

    def test_invalid_values_fail_closed_to_defaults(self):
        builder = BusinessMetadataBuilder()
        metadata = builder.build(
            {
                "reload_cost": "not-a-number",
                "priority": "bad-priority",
                "complete": "unknown-bool",
            }
        )
        self.assertEqual(metadata.estimated_reload_cost, 0.0)
        self.assertEqual(metadata.priority, 0)
        self.assertFalse(metadata.business_complete)


if __name__ == "__main__":
    unittest.main()
