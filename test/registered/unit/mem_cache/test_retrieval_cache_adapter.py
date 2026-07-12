import sys
import types
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

import unittest
from unittest.mock import Mock

from sglang.srt.mem_cache.retrieval_cache_adapter import (
    build_planning_context,
    plan_from_payload,
    summarize_plan,
)
from sglang.srt.mem_cache.retrieval_cache_planner import (
    RetrievedChunkKVStore,
    RetrievalConditionedKVPlanner,
    StoredRetrievalChunk,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestRetrievalCacheAdapter(unittest.TestCase):
    def _payload(self):
        return {
            "namespace": "waimai-poi",
            "model_name": "Qwen/Qwen2.5-3B",
            "model_fingerprint": "model-fp-v1",
            "tokenizer_rev": "tok-v1",
            "tokenizer_fingerprint": "tok-fp-v1",
            "template_rev": "tpl-v3",
            "render_rev": "render-v1",
            "special_token_config": "sp-v1",
            "schema_version": "schema-v1",
            "order_sensitive": True,
            "chunks": [
                {"id": "poi:1001", "content_hash": "hash-a"},
                {"id": "poi:1002", "content_hash": "hash-b"},
            ],
        }

    def test_build_planning_context(self):
        context = build_planning_context(self._payload())
        self.assertIsNotNone(context)
        self.assertEqual(context.render_key.template_rev, "tpl-v3")
        self.assertEqual(context.render_key.model_fingerprint, "model-fp-v1")
        self.assertEqual(len(context.retrieved_chunks), 2)

    def test_missing_chunk_id_raises(self):
        payload = self._payload()
        payload["chunks"] = [{"content_hash": "hash-a"}]
        with self.assertRaises(ValueError):
            build_planning_context(payload)

    def test_plan_and_summary_from_payload(self):
        payload = self._payload()
        store = RetrievedChunkKVStore()
        store.put(
            StoredRetrievalChunk(
                namespace="waimai-poi",
                chunk_id="poi:1001",
                content_hash="hash-a",
                model_name="Qwen/Qwen2.5-3B",
                model_fingerprint="model-fp-v1",
                tokenizer_rev="tok-v1",
                tokenizer_fingerprint="tok-fp-v1",
                template_rev="tpl-v3",
                render_rev="render-v1",
                special_token_config="sp-v1",
                schema_version="schema-v1",
                kv_blob_uri="kv://poi-1001",
                token_count=64,
            )
        )
        planner = RetrievalConditionedKVPlanner(store)
        plan = plan_from_payload(planner, payload)
        self.assertIsNotNone(plan)
        summary = summarize_plan(plan)
        self.assertTrue(summary.fallback_required)
        self.assertEqual(summary.hit_chunks, 1)
        self.assertEqual(summary.miss_chunks, 1)
        self.assertEqual(summary.reusable_token_count, 64)
        self.assertGreaterEqual(summary.plan_latency_ms, 0.0)
        self.assertEqual(summary.miss_breakdown, {"blob_missing": 1})

    def test_planner_exception_falls_back_to_none(self):
        planner = Mock()
        planner.plan.side_effect = RuntimeError("boom")
        plan = plan_from_payload(planner, self._payload())
        self.assertIsNone(plan)

    def test_no_payload_returns_none(self):
        self.assertIsNone(plan_from_payload(Mock(), None))


if __name__ == "__main__":
    unittest.main()
