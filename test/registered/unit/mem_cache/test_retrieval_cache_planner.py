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

from sglang.srt.mem_cache.retrieval_cache_planner import (
    RetrievedChunkKVStore,
    RetrievalCacheMissType,
    RetrievalChunkRef,
    RetrievalConditionedKVPlanner,
    RetrievalRenderKey,
    StoredRetrievalChunk,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class TestRetrievalConditionedKVPlanner(unittest.TestCase):
    def setUp(self):
        self.store = RetrievedChunkKVStore()
        self.planner = RetrievalConditionedKVPlanner(self.store)
        self.render_key = RetrievalRenderKey(
            namespace="waimai-poi",
            model_name="Qwen/Qwen2.5-3B",
            model_fingerprint="model-fp-v1",
            tokenizer_rev="tok-v1",
            tokenizer_fingerprint="tok-fp-v1",
            template_rev="tpl-v3",
            render_rev="render-v1",
            special_token_config="sp-v1",
            schema_version="schema-v1",
        )
        self.exact_entry = StoredRetrievalChunk(
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
            kv_format="radix_prefix_v1",
            kv_blob_uri="kv://poi-1001",
            token_count=128,
            size_bytes=4096,
        )

    def test_exact_chunk_hit(self):
        self.store.put(self.exact_entry)
        plan = self.planner.plan(
            self.render_key,
            [RetrievalChunkRef(chunk_id="poi:1001", content_hash="hash-a")],
        )
        self.assertFalse(plan.fallback_required)
        self.assertEqual(plan.hit_count, 1)
        self.assertEqual(plan.reusable_token_count, 128)
        self.assertGreaterEqual(plan.plan_latency_ms, 0.0)

    def test_cross_namespace_requires_fallback(self):
        self.store.put(self.exact_entry)
        wrong_ns = RetrievalRenderKey(**{**self.render_key.__dict__, "namespace": "other"})
        plan = self.planner.plan(
            wrong_ns,
            [RetrievalChunkRef(chunk_id="poi:1001", content_hash="hash-a")],
        )
        self.assertTrue(plan.fallback_required)
        self.assertEqual(plan.decisions[0].miss_type, RetrievalCacheMissType.BLOB_MISSING)

    def test_cross_model_requires_fallback(self):
        self.store.put(self.exact_entry)
        wrong_model = RetrievalRenderKey(**{**self.render_key.__dict__, "model_name": "OtherModel"})
        plan = self.planner.plan(wrong_model, [RetrievalChunkRef(chunk_id="poi:1001", content_hash="hash-a")])
        self.assertEqual(plan.decisions[0].miss_type, RetrievalCacheMissType.MODEL_MISMATCH)

    def test_cross_tokenizer_requires_fallback(self):
        self.store.put(self.exact_entry)
        wrong_tok = RetrievalRenderKey(**{**self.render_key.__dict__, "tokenizer_rev": "tok-v2"})
        plan = self.planner.plan(wrong_tok, [RetrievalChunkRef(chunk_id="poi:1001", content_hash="hash-a")])
        self.assertEqual(plan.decisions[0].miss_type, RetrievalCacheMissType.TOKENIZER_MISMATCH)

    def test_template_change_invalidates_reuse(self):
        self.store.put(self.exact_entry)
        wrong_tpl = RetrievalRenderKey(**{**self.render_key.__dict__, "template_rev": "tpl-v2"})
        plan = self.planner.plan(wrong_tpl, [RetrievalChunkRef(chunk_id="poi:1001", content_hash="hash-a")])
        self.assertEqual(plan.decisions[0].miss_type, RetrievalCacheMissType.TEMPLATE_MISMATCH)

    def test_render_change_invalidates_reuse(self):
        self.store.put(self.exact_entry)
        wrong_render = RetrievalRenderKey(**{**self.render_key.__dict__, "render_rev": "render-v2"})
        plan = self.planner.plan(wrong_render, [RetrievalChunkRef(chunk_id="poi:1001", content_hash="hash-a")])
        self.assertEqual(plan.decisions[0].miss_type, RetrievalCacheMissType.RENDER_MISMATCH)

    def test_content_hash_change_invalidates_reuse(self):
        self.store.put(self.exact_entry)
        plan = self.planner.plan(
            self.render_key,
            [RetrievalChunkRef(chunk_id="poi:1001", content_hash="hash-b")],
        )
        self.assertEqual(plan.decisions[0].miss_type, RetrievalCacheMissType.BLOB_MISSING)

    def test_request_missing_hash_must_fail_closed(self):
        self.store.put(self.exact_entry)
        plan = self.planner.plan(self.render_key, [RetrievalChunkRef(chunk_id="poi:1001")])
        self.assertEqual(plan.decisions[0].miss_type, RetrievalCacheMissType.MISSING_IDENTITY)

    def test_store_missing_identity_must_fail_closed(self):
        broken = StoredRetrievalChunk(
            namespace="waimai-poi",
            chunk_id="poi:1001",
            content_hash="hash-a",
            model_name="",
            model_fingerprint="model-fp-v1",
            tokenizer_rev="tok-v1",
            tokenizer_fingerprint="tok-fp-v1",
            template_rev="tpl-v3",
            render_rev="render-v1",
            special_token_config="sp-v1",
            schema_version="schema-v1",
            kv_blob_uri="kv://poi-1001",
        )
        self.store.put(broken)
        plan = self.planner.plan(self.render_key, [RetrievalChunkRef(chunk_id="poi:1001", content_hash="hash-a")])
        self.assertEqual(plan.decisions[0].miss_type, RetrievalCacheMissType.MISSING_IDENTITY)

    def test_blob_missing_requires_fallback(self):
        self.store.put(StoredRetrievalChunk(**{**self.exact_entry.__dict__, "kv_blob_uri": None}))
        plan = self.planner.plan(self.render_key, [RetrievalChunkRef(chunk_id="poi:1001", content_hash="hash-a")])
        self.assertEqual(plan.decisions[0].miss_type, RetrievalCacheMissType.BLOB_MISSING)

    def test_partial_hit_preserves_reusable_token_count(self):
        self.store.put(self.exact_entry)
        plan = self.planner.plan(
            self.render_key,
            [
                RetrievalChunkRef(chunk_id="poi:1001", content_hash="hash-a"),
                RetrievalChunkRef(chunk_id="poi:1002", content_hash="hash-b"),
            ],
        )
        self.assertTrue(plan.fallback_required)
        self.assertEqual(plan.hit_count, 1)
        self.assertEqual(plan.miss_count, 1)
        self.assertEqual(plan.reusable_token_count, 128)

    def test_empty_chunks_is_noop_plan(self):
        plan = self.planner.plan(self.render_key, [])
        self.assertFalse(plan.fallback_required)
        self.assertEqual(plan.hit_count, 0)
        self.assertEqual(plan.miss_count, 0)
        self.assertEqual(plan.reusable_token_count, 0)


if __name__ == "__main__":
    unittest.main()
