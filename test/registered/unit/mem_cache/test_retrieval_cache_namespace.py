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

from sglang.srt.mem_cache.retrieval_namespace import (
    compose_prefix_cache_extra_key,
    compute_retrieval_extra_key,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


class TestRetrievalCacheNamespace(unittest.TestCase):
    def _payload(self, **kwargs):
        base = dict(
            namespace="waimai-poi",
            template_rev="v3",
            tokenizer_rev="tok-v1",
            tokenizer_fingerprint="tok-fp-v1",
            model_name="Qwen/Qwen2.5-3B",
            model_fingerprint="model-fp-v1",
            render_rev="render-v1",
            special_token_config="sp-v1",
            schema_version="schema-v1",
            order_sensitive=True,
            chunks=[
                {"id": "poi:1001", "content_hash": "aaa"},
                {"id": "poi:1002", "content_hash": "bbb"},
            ],
        )
        base.update(kwargs)
        return base

    def test_same_retrieval_spec_yields_same_key(self):
        self.assertEqual(
            compute_retrieval_extra_key(self._payload()),
            compute_retrieval_extra_key(self._payload()),
        )

    def test_namespace_change_changes_key(self):
        self.assertNotEqual(
            compute_retrieval_extra_key(self._payload(namespace="waimai-poi")),
            compute_retrieval_extra_key(self._payload(namespace="docs")),
        )

    def test_content_hash_change_changes_key(self):
        self.assertNotEqual(
            compute_retrieval_extra_key(self._payload()),
            compute_retrieval_extra_key(
                self._payload(chunks=[{"id": "poi:1001", "content_hash": "ccc"}, {"id": "poi:1002", "content_hash": "bbb"}])
            ),
        )

    def test_model_tokenizer_template_render_change_changes_key(self):
        self.assertNotEqual(
            compute_retrieval_extra_key(self._payload(model_fingerprint="model-fp-v1")),
            compute_retrieval_extra_key(self._payload(model_fingerprint="model-fp-v2")),
        )
        self.assertNotEqual(
            compute_retrieval_extra_key(self._payload(tokenizer_fingerprint="tok-fp-v1")),
            compute_retrieval_extra_key(self._payload(tokenizer_fingerprint="tok-fp-v2")),
        )
        self.assertNotEqual(
            compute_retrieval_extra_key(self._payload(template_rev="v3")),
            compute_retrieval_extra_key(self._payload(template_rev="v4")),
        )
        self.assertNotEqual(
            compute_retrieval_extra_key(self._payload(render_rev="render-v1")),
            compute_retrieval_extra_key(self._payload(render_rev="render-v2")),
        )

    def test_order_sensitive_true_changes_key_when_order_changes(self):
        self.assertNotEqual(
            compute_retrieval_extra_key(self._payload(order_sensitive=True, chunks=[{"id": "a", "content_hash": "1"}, {"id": "b", "content_hash": "2"}])),
            compute_retrieval_extra_key(self._payload(order_sensitive=True, chunks=[{"id": "b", "content_hash": "2"}, {"id": "a", "content_hash": "1"}])),
        )

    def test_order_sensitive_false_uses_canonical_order(self):
        self.assertEqual(
            compute_retrieval_extra_key(self._payload(order_sensitive=False, chunks=[{"id": "a", "content_hash": "1"}, {"id": "b", "content_hash": "2"}])),
            compute_retrieval_extra_key(self._payload(order_sensitive=False, chunks=[{"id": "b", "content_hash": "2"}, {"id": "a", "content_hash": "1"}])),
        )

    def test_compose_with_cache_salt_and_extra_key_is_unambiguous(self):
        key = compose_prefix_cache_extra_key("tenant-a", self._payload(), "user-segment-1")
        self.assertIn("cache_salt=tenant-a", key)
        self.assertIn("retrieval_cache=retrieval:v1:", key)
        self.assertIn("extra_key=user-segment-1", key)
        self.assertEqual(key.count("|"), 2)

    def test_empty_retrieval_spec_preserves_old_behavior(self):
        self.assertIsNone(compute_retrieval_extra_key({"chunks": [], "order_sensitive": True}))
        self.assertIsNone(compose_prefix_cache_extra_key(None, {"chunks": [], "order_sensitive": True}, None))


if __name__ == "__main__":
    unittest.main()
