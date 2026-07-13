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

from sglang.srt.mem_cache.retrieval_runtime_prefix import (
    prepend_retrieval_runtime_prefix,
    render_retrieval_runtime_prefix_text,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


class TestRetrievalRuntimePrefix(unittest.TestCase):
    def _payload(self, **kwargs):
        base = {
            "namespace": "waimai-poi",
            "template_rev": "tpl-v3",
            "render_rev": "render-v1",
            "schema_version": "schema-v1",
            "order_sensitive": True,
            "chunks": [
                {"id": "poi:1001", "content_hash": "hash-a", "text": "店铺A: 满减信息"},
                {"id": "poi:1002", "content_hash": "hash-b", "text": "店铺B: 配送范围"},
            ],
        }
        base.update(kwargs)
        return base

    def test_same_payload_renders_same_prefix(self):
        self.assertEqual(
            render_retrieval_runtime_prefix_text(self._payload()),
            render_retrieval_runtime_prefix_text(self._payload()),
        )

    def test_suffix_variation_keeps_same_rendered_prefix(self):
        prefix = render_retrieval_runtime_prefix_text(self._payload())
        prompt_a = prepend_retrieval_runtime_prefix("用户问题A", self._payload())
        prompt_b = prepend_retrieval_runtime_prefix("用户问题B", self._payload())
        self.assertTrue(prompt_a.startswith(prefix))
        self.assertTrue(prompt_b.startswith(prefix))
        self.assertNotEqual(prompt_a, prompt_b)

    def test_content_change_changes_rendered_prefix(self):
        self.assertNotEqual(
            render_retrieval_runtime_prefix_text(self._payload()),
            render_retrieval_runtime_prefix_text(
                self._payload(
                    chunks=[
                        {"id": "poi:1001", "content_hash": "hash-a2", "text": "店铺A: 新满减信息"},
                        {"id": "poi:1002", "content_hash": "hash-b", "text": "店铺B: 配送范围"},
                    ]
                )
            ),
        )

    def test_template_change_changes_rendered_prefix(self):
        self.assertNotEqual(
            render_retrieval_runtime_prefix_text(self._payload(template_rev="tpl-v3")),
            render_retrieval_runtime_prefix_text(self._payload(template_rev="tpl-v4")),
        )

    def test_order_sensitive_false_canonicalizes_chunks(self):
        payload_a = self._payload(
            order_sensitive=False,
            chunks=[
                {"id": "b", "content_hash": "2", "text": "B"},
                {"id": "a", "content_hash": "1", "text": "A"},
            ],
        )
        payload_b = self._payload(
            order_sensitive=False,
            chunks=[
                {"id": "a", "content_hash": "1", "text": "A"},
                {"id": "b", "content_hash": "2", "text": "B"},
            ],
        )
        self.assertEqual(
            render_retrieval_runtime_prefix_text(payload_a),
            render_retrieval_runtime_prefix_text(payload_b),
        )

    def test_missing_runtime_text_fails_closed(self):
        payload = self._payload(chunks=[{"id": "poi:1001", "content_hash": "hash-a"}])
        self.assertIsNone(render_retrieval_runtime_prefix_text(payload))
        self.assertEqual(prepend_retrieval_runtime_prefix("hello", payload), "hello")

    def test_token_id_prompt_is_not_modified(self):
        token_prompt = [1, 2, 3]
        self.assertEqual(
            prepend_retrieval_runtime_prefix(token_prompt, self._payload()),
            token_prompt,
        )


if __name__ == "__main__":
    unittest.main()
