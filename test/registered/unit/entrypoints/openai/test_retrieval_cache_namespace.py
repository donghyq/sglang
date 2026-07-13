import sys
import types
from pathlib import Path


def _bootstrap_local_sglang_import() -> None:
    if "sglang" in sys.modules:
        return

    repo_root = Path(__file__).resolve().parents[5]
    package_root = repo_root / "python" / "sglang"
    sglang_stub = types.ModuleType("sglang")
    sglang_stub.__path__ = [str(package_root)]
    sys.modules["sglang"] = sglang_stub


_bootstrap_local_sglang_import()

def maybe_stub_sgl_kernel():
    return None


maybe_stub_sgl_kernel()

import unittest
from unittest.mock import Mock

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    RetrievalCacheSpec,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _DummyServing(OpenAIServingBase):
    def __init__(self):
        tokenizer_manager = Mock()
        tokenizer_manager.server_args = Mock(tokenizer_metrics_allowed_custom_labels=None)
        super().__init__(tokenizer_manager)

    def _request_id_prefix(self) -> str:
        return "dummy-"

    def _convert_to_internal_request(self, request, raw_request=None):
        return request, request


class TestRetrievalCacheNamespace(unittest.TestCase):
    def setUp(self):
        self.serving = _DummyServing()

    def _spec(self, **kwargs):
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
            chunks=[
                {"id": "poi:1001", "content_hash": "aaa"},
                {"id": "poi:1002", "content_hash": "bbb"},
            ],
        )
        base.update(kwargs)
        return RetrievalCacheSpec(**base)

    def test_same_retrieval_spec_yields_same_key(self):
        self.assertEqual(self._spec().to_extra_key(), self._spec().to_extra_key())

    def test_correctness_field_change_changes_key(self):
        self.assertNotEqual(
            self._spec(model_fingerprint="model-fp-v1").to_extra_key(),
            self._spec(model_fingerprint="model-fp-v2").to_extra_key(),
        )

    def test_order_sensitive_spec_distinguishes_chunk_order(self):
        self.assertNotEqual(
            self._spec(chunks=[{"id": "a", "content_hash": "1"}, {"id": "b", "content_hash": "2"}], order_sensitive=True).to_extra_key(),
            self._spec(chunks=[{"id": "b", "content_hash": "2"}, {"id": "a", "content_hash": "1"}], order_sensitive=True).to_extra_key(),
        )

    def test_order_insensitive_spec_collapses_chunk_order(self):
        self.assertEqual(
            self._spec(chunks=[{"id": "a", "content_hash": "1"}, {"id": "b", "content_hash": "2"}], order_sensitive=False).to_extra_key(),
            self._spec(chunks=[{"id": "b", "content_hash": "2"}, {"id": "a", "content_hash": "1"}], order_sensitive=False).to_extra_key(),
        )

    def test_no_retrieval_cache_keeps_old_behavior(self):
        request = CompletionRequest(model="test-model", prompt="hello")
        self.assertIsNone(self.serving._compute_extra_key(request))

    def test_compute_extra_key_preserves_existing_fields(self):
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            cache_salt="tenant-a",
            extra_key="user-segment-1",
            retrieval_cache=self._spec(),
        )
        extra_key = self.serving._compute_extra_key(request)
        self.assertIn("cache_salt=tenant-a", extra_key)
        self.assertIn("retrieval_cache=retrieval:v1:", extra_key)
        self.assertIn("extra_key=user-segment-1", extra_key)


if __name__ == "__main__":
    unittest.main()
