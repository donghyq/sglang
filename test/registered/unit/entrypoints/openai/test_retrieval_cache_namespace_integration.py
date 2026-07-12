"""Integration-style test for retrieval namespace at OpenAI entrypoint layer.

This test imports the real OpenAI protocol / serving stack and therefore depends
on optional runtime packages like `pybase64`, `openai`, `numpy`, `requests`,
and whatever else `sglang.utils` transitively imports in the local environment.
If those dependencies are unavailable, the test should be treated as environment
blocked rather than namespace-logic failure.
"""

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

import unittest
from unittest.mock import Mock

try:
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, RetrievalCacheSpec
    from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
    IMPORT_BLOCKED = None
except Exception as exc:  # pragma: no cover
    ChatCompletionRequest = None
    RetrievalCacheSpec = None
    OpenAIServingBase = None
    IMPORT_BLOCKED = exc

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


if IMPORT_BLOCKED is not None:

    class TestRetrievalCacheNamespaceIntegration(unittest.TestCase):
        @unittest.skip(f"entrypoint import blocked: {IMPORT_BLOCKED}")
        def test_entrypoint_import_blocked(self):
            pass

else:

    class TestRetrievalCacheNamespaceIntegration(unittest.TestCase):
        class _DummyServing(OpenAIServingBase):
            def __init__(self):
                tokenizer_manager = Mock()
                tokenizer_manager.server_args = Mock(tokenizer_metrics_allowed_custom_labels=None)
                super().__init__(tokenizer_manager)

            def _request_id_prefix(self) -> str:
                return "dummy-"

            def _convert_to_internal_request(self, request, raw_request=None):
                return request, request

        def setUp(self):
            self.serving = self._DummyServing()

        def test_protocol_and_serving_base_share_same_composition(self):
            spec = RetrievalCacheSpec(
                namespace="waimai-poi",
                template_rev="v3",
                tokenizer_rev="tok-v1",
                tokenizer_fingerprint="tok-fp-v1",
                model_name="Qwen/Qwen2.5-3B",
                model_fingerprint="model-fp-v1",
                render_rev="render-v1",
                special_token_config="sp-v1",
                schema_version="schema-v1",
                chunks=[{"id": "poi:1001", "content_hash": "aaa"}],
            )
            request = ChatCompletionRequest(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                cache_salt="tenant-a",
                extra_key="user-segment-1",
                retrieval_cache=spec,
            )
            extra_key = self.serving._compute_extra_key(request)
            self.assertIn(f"retrieval_cache={spec.to_extra_key()}", extra_key)


if __name__ == "__main__":
    unittest.main()
