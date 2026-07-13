import sys
import types
import unittest
from pathlib import Path

import torch


def _bootstrap_local_sglang_import() -> None:
    if "sglang" in sys.modules:
        return

    repo_root = Path(__file__).resolve().parents[4]
    package_root = repo_root / "python" / "sglang"
    sglang_stub = types.ModuleType("sglang")
    sglang_stub.__path__ = [str(package_root)]
    sys.modules["sglang"] = sglang_stub

    # Lightweight stubs so RadixCache.create_simulated() can be imported without
    # pulling the full GPU/triton/multiprocess stack. This test only validates
    # exact prefix-match semantics on CPU.
    kv_events = types.ModuleType("sglang.srt.disaggregation.kv_events")

    class _StorageMedium:
        GPU = "GPU"

    class _BlockStored:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _BlockRemoved:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _AllBlocksCleared:
        pass

    kv_events.StorageMedium = _StorageMedium
    kv_events.BlockStored = _BlockStored
    kv_events.BlockRemoved = _BlockRemoved
    kv_events.AllBlocksCleared = _AllBlocksCleared
    sys.modules["sglang.srt.disaggregation.kv_events"] = kv_events

    allocator = types.ModuleType("sglang.srt.mem_cache.allocator")

    class _BaseTokenToKVPoolAllocator:
        pass

    allocator.BaseTokenToKVPoolAllocator = _BaseTokenToKVPoolAllocator
    sys.modules["sglang.srt.mem_cache.allocator"] = allocator

    memory_pool = types.ModuleType("sglang.srt.mem_cache.memory_pool")

    class _ReqToTokenPool:
        pass

    memory_pool.ReqToTokenPool = _ReqToTokenPool
    sys.modules["sglang.srt.mem_cache.memory_pool"] = memory_pool

    metrics = types.ModuleType("sglang.srt.observability.metrics_collector")

    class _RadixCacheMetricsCollector:
        def __init__(self, *args, **kwargs):
            pass

        def observe_eviction_duration(self, *args, **kwargs):
            pass

        def increment_eviction_num_tokens(self, *args, **kwargs):
            pass

    metrics.RadixCacheMetricsCollector = _RadixCacheMetricsCollector
    sys.modules["sglang.srt.observability.metrics_collector"] = metrics

    utils = types.ModuleType("sglang.srt.mem_cache.utils")

    def _hash_str_to_int64(value: str) -> int:
        return abs(hash(value)) & ((1 << 63) - 1)

    utils.hash_str_to_int64 = _hash_str_to_int64
    sys.modules["sglang.srt.mem_cache.utils"] = utils


_bootstrap_local_sglang_import()

from sglang.srt.mem_cache.radix_cache import (
    InsertParams,
    MatchPrefixParams,
    RadixCache,
    RadixKey,
)
from sglang.srt.mem_cache.retrieval_runtime_prefix import (
    prepend_retrieval_runtime_prefix,
    render_retrieval_runtime_prefix_text,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class TestRetrievalRuntimeReuse(unittest.TestCase):
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

    def _insert_prompt(self, cache: RadixCache, prompt_text: str, extra_key: str):
        token_ids = [ord(ch) for ch in prompt_text]
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids, extra_key),
                value=torch.tensor(list(range(len(token_ids))), dtype=torch.int64),
            )
        )
        return token_ids

    def _runtime_extra_key(self, payload):
        return render_retrieval_runtime_prefix_text(payload)

    def test_second_request_hits_exact_rendered_prefix_with_different_suffix(self):
        cache = RadixCache.create_simulated()
        payload = self._payload()
        retrieval_extra_key = self._runtime_extra_key(payload)

        # Use suffixes that diverge at the first token after the rendered
        # retrieval prefix, so the expected match length is exactly the
        # runtime prefix length rather than prefix+shared-query-substring.
        first_prompt = prepend_retrieval_runtime_prefix("A问题", payload)
        second_prompt = prepend_retrieval_runtime_prefix("B问题", payload)

        first_tokens = self._insert_prompt(cache, first_prompt, retrieval_extra_key)
        second_tokens = [ord(ch) for ch in second_prompt]
        expected_prefix_len = len([ord(ch) for ch in render_retrieval_runtime_prefix_text(payload)])

        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(second_tokens, retrieval_extra_key))
        )
        self.assertEqual(len(match.device_indices), expected_prefix_len)
        self.assertLess(expected_prefix_len, len(first_tokens))
        self.assertLess(expected_prefix_len, len(second_tokens))

    def test_chunk_content_change_is_safe_miss(self):
        cache = RadixCache.create_simulated()
        payload_a = self._payload()
        payload_b = self._payload(
            chunks=[
                {"id": "poi:1001", "content_hash": "hash-a2", "text": "店铺A: 新满减信息"},
                {"id": "poi:1002", "content_hash": "hash-b", "text": "店铺B: 配送范围"},
            ]
        )

        extra_key_a = self._runtime_extra_key(payload_a)
        extra_key_b = self._runtime_extra_key(payload_b)
        prompt_a = prepend_retrieval_runtime_prefix("问题A", payload_a)
        prompt_b = prepend_retrieval_runtime_prefix("问题A", payload_b)
        self._insert_prompt(cache, prompt_a, extra_key_a)

        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([ord(ch) for ch in prompt_b], extra_key_b))
        )
        self.assertEqual(len(match.device_indices), 0)

    def test_template_change_is_safe_miss(self):
        cache = RadixCache.create_simulated()
        payload_a = self._payload(template_rev="tpl-v3")
        payload_b = self._payload(template_rev="tpl-v4")
        extra_key_a = self._runtime_extra_key(payload_a)
        extra_key_b = self._runtime_extra_key(payload_b)

        self._insert_prompt(
            cache,
            prepend_retrieval_runtime_prefix("问题A", payload_a),
            extra_key_a,
        )
        match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    [ord(ch) for ch in prepend_retrieval_runtime_prefix("问题A", payload_b)],
                    extra_key_b,
                )
            )
        )
        self.assertEqual(len(match.device_indices), 0)


if __name__ == "__main__":
    unittest.main()
