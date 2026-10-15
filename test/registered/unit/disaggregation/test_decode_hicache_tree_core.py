"""Unit tests for decode HiCache TreeCore interactions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodePrefixMatch,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDecodeHiCacheTreeCore(CustomTestCase):
    def test_storage_probe_only_promises_prefetchable_prefix(self):
        # A long query can have only a short remote hit. Decode publishes this
        # result to prefill, so it must meet the actual prefetch threshold.
        for storage_hit, page_size, expected in (
            (0, 1, 0),
            (10, 1, 0),
            (255, 1, 0),
            (256, 1, 256),
            (255, 16, 0),
            (300, 16, 288),
        ):
            with self.subTest(storage_hit=storage_hit, page_size=page_size):
                cache = SimpleNamespace(
                    enable_storage=True,
                    cache_controller=SimpleNamespace(
                        prefetch_rate_limited=Mock(return_value=False),
                        _storage_hit_query=Mock(return_value=(None, storage_hit)),
                    ),
                    tree_core=SimpleNamespace(is_eagle=False),
                    page_size=page_size,
                    prefetch_threshold=256,
                    _all_reduce_attn_groups=Mock(),
                )
                hit = UnifiedRadixCache.query_storage_hit_length(
                    cache, 0, list(range(1024))
                )
                self.assertEqual(hit, expected)
                cache._all_reduce_attn_groups.assert_called_once()

    def test_storage_probe_applies_threshold_after_tp_min(self):
        def reduce_to_short_peer_hit(value, _op):
            value.fill_(10)

        cache = SimpleNamespace(
            enable_storage=True,
            cache_controller=SimpleNamespace(
                prefetch_rate_limited=Mock(return_value=False),
                _storage_hit_query=Mock(return_value=(None, 512)),
            ),
            tree_core=SimpleNamespace(is_eagle=False),
            page_size=1,
            prefetch_threshold=256,
            _all_reduce_attn_groups=Mock(side_effect=reduce_to_short_peer_hit),
        )
        self.assertEqual(
            UnifiedRadixCache.query_storage_hit_length(
                cache, 0, list(range(1024))
            ),
            0,
        )
        cache._all_reduce_attn_groups.assert_called_once()

    def test_asymmetric_dma_completion_keeps_poll_rank_local(self):
        # One rank's event can finish before another's. Neither completion
        # branch may enter a collective that the other rank skips.
        for done in (True, False):
            with self.subTest(event_done=done):
                event = Mock()
                event.query.return_value = done
                cache = SimpleNamespace(
                    cache_controller=SimpleNamespace(
                        layer_done_counter=SimpleNamespace(
                            events=[SimpleNamespace(finish_event=event)]
                        )
                    ),
                    loading_check=Mock(
                        side_effect=AssertionError("conditional rank collective")
                    ),
                )
                self.assertEqual(
                    UnifiedRadixCache.is_load_back_event_done(cache, 0), done
                )
                cache.loading_check.assert_not_called()

    def test_pd_success_waits_for_local_history_restore(self):
        receiver = Mock()
        receiver.poll.return_value = KVPoll.Success
        req = SimpleNamespace(
            kv_receiver=receiver,
            hicache_restore_status=HiCacheRestoreResult.PENDING,
        )
        gated = HiCacheRestoreGatedKVReceiver(req)
        self.assertEqual(gated.poll(), KVPoll.Transferring)
        req.hicache_restore_status = HiCacheRestoreResult.READY
        self.assertEqual(gated.poll(), KVPoll.Success)
        receiver.poll.return_value = KVPoll.Transferring
        self.assertEqual(gated.poll(), KVPoll.Transferring)

    def test_storage_probe_and_prefetch_use_node_handles(self):
        ongoing_prefetch = {}

        def register_prefetch(req_id, *_args, **_kwargs):
            ongoing_prefetch[req_id] = object()

        tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=True,
            ongoing_prefetch=ongoing_prefetch,
            has_ongoing_prefetch=ongoing_prefetch.__contains__,
            is_backuped=Mock(return_value=True),
            is_root=Mock(return_value=False),
            get_last_hash_value=Mock(return_value="h2"),
            get_prefix_hash_values=Mock(return_value=["h0", "h1"]),
            query_storage_hit_length=Mock(return_value=2),
            prefetch_from_storage=Mock(side_effect=register_prefetch),
        )
        harness = SimpleNamespace(
            scheduler=SimpleNamespace(enable_decode_hicache=True),
            tree_cache=tree_cache,
        )
        req = SimpleNamespace(
            rid="req-0",
            cache_request_handle=CacheRequestHandle("req-0", 0),
            origin_input_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            extra_key="model",
            cache_salt="tenant-a",
        )
        result = SimpleNamespace(
            device_indices=torch.tensor([10, 11]),
            host_hit_length=2,
            last_device_node=11,
            last_host_node=22,
        )

        prefix_match = DecodeHiCachePreallocMixin._build_decode_prefix_match(
            harness, req, result
        )

        self.assertEqual(prefix_match.l3_storage_hit_length, 2)
        tree_cache.query_storage_hit_length.assert_called_once_with(
            22,
            [4, 5, 6, 7],
            "h2",
            ["h0", "h1"],
            extra_key="model",
            cache_salt="tenant-a",
        )

        DecodeHiCachePreallocMixin._start_hicache_prefetch(harness, req, prefix_match)

        self.assertTrue(prefix_match.prefetch_registered)
        tree_cache.prefetch_from_storage.assert_called_once_with(
            req.cache_request_handle,
            22,
            [4, 5],
            "h2",
            ["h0", "h1"],
            extra_key="model",
            cache_salt="tenant-a",
        )

    def test_stale_prefetch_anchor_degrades_to_l2(self):
        tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=True,
            ongoing_prefetch={},
            get_last_hash_value=Mock(side_effect=KeyError(22)),
            get_prefix_hash_values=Mock(),
            prefetch_from_storage=Mock(),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)
        req = SimpleNamespace(
            rid="req-0",
            cache_request_handle=CacheRequestHandle("req-0", 0),
            origin_input_ids=[0, 1, 2, 3, 4, 5],
            extra_key=None,
            cache_salt=None,
        )
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.tensor([10, 11]),
            l2_host_hit_length=2,
            l3_storage_hit_length=2,
            last_device_node=11,
            last_host_node=22,
        )

        DecodeHiCachePreallocMixin._start_hicache_prefetch(harness, req, prefix_match)

        self.assertEqual(prefix_match.l3_storage_hit_length, 0)
        self.assertFalse(prefix_match.prefetch_registered)
        tree_cache.get_prefix_hash_values.assert_not_called()
        tree_cache.prefetch_from_storage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
