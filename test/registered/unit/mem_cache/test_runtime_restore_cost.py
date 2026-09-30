"""Measured cost profile, rank votes, rejection and cleanup; no GPU kernels."""

import json
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from sglang.srt.mem_cache.restore_cost import (
    RuntimeRestoreCostGate,
    consensus_gate_action,
)
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from test_storage_prefetch_lifecycle import _cache_mode_hit_fixture

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def profile(mode="enforce"):
    return {
        "schema_version": 1,
        "mode": mode,
        "model": "30B",
        "tp": 2,
        "kv_bytes_per_token_per_rank": 49152,
        "cost_scope": "request_ttft_ms",
        "min_samples": 3,
        "window_size": 20,
        "risk_margin_ms": 0,
        "buckets": [
            {
                "prefix_tokens": 8,
                "source_tier": "mooncake",
                "load_bucket": "idle",
                "restore_ttft_ms": [60, 62, 61],
                "recompute_ttft_ms": [35, 37, 36],
            }
        ],
    }


def two_rank(rank, rendezvous, missing):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        cache, operation = _cache_mode_hit_fixture(8)
        cache._restore_cost_records = {}
        cache._restore_cost_declined = set()
        cache.cache_controller.storage_backend_type = "mooncake"
        cache.restore_cost_gate = SimpleNamespace(
            evaluate=lambda *args: {
                "mode": "enforce",
                "valid": not (missing and rank == 1),
                "should_restore": rank == 1,
                "profile_signature": 123,
                "fallback_reason": None,
                "predicted_restore_ms": 60,
                "predicted_recompute_ms": 36,
            }
        )
        cache._all_reduce_storage_admission = lambda vote: torch.distributed.all_reduce(
            vote, op=torch.distributed.ReduceOp.MIN
        )
        permit = cache._runtime_restore_cost_action(
            operation, 8, cache.ongoing_prefetch[operation.handle]
        )
        assert permit == missing
        assert cache._restore_cost_records[operation.handle]["action"] == (
            "restore" if missing else "recompute"
        )
    finally:
        torch.distributed.destroy_process_group()


class TestRuntimeRestoreCost(CustomTestCase):
    def test_collector_is_wired_on_attach_and_cleared_on_disable(self):
        collector = Mock()
        cache = SimpleNamespace(
            enable_storage=True,
            page_size=1,
            cache_controller=SimpleNamespace(),
            storage_metrics_collector=None,
        )
        attachment = StorageAttachment(cache)
        attachment._resolve_metrics_collector = Mock(return_value=collector)
        kwargs = {
            "storage_backend": "mooncake",
            "prefetch_threshold": 256,
            "prefetch_timeout_base": 1.0,
            "prefetch_timeout_per_ki_token": 0.25,
            "hicache_storage_pass_prefix_keys": False,
            "enable_storage": True,
            "extra_metric_labels": None,
        }
        attachment.apply_runtime_config(**kwargs, enable_storage_metrics=True)
        self.assertIs(cache.cache_controller.storage_metrics_collector, collector)
        attachment.apply_runtime_config(**kwargs, enable_storage_metrics=False)
        self.assertIsNone(cache.cache_controller.storage_metrics_collector)

    def test_profile_scope_window_reload_and_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            filename = Path(d) / "profile.json"
            gate = RuntimeRestoreCostGate(str(filename), "30B", 2, 49152)
            self.assertFalse(gate.evaluate(8, "mooncake", "idle")["valid"])
            filename.write_text(json.dumps(profile()))
            decision = gate.evaluate(8, "mooncake", "idle")
            self.assertFalse(decision["should_restore"])
            self.assertEqual(decision["kv_payload_bytes_per_rank"], 8 * 49152)
            self.assertEqual(decision["kv_payload_bytes_all_ranks"], 8 * 98304)
            self.assertFalse(gate.evaluate(8192, "mooncake", "idle")["valid"])
            self.assertFalse(gate.evaluate(8, "mooncake", "transfer_busy")["valid"])
            p = profile("shadow")
            p["buckets"][0]["restore_ttft_ms"] = [20, 22, 21]
            filename.write_text(json.dumps(p))
            self.assertTrue(gate.evaluate(8, "mooncake", "idle")["should_restore"])
            p["tp"] = 4
            filename.write_text(json.dumps(p))
            self.assertFalse(gate.evaluate(8, "mooncake", "idle")["valid"])
            filename.write_text('{"schema_version":1,"buckets":null}')
            self.assertTrue(gate.evaluate(8, "mooncake", "idle")["should_restore"])

    def test_two_rank_consensus_and_missing_rank_fallback(self):
        for missing in (False, True):
            with tempfile.TemporaryDirectory() as d:
                torch.multiprocessing.spawn(
                    two_rank, args=(str(Path(d) / "rdzv"), missing), nprocs=2, join=True
                )
        self.assertEqual(
            consensus_gate_action([1, 0, 123, -124]), (True, "rank_profile_mismatch")
        )

    def test_rejection_skips_allocation_and_allows_native_prefill(self):
        with tempfile.TemporaryDirectory() as d:
            filename = Path(d) / "profile.json"
            filename.write_text(json.dumps(profile()))
            cache, operation = _cache_mode_hit_fixture(8)
            cache.enable_storage = True
            cache.linker = None
            cache.prefetch_loaded_tokens_by_reqid = {}
            cache.prefetch_loaded_storage_start_by_reqid = {}
            cache._restore_cost_records = {}
            cache._restore_cost_declined = set()
            cache.restore_cost_gate = RuntimeRestoreCostGate(
                str(filename), "30B", 2, 49152
            )
            cache.cache_controller.storage_backend_type = "mooncake"
            cache.cache_controller.prefetch_tokens_occupied = 8
            cache.revoke_pending_prefetch = lambda h: (
                UnifiedRadixCache.revoke_pending_prefetch(cache, h)
            )
            operation.mark_terminate = Mock()
            cache._drain_storage_control_queues_impl(1, 0, 0, 0, {}, False)
            cache.cache_controller.mem_pool_host.alloc.assert_not_called()
            self.assertEqual(cache.ongoing_prefetch, {})
            self.assertTrue(cache.cache_controller.prefetch_buffer.empty())
            self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
            self.assertFalse(
                cache.prefetch_from_storage(operation.handle, 0, list(range(8)))
            )
            cache.release_aborted_request(operation.handle)
            self.assertEqual(cache._restore_cost_records, {})
            self.assertEqual(cache._restore_cost_declined, set())

    def test_accepted_hit_timeout_and_cancel_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            filename = Path(d) / "profile.json"
            p = profile()
            p["buckets"][0]["restore_ttft_ms"] = [20, 22, 21]
            filename.write_text(json.dumps(p))
            cache, operation = _cache_mode_hit_fixture(8)
            cache.linker = None
            cache.prefetch_loaded_tokens_by_reqid = {}
            cache.prefetch_loaded_storage_start_by_reqid = {}
            cache._restore_cost_records = {}
            cache._restore_cost_declined = set()
            cache.restore_cost_gate = RuntimeRestoreCostGate(
                str(filename), "30B", 2, 49152
            )
            cc = cache.cache_controller
            cc.storage_backend_type = "mooncake"
            cc.prefetch_tokens_occupied = 8
            cache._drain_storage_control_queues_impl(1, 0, 0, 0, {}, False)
            operation.start_time = time.monotonic() - 100
            operation.completed_tokens = 4
            operation.pool_transfers_done = False
            cache.prefetch_stop_policy = "timeout"
            cache.prefetch_timeout_base = 1.0
            cache.prefetch_timeout_per_page = 0.1
            cache.pp_rank = 0
            cache._all_reduce = lambda *args: None
            self.assertTrue(cache._can_terminate_prefetch(operation))
            cc.terminate_prefetch = Mock(return_value=(4, operation.hash_value))
            cache.release_aborted_request(operation.handle)
            self.assertEqual(cache.ongoing_prefetch, {})
            self.assertEqual(cache._restore_cost_records, {})
            self.assertEqual(cache._restore_cost_declined, set())
            self.assertEqual(cc.prefetch_tokens_occupied, 0)
            # Completed prefix goes back once; unread tail remains IO-owned
            # until the native termination ACK, as covered by lifecycle tests.
            self.assertEqual(
                cc.append_host_mem_release.call_args.kwargs["host_indices"].numel(), 4
            )

    def test_shadow_never_rejects_and_observes_actual_restore(self):
        with tempfile.TemporaryDirectory() as d:
            filename = Path(d) / "profile.json"
            filename.write_text(json.dumps(profile("shadow")))
            cache, operation = _cache_mode_hit_fixture(8)
            cache._restore_cost_records = {}
            cache._restore_cost_declined = set()
            cache.restore_cost_gate = RuntimeRestoreCostGate(
                str(filename), "30B", 2, 49152
            )
            cache.cache_controller.storage_backend_type = "mooncake"
            cache._drain_storage_control_queues_impl(1, 0, 0, 0, {}, False)
            self.assertIs(
                cache.cache_controller.prefetch_buffer.get_nowait(), operation
            )
            self.assertFalse(
                cache._restore_cost_records[operation.handle]["should_restore"]
            )
            self.assertEqual(
                cache._restore_cost_records[operation.handle]["action"], "restore"
            )


if __name__ == "__main__":
    unittest.main()
