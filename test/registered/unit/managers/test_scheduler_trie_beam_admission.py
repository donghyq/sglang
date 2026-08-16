# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.constrained.trie_grammar_backend import TrieGrammar
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerTrieBeamAdmission(unittest.TestCase):
    def _scheduler(self, *, disaggregation_mode=DisaggregationMode.NULL, spec_none=True):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = disaggregation_mode
        scheduler.enable_overlap = False
        scheduler.trie_beam_executions = {}
        scheduler.spec_algorithm = MagicMock()
        scheduler.spec_algorithm.is_none.return_value = spec_none
        return scheduler

    @staticmethod
    def _req(beam_width=2, stream=False):
        return SimpleNamespace(
            sampling_params=SimpleNamespace(
                beam_width=beam_width, num_return_sequences=1
            ),
            stream=stream,
        )

    @staticmethod
    def _recv(*, session_id=None, mm_inputs=None):
        return SimpleNamespace(session_id=session_id, mm_inputs=mm_inputs)

    def test_non_beam_request_is_not_rejected(self):
        self.assertIsNone(
            self._scheduler()._trie_beam_request_error(
                self._req(beam_width=1), self._recv(), None
            )
        )

    def test_beam_request_rejects_unsupported_modes_before_general_guard(self):
        self.assertIn(
            "sessions",
            self._scheduler()._trie_beam_request_error(
                self._req(), self._recv(session_id="session"), None
            ),
        )
        self.assertIn(
            "streaming",
            self._scheduler()._trie_beam_request_error(
                self._req(stream=True), self._recv(), None
            ),
        )
        self.assertIn(
            "multimodal",
            self._scheduler()._trie_beam_request_error(
                self._req(), self._recv(mm_inputs=object()), None
            ),
        )
        self.assertIsNone(
            self._scheduler(disaggregation_mode=DisaggregationMode.PREFILL)._trie_beam_request_error(
                self._req(), self._recv(), None
            )
        )
        self.assertIn(
            "speculative",
            self._scheduler(spec_none=False)._trie_beam_request_error(
                self._req(), self._recv(), None
            ),
        )

    def test_beam_request_is_admitted_for_the_isolated_scheduler_path(self):
        self.assertIsNone(
            self._scheduler()._trie_beam_request_error(self._req(), self._recv(), None)
        )

    def test_pd_handoff_rejects_a_beam_width_that_exceeds_metadata_capacity(self):
        req = self._req(beam_width=16)
        self.assertIn(
            "16 first-step candidates",
            self._scheduler(
                disaggregation_mode=DisaggregationMode.DECODE
            )._trie_beam_request_error(req, self._recv(), None),
        )

    def test_beam_request_rejects_overlap_and_multi_return(self):
        overlap = self._scheduler()
        overlap.enable_overlap = True
        self.assertIn("overlap", overlap._trie_beam_request_error(self._req(), self._recv(), None))

        multi_return = self._scheduler()
        req = self._req()
        req.sampling_params.num_return_sequences = 2
        self.assertIn("num_return_sequences", multi_return._trie_beam_request_error(req, self._recv(), None))

    def test_trie_beam_batch_rejects_mixed_internal_and_normal_requests(self):
        beam = self._req(beam_width=2)
        normal = self._req(beam_width=1)

        self.assertTrue(ScheduleBatch(reqs=[beam]).is_trie_beam_batch)
        self.assertFalse(ScheduleBatch(reqs=[beam, normal]).is_trie_beam_batch)
        self.assertFalse(ScheduleBatch(reqs=[normal]).is_trie_beam_batch)

    def test_trie_beam_prefill_batch_accepts_multiple_external_roots(self):
        self.assertTrue(
            ScheduleBatch(reqs=[self._req(beam_width=2), self._req(beam_width=3)]).is_trie_beam_batch
        )

    def test_trie_beam_root_request_is_identified_before_decode_markers_exist(self):
        trie_root = self._req(beam_width=2)
        trie_root.grammar = TrieGrammar(({32: 1}, {}), (False, True))
        normal = self._req(beam_width=1)
        normal.grammar = trie_root.grammar

        self.assertTrue(Scheduler._is_trie_beam_root_request(trie_root))
        self.assertFalse(Scheduler._is_trie_beam_root_request(normal))

    def test_trie_beam_decode_batch_accepts_multiple_independent_roots(self):
        branch_a = self._req(beam_width=2)
        branch_b = self._req(beam_width=2)
        branch_a.trie_beam_root_rid = "root"
        branch_b.trie_beam_root_rid = "root"
        self.assertTrue(ScheduleBatch(reqs=[branch_a, branch_b]).is_trie_beam_batch)

        branch_b.trie_beam_root_rid = "other-root"
        self.assertTrue(ScheduleBatch(reqs=[branch_a, branch_b]).is_trie_beam_batch)

    def test_completed_trie_beam_batch_can_be_cleared_before_idle(self):
        batch = ScheduleBatch(reqs=[self._req(beam_width=2)])

        # The scheduler clears the just-completed internal Beam batch before
        # assigning it to ``last_batch``.  This ensures health checks and idle
        # housekeeping do not treat released branch requests as active work.
        batch.filter_batch(keep_indices=[])

        self.assertTrue(batch.is_empty())

    def test_trie_beam_decode_does_not_leave_the_normal_batch_marked_full(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.trie_beam_executions = {"root": object()}
        scheduler._build_trie_beam_decode_batch = MagicMock(
            return_value=ScheduleBatch(reqs=[self._req(beam_width=2)])
        )
        scheduler.process_pending_chunked_abort = MagicMock()

        plan = scheduler.get_next_batch_to_run(
            running_batch=ScheduleBatch(reqs=[]), last_batch=None
        )

        self.assertFalse(plan.running_batch.batch_is_full)
        self.assertTrue(plan.running_batch.is_empty())

    def test_trie_beam_handoff_waits_for_an_existing_normal_decode_batch(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.trie_beam_executions = {"root": object()}
        scheduler.get_new_prebuilt_batch = MagicMock()
        scheduler.update_running_batch = MagicMock(
            return_value=ScheduleBatch(reqs=[self._req(beam_width=1)])
        )
        scheduler.dp_attn_adapter = MagicMock()
        scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda x: x

        normal_batch = ScheduleBatch(reqs=[self._req(beam_width=1)])
        plan = scheduler.get_next_disagg_decode_batch_to_run(normal_batch)

        scheduler.get_new_prebuilt_batch.assert_not_called()
        scheduler.update_running_batch.assert_called_once_with(normal_batch)
        self.assertIs(plan.batch_to_run, scheduler.update_running_batch.return_value)
        self.assertIs(plan.running_batch, scheduler.update_running_batch.return_value)

    def test_pending_trie_handoff_waits_instead_of_crashing_during_active_execution(self):
        scheduler = self._scheduler(disaggregation_mode=DisaggregationMode.DECODE)
        handoff = SimpleNamespace(trie_beam_handoff_candidates=[])
        scheduler.waiting_queue = [handoff]
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.trie_beam_executions = {"active-root": object()}

        result = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
            scheduler, ScheduleBatch(reqs=[])
        )

        self.assertIsNone(result)
        self.assertEqual(scheduler.waiting_queue, [handoff])

    def test_private_kv_suffix_is_copied_after_page_aligned_prefix(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.page_size = 4
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.token_to_kv_pool_allocator.alloc.return_value = torch.tensor(
            [20, 21, 22, 23]
        )
        scheduler.token_to_kv_pool = MagicMock()
        parent = SimpleNamespace(kv_indices=torch.tensor([4, 5, 6, 7, 8, 9]))

        mapping = scheduler._fork_trie_beam_kv_mapping(
            parent, SimpleNamespace(), torch.tensor([4, 5, 6, 7])
        )

        self.assertEqual(mapping.tolist(), [4, 5, 6, 7, 20, 21])
        scheduler.token_to_kv_pool_allocator.alloc.assert_called_once_with(4)
        scheduler.token_to_kv_pool.move_kv_cache.assert_called_once()
        target, source = scheduler.token_to_kv_pool.move_kv_cache.call_args.args
        self.assertEqual(target.tolist(), [20, 21])
        self.assertEqual(source.tolist(), [8, 9])

    def test_private_kv_suffix_is_freed_when_copy_fails(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.page_size = 4
        scheduler.token_to_kv_pool_allocator = MagicMock()
        private_page = torch.tensor([20, 21, 22, 23])
        scheduler.token_to_kv_pool_allocator.alloc.return_value = private_page
        scheduler.token_to_kv_pool = MagicMock()
        scheduler.token_to_kv_pool.move_kv_cache.side_effect = RuntimeError("copy failed")
        parent = SimpleNamespace(kv_indices=torch.tensor([4, 5, 6, 7, 8]))

        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            scheduler._fork_trie_beam_kv_mapping(
                parent, SimpleNamespace(), torch.tensor([4, 5, 6, 7])
            )

        scheduler.token_to_kv_pool_allocator.free.assert_called_once_with(private_page)

    def test_unregistered_child_cleanup_keeps_shared_beam_pages(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.page_size = 4
        scheduler.token_to_kv_pool_allocator = MagicMock(
            beam_page_refcounts={1: 2}
        )

        scheduler._release_unregistered_trie_beam_kv_mapping(
            torch.tensor([4, 5, 6, 7, 20, 21])
        )

        scheduler.token_to_kv_pool_allocator.free.assert_called_once()
        self.assertEqual(
            scheduler.token_to_kv_pool_allocator.free.call_args.args[0].tolist(),
            [20, 21],
        )

    def test_abort_active_trie_beam_reclaims_branches_and_root_cache_lock(self):
        """Cancellation must release Beam-owned KV and the root prefix lock."""
        scheduler = self._scheduler()
        root_cache_node = object()
        root_req = SimpleNamespace(rid="trie-root", last_node=root_cache_node)
        execution = MagicMock()
        scheduler.trie_beam_executions = {
            root_req.rid: SimpleNamespace(root_req=root_req, execution=execution)
        }
        scheduler.tree_cache = MagicMock()
        scheduler.ipc_channels = MagicMock()
        scheduler.chunked_req = None
        scheduler.partial_rollout_paused_queue = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = MagicMock()
        scheduler.ps = SimpleNamespace(pp_size=1)
        scheduler.running_batch = None
        scheduler.last_batch = None

        scheduler.abort_request(AbortReq(rid=root_req.rid))

        execution.release_all.assert_called_once_with()
        scheduler.tree_cache.dec_lock_ref.assert_called_once_with(root_cache_node)
        self.assertIsNone(root_req.last_node)
        self.assertNotIn(root_req.rid, scheduler.trie_beam_executions)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()

    def test_missing_trie_logits_marker_uses_normal_result_processor(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.publish_load_snapshot = MagicMock()
        scheduler._process_trie_beam_logits = MagicMock()
        scheduler.batch_result_processor = MagicMock()
        scheduler.metrics_reporter = MagicMock()
        scheduler.enable_fpm = False
        scheduler._maybe_clear_mm_inputs = MagicMock()
        scheduler.maybe_send_health_check_signal = MagicMock()

        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: False,
                is_decode=lambda: True,
                is_prebuilt=lambda: False,
                is_idle=lambda: False,
            )
        )
        result = GenerationBatchResult()

        # Reproduce a result deserialized by a producer that predates the
        # optional Trie coordination marker.
        marker = GenerationBatchResult.__dict__.get(
            "is_trie_beam_logits", NotImplemented
        )
        if marker is not NotImplemented:
            del GenerationBatchResult.is_trie_beam_logits
        try:
            scheduler.process_batch_result(batch, result)
        finally:
            if marker is not NotImplemented:
                GenerationBatchResult.is_trie_beam_logits = marker

        scheduler._process_trie_beam_logits.assert_not_called()
        scheduler.batch_result_processor.process_batch_result_decode.assert_called_once_with(
            batch, result
        )


if __name__ == "__main__":
    unittest.main()
