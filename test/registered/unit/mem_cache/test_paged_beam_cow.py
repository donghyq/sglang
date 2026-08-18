import unittest
from types import SimpleNamespace

import torch

from sglang.srt.constrained.trie_beam_search import (
    TrieBeam,
    TrieBeamExecution,
    TrieBeamGroup,
    TrieBeamRuntime,
    TrieBeamSlotTransition,
)
from sglang.srt.constrained.trie_grammar_backend import TrieGrammarBackend
from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.observability.metrics_collector import SchedulerStats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(2.0, "base-a-test-cpu")


class TestPagedBeamCOW(CustomTestCase):
    def _allocator(self):
        allocator = object.__new__(PagedTokenToKVPoolAllocator)
        allocator.page_size = 4
        allocator.device = "cpu"
        allocator.need_sort = False
        allocator.is_not_in_free_group = True
        allocator.free_group = []
        allocator.release_pages = torch.empty(0, dtype=torch.int64)
        allocator.free_pages = torch.empty(0, dtype=torch.int64)
        allocator.beam_page_refcounts = {}
        allocator.debug_mode = True
        return allocator

    def test_metrics_snapshot_reports_physical_kv_units_and_references(self):
        allocator = self._allocator()
        prefix = torch.tensor([4, 5, 6, 7])
        allocator.register_beam_pages(prefix)
        allocator.fork_shared_prefix(prefix, child_count=2)

        reporter = object.__new__(SchedulerMetricsReporter)
        reporter.scheduler = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        reporter.stats = SchedulerStats()
        reporter._update_beam_kv_lifecycle_stats()

        self.assertEqual(reporter.stats.beam_kv_registered_total, 1)
        self.assertEqual(reporter.stats.beam_kv_released_total, 0)
        self.assertEqual(reporter.stats.beam_kv_live, 1)
        self.assertEqual(reporter.stats.beam_kv_live_references, 3)

    def test_metrics_snapshot_keeps_zero_for_non_beam_allocator(self):
        reporter = object.__new__(SchedulerMetricsReporter)
        reporter.scheduler = SimpleNamespace(token_to_kv_pool_allocator=object())
        reporter.stats = SchedulerStats()

        reporter._update_beam_kv_lifecycle_stats()

        self.assertEqual(reporter.stats.beam_kv_registered_total, 0)
        self.assertEqual(reporter.stats.beam_kv_released_total, 0)
        self.assertEqual(reporter.stats.beam_kv_live, 0)
        self.assertEqual(reporter.stats.beam_kv_live_references, 0)

    def test_fork_then_prune_frees_pages_once(self):
        allocator = self._allocator()
        prefix = torch.tensor([4, 5, 6, 7])
        suffix = torch.tensor([8, 9, 10, 11])
        allocator.register_beam_pages(prefix)
        allocator.register_beam_pages(suffix)
        allocator.fork_shared_prefix(prefix)

        allocator.release_beam_suffix(torch.cat((prefix, suffix)))
        self.assertEqual(allocator.beam_page_refcounts, {1: 1})
        self.assertEqual(allocator.free_pages.tolist(), [2])

        allocator.release_beam_suffix(prefix)
        self.assertEqual(allocator.beam_page_refcounts, {})
        self.assertEqual(sorted(allocator.free_pages.tolist()), [1, 2])
        self.assertEqual(
            allocator.beam_lifecycle_snapshot(),
            {"registered": 2, "released": 2, "live": 0, "live_references": 0},
        )

    def test_page_lifecycle_counts_physical_pages_not_shared_references(self):
        allocator = self._allocator()
        prefix = torch.tensor([4, 5, 6, 7])

        allocator.register_beam_pages(prefix)
        allocator.fork_shared_prefix(prefix, child_count=2)
        self.assertEqual(
            allocator.beam_lifecycle_snapshot(),
            {"registered": 1, "released": 0, "live": 1, "live_references": 3},
        )

        allocator.release_beam_suffix(prefix)
        allocator.release_beam_suffix(prefix)
        allocator.release_beam_suffix(prefix)
        self.assertEqual(
            allocator.beam_lifecycle_snapshot(),
            {"registered": 1, "released": 1, "live": 0, "live_references": 0},
        )

    def test_direct_free_of_beam_page_is_rejected(self):
        allocator = self._allocator()
        indices = torch.tensor([4, 5, 6, 7])
        allocator.register_beam_pages(indices)
        with self.assertRaisesRegex(ValueError, "release_beam_suffix"):
            allocator.free(indices)

    def test_failed_register_or_release_keeps_ownership_unchanged(self):
        allocator = self._allocator()
        page_one = torch.tensor([4, 5, 6, 7])
        page_two = torch.tensor([8, 9, 10, 11])
        allocator.register_beam_pages(page_one)

        with self.assertRaisesRegex(ValueError, "already beam-owned"):
            allocator.register_beam_pages(torch.cat((page_one, page_two)))
        self.assertEqual(allocator.beam_page_refcounts, {1: 1})

        with self.assertRaisesRegex(ValueError, "unregistered"):
            allocator.release_beam_suffix(torch.cat((page_one, page_two)))
        self.assertEqual(allocator.beam_page_refcounts, {1: 1})

    def test_reserved_padding_page_cannot_be_beam_owned(self):
        allocator = self._allocator()
        with self.assertRaisesRegex(ValueError, "page 0 is reserved"):
            allocator.register_beam_pages(torch.tensor([0, 1, 2, 3]))

    def test_fork_is_atomic_and_requires_page_aligned_prefix(self):
        allocator = self._allocator()
        page_one = torch.tensor([4, 5, 6, 7])
        page_two = torch.tensor([8, 9, 10, 11])
        allocator.register_beam_pages(page_one)

        with self.assertRaisesRegex(ValueError, "page boundary"):
            allocator.fork_shared_prefix(page_one[:3])
        with self.assertRaisesRegex(ValueError, "unregistered"):
            allocator.fork_shared_prefix(torch.cat((page_one, page_two)))
        self.assertEqual(allocator.beam_page_refcounts, {1: 1})

        allocator.fork_shared_prefix(page_one, child_count=2)
        self.assertEqual(allocator.beam_page_refcounts, {1: 3})

    def test_request_slots_copy_only_the_shared_prefix(self):
        pool = ReqToTokenPool(
            size=4, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        parent = SimpleNamespace(req_pool_idx=None)
        child_one = SimpleNamespace(req_pool_idx=None)
        child_two = SimpleNamespace(req_pool_idx=None)
        pool.alloc([parent])
        pool.req_to_token[parent.req_pool_idx, :4] = torch.tensor([4, 5, 6, 7])

        slots = pool.fork_beam_slots_from_prefix(
            parent, [child_one, child_two], 4, page_size=4
        )
        self.assertEqual(slots, [2, 3])
        self.assertEqual([child_one.req_pool_idx, child_two.req_pool_idx], [2, 3])
        self.assertEqual(pool.req_to_token[2, :4].tolist(), [4, 5, 6, 7])
        self.assertEqual(pool.req_to_token[3, :4].tolist(), [4, 5, 6, 7])

        with self.assertRaisesRegex(ValueError, "already owns"):
            pool.fork_beam_slots_from_prefix(parent, [child_one], 4, page_size=4)

        with self.assertRaisesRegex(ValueError, "page boundary"):
            pool.fork_beam_slots_from_prefix(parent, [], 3, page_size=4)

    def test_decode_request_slots_copy_the_shared_beam_prefix(self):
        pool = DecodeReqToTokenPool(
            size=2,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
            pre_alloc_size=2,
        )
        parent = SimpleNamespace(req_pool_idx=None)
        child = SimpleNamespace(req_pool_idx=None)
        pool.alloc([parent])
        pool.req_to_token[parent.req_pool_idx, :4] = torch.tensor([4, 5, 6, 7])

        slots = pool.fork_beam_slots_from_prefix(
            parent, [child], 4, page_size=4
        )

        self.assertEqual(slots, [2])
        self.assertEqual(child.req_pool_idx, 2)
        self.assertEqual(pool.req_to_token[2, :4].tolist(), [4, 5, 6, 7])

    def test_runtime_transition_forks_page_aligned_prefix_and_releases_all_pages(self):
        allocator = self._allocator()
        pool = ReqToTokenPool(
            size=4, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        parent_req = SimpleNamespace(req_pool_idx=None)
        pool.alloc([parent_req])
        parent_req_slot = parent_req.req_pool_idx
        parent_kv = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
        pool.req_to_token[parent_req_slot, :8] = parent_kv
        runtime = TrieBeamRuntime(pool, allocator, page_size=4)
        runtime.register_root(0, parent_req, parent_kv)
        children = [
            SimpleNamespace(branch_id=1),
            SimpleNamespace(branch_id=2),
        ]
        runtime.apply_transition(
            TrieBeamSlotTransition(slot_reuses={1: 0}, slot_forks={2: 0}, parent_ids_to_release=[]),
            lambda _beam: SimpleNamespace(req_pool_idx=None),
            children,
            {0: parent_kv[:4]},
            lambda _parent, _child, prefix: torch.cat(
                (prefix, torch.tensor([12, 13, 14, 15]))
            ),
        )

        self.assertEqual(set(runtime.branches), {1, 2})
        self.assertIs(runtime.branches[1].req, parent_req)
        self.assertEqual(
            runtime.branches[2].kv_indices.tolist(), [4, 5, 6, 7, 12, 13, 14, 15]
        )
        self.assertEqual(allocator.beam_page_refcounts, {1: 2, 2: 1, 3: 1})
        self.assertNotEqual(runtime.branches[2].req.req_pool_idx, parent_req_slot)
        self.assertEqual(
            pool.req_to_token[runtime.branches[2].req.req_pool_idx, :8].tolist(),
            [4, 5, 6, 7, 12, 13, 14, 15],
        )

        runtime.release_all()
        self.assertEqual(runtime.branches, {})
        self.assertEqual(allocator.beam_page_refcounts, {})
        self.assertEqual(sorted(allocator.free_pages.tolist()), [1, 2, 3])
        self.assertIn(parent_req_slot, pool.free_slots)

    def test_runtime_rejects_a_forked_mapping_that_reuses_writable_tail_pages(self):
        allocator = self._allocator()
        pool = ReqToTokenPool(
            size=3, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        parent_req = SimpleNamespace(req_pool_idx=None)
        pool.alloc([parent_req])
        parent_kv = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
        pool.req_to_token[parent_req.req_pool_idx, :8] = parent_kv
        runtime = TrieBeamRuntime(pool, allocator, page_size=4)
        runtime.register_root(0, parent_req, parent_kv)

        with self.assertRaisesRegex(ValueError, "private KV pages"):
            runtime.apply_transition(
                TrieBeamSlotTransition(
                    slot_reuses={}, slot_forks={1: 0}, parent_ids_to_release=[]
                ),
                lambda _beam: SimpleNamespace(req_pool_idx=None),
                [SimpleNamespace(branch_id=1)],
                {0: parent_kv[:4]},
                lambda _parent, _child, prefix: torch.cat((prefix, parent_kv[4:])),
            )

    def test_runtime_records_decode_kv_location_and_releases_new_page(self):
        allocator = self._allocator()
        pool = ReqToTokenPool(
            size=2, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        req = SimpleNamespace(req_pool_idx=None)
        pool.alloc([req])
        root_kv = torch.tensor([4, 5, 6, 7])
        runtime = TrieBeamRuntime(pool, allocator, page_size=4)
        runtime.register_root(0, req, root_kv)

        runtime.append_decode_kv_locations([0], torch.tensor([8]))

        self.assertEqual(runtime.branches[0].kv_indices.tolist(), [4, 5, 6, 7, 8])
        self.assertEqual(allocator.beam_page_refcounts, {1: 1, 2: 1})
        runtime.release_all()
        self.assertEqual(allocator.beam_page_refcounts, {})
        self.assertEqual(sorted(allocator.free_pages.tolist()), [1, 2])

    def test_runtime_keeps_prefix_cache_pages_outside_beam_ownership(self):
        allocator = self._allocator()
        pool = ReqToTokenPool(
            size=2, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        req = SimpleNamespace(req_pool_idx=None)
        pool.alloc([req])
        root_kv = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
        runtime = TrieBeamRuntime(pool, allocator, page_size=4)

        # Page 1 is protected by the prefix cache.  It must remain outside
        # Beam reference counting and therefore cannot be reclaimed here.
        runtime.register_root(0, req, root_kv, root_kv[4:])
        self.assertEqual(allocator.beam_page_refcounts, {2: 1})

        runtime.release_all()
        self.assertEqual(allocator.beam_page_refcounts, {})
        self.assertEqual(allocator.free_pages.tolist(), [2])

    def test_nonpaged_runtime_records_decode_kv_location_and_releases_token(self):
        allocator = object.__new__(TokenToKVPoolAllocator)
        allocator.size = 32
        allocator.device = "cpu"
        allocator.need_sort = False
        allocator.beam_token_refcounts = {}
        allocator.clear()
        pool = ReqToTokenPool(
            size=2, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        req = SimpleNamespace(req_pool_idx=None)
        pool.alloc([req])
        root_kv = torch.tensor([4, 5, 6])
        runtime = TrieBeamRuntime(pool, allocator, page_size=1)
        runtime.register_root(0, req, root_kv)

        runtime.append_decode_kv_locations([0], torch.tensor([7]))

        self.assertEqual(allocator.beam_token_refcounts, {4: 1, 5: 1, 6: 1, 7: 1})
        runtime.release_all()
        self.assertEqual(allocator.beam_token_refcounts, {})
        self.assertTrue(set([4, 5, 6, 7]).issubset(set(allocator.free_pages.tolist())))
        self.assertEqual(
            allocator.beam_lifecycle_snapshot(),
            {"registered": 4, "released": 4, "live": 0, "live_references": 0},
        )

    def test_runtime_fork_failure_rolls_back_kv_references_and_request_slots(self):
        allocator = self._allocator()
        pool = ReqToTokenPool(
            size=4, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        parent_req = SimpleNamespace(req_pool_idx=None)
        pool.alloc([parent_req])
        parent_kv = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
        pool.req_to_token[parent_req.req_pool_idx, :8] = parent_kv
        runtime = TrieBeamRuntime(pool, allocator, page_size=4)
        runtime.register_root(0, parent_req, parent_kv)

        original_register = allocator.register_beam_pages
        register_calls = 0

        def fail_second_private_tail(kv_indices):
            nonlocal register_calls
            register_calls += 1
            if register_calls == 2:
                raise RuntimeError("injected private-tail registration failure")
            original_register(kv_indices)

        allocator.register_beam_pages = fail_second_private_tail
        prepared_but_unregistered = []
        created_child_reqs = []
        children = [SimpleNamespace(branch_id=1), SimpleNamespace(branch_id=2)]

        def create_child_req(_beam):
            child_req = SimpleNamespace(req_pool_idx=None)
            created_child_reqs.append(child_req)
            return child_req

        with self.assertRaisesRegex(RuntimeError, "injected private-tail"):
            runtime.apply_transition(
                TrieBeamSlotTransition(
                    slot_reuses={}, slot_forks={1: 0, 2: 0}, parent_ids_to_release=[]
                ),
                create_child_req,
                children,
                {0: parent_kv[:4]},
                lambda _parent, child, prefix: torch.cat(
                    (
                        prefix,
                        torch.tensor(
                            [12, 13, 14, 15]
                            if child.branch_id == 1
                            else [16, 17, 18, 19]
                        ),
                    )
                ),
                lambda child_kv: prepared_but_unregistered.append(child_kv.tolist()),
            )

        self.assertEqual(set(runtime.branches), {0})
        self.assertIs(runtime.branches[0].req, parent_req)
        self.assertEqual(allocator.beam_page_refcounts, {1: 1, 2: 1})
        self.assertEqual(sorted(allocator.free_pages.tolist()), [3])
        # The page-aligned prefix belongs to the parent (and can be protected
        # by the radix cache).  Rollback may return only the child-private
        # tail that was allocated but never registered with Beam ownership.
        self.assertEqual(prepared_but_unregistered, [[16, 17, 18, 19]])
        self.assertEqual(sorted(pool.free_slots), [2, 3, 4])
        self.assertEqual(len(created_child_reqs), 2)
        self.assertIsNone(created_child_reqs[0].req_pool_idx)
        self.assertIsNone(created_child_reqs[1].req_pool_idx)

    def test_execution_commits_group_and_kv_lifecycle_from_one_logits_result(self):
        allocator = self._allocator()
        pool = ReqToTokenPool(
            size=3, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        parent_req = SimpleNamespace(req_pool_idx=None)
        pool.alloc([parent_req])
        parent_kv = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
        pool.req_to_token[parent_req.req_pool_idx, :8] = parent_kv
        backend = TrieGrammarBackend(vocab_size=8)
        self.addCleanup(backend.executor.shutdown, wait=True)
        execution = TrieBeamExecution(
            TrieBeamGroup(backend.dispatch_trie("[[1, 3], [1, 4]]"), width=2),
            TrieBeamRuntime(pool, allocator, page_size=4),
        )
        execution.register_root(parent_req, parent_kv)

        active = execution.advance(
            torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            lambda _beam: SimpleNamespace(req_pool_idx=None),
            {},
            lambda _parent, _child, _prefix: self.fail("first child reuses root slot"),
        )
        self.assertEqual([beam.tokens for beam in active], [[1]])
        self.assertEqual(set(execution.runtime.branches), {1})
        self.assertIs(execution.runtime.branches[1].req, parent_req)

        active = execution.advance(
            torch.tensor([[0.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 0.0]]),
            lambda _beam: SimpleNamespace(req_pool_idx=None),
            {1: parent_kv[:4]},
            lambda _parent, child, prefix: torch.cat(
                (
                    prefix,
                    torch.tensor(
                        [12, 13, 14, 15]
                        if child.tokens[-1] == 4
                        else [16, 17, 18, 19]
                    ),
                )
            ),
        )
        self.assertEqual(active, [])
        self.assertTrue(execution.is_finished)
        self.assertEqual([beam.tokens for beam in execution.results], [[1, 3]])
        self.assertEqual(execution.runtime.branches, {})
        self.assertEqual(allocator.beam_page_refcounts, {})
        self.assertEqual(sorted(allocator.free_pages.tolist()), [1, 2])

    def test_execution_restores_beam_group_when_kv_transition_fails(self):
        allocator = self._allocator()
        pool = ReqToTokenPool(
            size=3, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        parent_req = SimpleNamespace(req_pool_idx=None)
        pool.alloc([parent_req])
        parent_kv = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
        backend = TrieGrammarBackend(vocab_size=8)
        self.addCleanup(backend.executor.shutdown, wait=True)
        execution = TrieBeamExecution(
            TrieBeamGroup(backend.dispatch_trie("[[1, 3], [2, 4]]"), width=2),
            TrieBeamRuntime(pool, allocator, page_size=4),
        )
        execution.register_root(parent_req, parent_kv)

        with self.assertRaisesRegex(ValueError, "Missing shared KV prefix"):
            execution.advance(
                torch.tensor([[0.0, 5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                lambda _beam: SimpleNamespace(req_pool_idx=None),
                {},
                lambda _parent, _child, _prefix: parent_kv,
            )

        self.assertEqual([beam.branch_id for beam in execution.group.active], [0])
        self.assertEqual(execution.group.completed, [])
        self.assertEqual(set(execution.runtime.branches), {0})
        self.assertEqual(allocator.beam_page_refcounts, {1: 1, 2: 1})


class TestTokenBeamCOW(CustomTestCase):
    def _allocator(self):
        allocator = object.__new__(TokenToKVPoolAllocator)
        allocator.device = "cpu"
        allocator.need_sort = False
        allocator.is_not_in_free_group = True
        allocator.free_group = []
        allocator.release_pages = torch.empty(0, dtype=torch.int64)
        allocator.free_pages = torch.empty(0, dtype=torch.int64)
        allocator.beam_token_refcounts = {}
        return allocator

    def test_fork_then_prune_recycles_each_token_after_its_last_reference(self):
        allocator = self._allocator()
        prefix = torch.tensor([4, 5, 6])
        suffix = torch.tensor([7, 8])
        allocator.register_beam_pages(prefix)
        allocator.register_beam_pages(suffix)
        allocator.fork_shared_prefix(prefix)

        allocator.release_beam_suffix(torch.cat((prefix, suffix)))
        self.assertEqual(allocator.beam_token_refcounts, {4: 1, 5: 1, 6: 1})
        self.assertEqual(sorted(allocator.free_pages.tolist()), [7, 8])

        allocator.release_beam_suffix(prefix)
        self.assertEqual(allocator.beam_token_refcounts, {})
        self.assertEqual(sorted(allocator.free_pages.tolist()), [4, 5, 6, 7, 8])

    def test_nonpaged_prefix_allows_any_token_length_and_direct_free_is_rejected(self):
        allocator = self._allocator()
        indices = torch.tensor([4, 5, 6])
        allocator.register_beam_pages(indices)
        allocator.fork_shared_prefix(indices, child_count=2)
        self.assertEqual(allocator.beam_token_refcounts, {4: 3, 5: 3, 6: 3})

        with self.assertRaisesRegex(ValueError, "release_beam_suffix"):
            allocator.free(indices)
        with self.assertRaisesRegex(ValueError, "token 0 is reserved"):
            allocator.register_beam_pages(torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
