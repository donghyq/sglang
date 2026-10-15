import math
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.agent_lifecycle import AgentLifecycle, LifecycleHint
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAgentLifecycleV1(CustomTestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"SGLANG_AGENT_LIFECYCLE_MODE": "v1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.parallel = patch("sglang.srt.runtime_context.get_parallel", return_value=SimpleNamespace(tp_rank=0))
        self.parallel.start()
        self.addCleanup(self.parallel.stop)
        self.clock = patch("sglang.srt.mem_cache.agent_lifecycle.time.time", return_value=1000)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.policy = AgentLifecycle("P", 49152)
        self.cost = dict(valid=True, predicted_restore_ms=400, predicted_recompute_ms=520)

    def ready(self, **kw):
        data = dict(state="READY_SOON", expected_resume_at=1003, remaining_dependencies=0, confidence=1)
        data.update(kw)
        self.policy.update("parent", LifecycleHint(**data))

    def test_no_guess_without_deadline_and_explicit_lead_is_supported(self):
        self.ready(expected_resume_at=None)
        permit, fields = self.policy.admission("parent", 8192, "mooncake", True, self.cost)
        self.assertFalse(permit)
        self.assertEqual(fields["reason"], "missing_deadline")
        self.ready(expected_resume_at=None, resume_lead_ms=3000)
        self.assertEqual(self.policy.hints["parent"].expected_resume_at, 1003)
        self.assertTrue(self.policy.admission("parent", 8192, "mooncake", True, self.cost)[0])

    def test_reject_boundaries_and_bad_hint_does_not_replace_valid_hint(self):
        for changes, complete, capacity, reason in [
            ({"confidence": 0.2}, True, True, "low_confidence"),
            ({"remaining_dependencies": 1}, True, True, "dependencies_not_converged"),
            ({}, False, True, "incomplete_cold_copy"),
            ({}, True, False, "target_capacity"),
            ({"expected_resume_at": 999}, True, True, "no_remaining_wait_benefit"),
            ({"state": "WAITING_INPUT"}, True, True, "state_not_prefetch_candidate"),
        ]:
            with self.subTest(reason=reason):
                self.ready(**changes)
                permit, fields = self.policy.admission("parent", 8192, "mooncake", capacity, self.cost, complete)
                self.assertFalse(permit)
                self.assertEqual(fields["reason"], reason)
        saved = self.policy.hints["parent"]
        with self.assertRaises(ValueError):
            self.policy.update("parent", LifecycleHint("READY_SOON", confidence=math.nan))
        self.assertEqual(self.policy.hints["parent"], saved)

    def test_D_never_uses_P_recompute_cost_and_requires_measured_alternative(self):
        self.policy.role = "D"
        self.ready()
        for _ in range(3):
            self.policy.observe("host", 8192, 20)
        self.assertEqual(self.policy.admission("parent", 8192, "host", True, self.cost)[1]["reason"], "missing_D_history_retransmit_samples")
        for _ in range(3):
            self.policy.observe("pd_history", 8192, 10)
        permit, fields = self.policy.admission("parent", 8192, "host", True, self.cost)
        self.assertTrue(permit)
        self.assertEqual(fields["alternate_estimate_ms"], 10)

    def test_demote_requires_full_host_copy_and_no_active_owner(self):
        ct = "FULL"
        root = SimpleNamespace(parent=None)
        cd = SimpleNamespace(host_value=[1, 2], session_ids={"parent"})
        node = SimpleNamespace(parent=root, id=1, key=[3, 4], component_data={ct: cd})
        self.ready(state="DEPENDENCY_BLOCKED", remaining_dependencies=2)
        self.assertTrue(self.policy.can_demote(["parent"], node, ct, {}, 400))
        for host, pending, owner in [([1], {}, None), ([1, 2], {1: object()}, None), ([1,2], {99:SimpleNamespace(node_id=99,publish_node_ids=[1])}, None), ([1, 2], {}, "active")]:
            cd.host_value = host
            cd.session_ids = {"parent"} | ({owner} if owner else set())
            self.assertFalse(self.policy.can_demote(["parent"], node, ct, pending, 400))

    def test_peer_capacity_rejection_issues_no_H2D(self):
        self.ready()
        self.policy.request("parent", 8192, "host")
        self.policy.attempts["parent"].update(tokens=list(range(8192)))
        def peer_reject(vote):
            vote[0] = 0
        cache = SimpleNamespace(agent_lifecycle=self.policy,
            _agent_match=lambda _: SimpleNamespace(device_indices=torch.empty(0), host_hit_length=8192),
            _component_available_size=lambda _: 8192, _agent_gpu_capacity=lambda:8192,
            _all_reduce_storage_admission=peer_reject, load_back=Mock())
        self.assertFalse(UnifiedRadixCache._agent_warm_host(cache, "parent"))
        cache.load_back.assert_not_called()
        self.assertEqual(self.policy.attempts["parent"]["state"], "CAPACITY_REJECT")

    def test_warm_starts_native_loader_without_a_generation_batch(self):
        self.ready()
        self.policy.request("parent", 8192, "host")
        self.policy.attempts["parent"].update(tokens=array("q",range(8192)))
        event=Mock()
        event.query.return_value=True
        cc=SimpleNamespace(load_queue=[],layer_done_counter=SimpleNamespace(
            producer_index=-1,num_counters=3,events=[SimpleNamespace(finish_event=event)]*3))
        self.policy.loads[7]=dict(session_id=None,tokens=8192,device_indices=torch.arange(8192))
        cache=SimpleNamespace(agent_lifecycle=self.policy,cache_controller=cc,
            _agent_match=lambda _:SimpleNamespace(device_indices=torch.empty(0),host_hit_length=8192,best_match_node=7),
            _agent_gpu_capacity=lambda:8192,_all_reduce_storage_admission=lambda _:None,
            load_back=Mock(return_value=True),ready_to_load_host_cache=Mock(return_value=0))
        self.assertTrue(UnifiedRadixCache._agent_warm_host(cache,"parent"))
        cache.ready_to_load_host_cache.assert_called_once_with()
        self.assertIs(self.policy.gpu_fence,event)

    def test_pending_prefetch_is_fenced_before_a_device_match(self):
        event=Mock()
        event.query.return_value=False
        self.policy.gpu_fence=event
        complete = []
        event.synchronize.side_effect=lambda:complete.append(True)
        result=object()
        def matched(_):
            self.assertTrue(complete, "DMA must finish before exposing a device hit")
            return result
        cache=SimpleNamespace(agent_lifecycle=self.policy,session=SimpleNamespace(try_match_prefix=matched))
        cache._wait_agent_prefetch_fence=lambda:UnifiedRadixCache._wait_agent_prefetch_fence(cache)
        match=getattr(UnifiedRadixCache.match_prefix,"__wrapped__",UnifiedRadixCache.match_prefix)
        self.assertIs(match(cache,object()),result)
        event.synchronize.assert_called_once_with()
        self.assertIsNone(self.policy.gpu_fence)

    def test_native_loader_records_actual_start_not_only_enqueue(self):
        self.policy.request("parent",8192,"host")
        self.policy.loads[7]=dict(session_id="parent",tokens=8192)
        cc=SimpleNamespace(load_queue=[SimpleNamespace(node_ids=[7])],start_loading=Mock(return_value=1))
        cache=SimpleNamespace(agent_lifecycle=self.policy,cache_controller=cc,linker=None)
        self.assertEqual(UnifiedRadixCache.ready_to_load_host_cache(cache),1)
        self.assertEqual(self.policy.attempts["parent"]["state"],"H2D_STARTED")

    def test_old_generation_cannot_complete_new_session_attempt(self):
        self.ready()
        self.policy.request("parent", 8192, "host", generation=2)
        self.assertIsNone(self.policy.session_for_handle(SimpleNamespace(rid="_agent_prefetch_parent", attempt_id=1)))
        self.assertEqual(self.policy.session_for_handle(SimpleNamespace(rid="_agent_prefetch_parent", attempt_id=self.policy.attempts["parent"]["attempt_id"])), "parent")

    def test_retry_handle_cannot_consume_old_async_result(self):
        self.policy.request("parent", 8192, "mooncake", generation=2)
        old = SimpleNamespace(rid="_agent_prefetch_parent", attempt_id=self.policy.attempts["parent"]["attempt_id"])
        self.policy.record("parent", "REJECTED", reason="capacity")
        self.policy.request("parent", 8192, "mooncake", generation=2)
        self.assertIsNone(self.policy.session_for_handle(old))

    def test_host_retention_release_is_idempotent(self):
        receipt = object()
        self.policy.host_retention["parent"] = dict(node_id=7, receipt=receipt, slots={1,2})
        cache = SimpleNamespace(agent_lifecycle=self.policy, dec_host_lock_ref=Mock())
        UnifiedRadixCache._release_agent_host_retention(cache, "parent", "cancelled")
        UnifiedRadixCache._release_agent_host_retention(cache, "parent", "cancelled")
        cache.dec_host_lock_ref.assert_called_once_with(7, receipt)
        self.assertNotIn("parent", self.policy.host_retention)

    def test_missing_deadline_does_not_issue_storage_IO_or_change_KV_key(self):
        self.ready(expected_resume_at=None)
        req = SimpleNamespace(origin_input_ids=list(range(8192)), extra_key="layout", cache_salt="same-namespace")
        native_prefetch = Mock()
        cache = SimpleNamespace(agent_lifecycle=self.policy, enable_storage=True, cache_controller=object(),
            session_refs=SimpleNamespace(ensure_session_generation=lambda _:1),
            _agent_match=lambda _:SimpleNamespace(device_indices=torch.empty(0),host_hit_length=0),
            _agent_gpu_capacity=lambda:8192, _agent_cost_record=lambda _:self.cost,
            _all_reduce_storage_admission=lambda _:None, prefetch_from_storage=native_prefetch)
        self.assertFalse(UnifiedRadixCache._agent_prefetch_candidate(cache, "parent", req))
        native_prefetch.assert_not_called()
        self.assertEqual(req.cache_salt, "same-namespace")
        self.assertEqual(req.extra_key, "layout")
        self.assertEqual(req.origin_input_ids, list(range(8192)))

    def test_candidate_matches_native_token_array_without_conversion(self):
        self.ready(expected_resume_at=None)
        tokens = array("q", range(8192))
        key = RadixKey(tokens, extra_key="layout", cache_salt="namespace")
        def match(params):
            self.assertEqual(key.match(params.key), len(tokens))
            return SimpleNamespace(device_indices=torch.empty(0), host_hit_length=0)
        cache = SimpleNamespace(agent_lifecycle=self.policy, enable_storage=True, cache_controller=object(),
            session_refs=SimpleNamespace(ensure_session_generation=lambda _:1), match_prefix=match,
            _agent_gpu_capacity=lambda:8192, _agent_cost_record=lambda _:self.cost,
            _all_reduce_storage_admission=lambda _:None, prefetch_from_storage=Mock())
        cache._agent_match=lambda sid:UnifiedRadixCache._agent_match(cache,sid)
        req=SimpleNamespace(origin_input_ids=tokens, extra_key="layout", cache_salt="namespace")
        self.assertFalse(UnifiedRadixCache._agent_prefetch_candidate(cache,"parent",req))
        cache.prefetch_from_storage.assert_not_called()

    def test_demote_matches_native_history_and_output_arrays(self):
        self.ready(state="DEPENDENCY_BLOCKED",remaining_dependencies=1)
        root=SimpleNamespace(parent=None)
        key=RadixKey(array("q",[1,2,3]),extra_key="layout",cache_salt="namespace")
        def match(params):
            self.assertEqual(key.match(params.key),3)
            return SimpleNamespace(best_match_node=0)
        cache=SimpleNamespace(agent_lifecycle=self.policy,match_prefix=match,
            tree_core=SimpleNamespace(node_by_id=lambda _:root,evictable_device_leaves=set()),
            _agent_cost_record=lambda _:self.cost,_component_available_size=lambda _:1024,
            cache_controller=SimpleNamespace(mem_pool_host=SimpleNamespace(anchor_entry=SimpleNamespace(host_pool=SimpleNamespace(size=1024)))),
            _all_reduce_storage_admission=lambda _:None)
        req=SimpleNamespace(origin_input_ids=array("q",[1,2]),output_ids=array("q",[3]),extra_key="layout",cache_salt="namespace")
        self.assertFalse(UnifiedRadixCache.demote_agent_session(cache,"parent",req))

    def test_completed_radix_tag_keeps_native_key_type_without_request_handles(self):
        req = SimpleNamespace(origin_input_ids=array("q", [1, 2]),
                              output_ids=array("q", [3]), extra_key="layout",
                              cache_salt="tenant", session_generation=2)
        self.policy.remember_request("parent", req, 2)
        req.origin_input_ids.append(99)
        saved = self.policy.completed_request("parent", 2)
        self.assertEqual(RadixKey(array("q", [1, 2]), extra_key="layout", cache_salt="tenant").match(
            RadixKey(saved.origin_input_ids, extra_key=saved.extra_key, cache_salt=saved.cache_salt)), 2)
        self.assertEqual(saved.output_ids, array("q", [3]))
        self.assertFalse(hasattr(saved, "last_node"))
        saved.origin_input_ids.append(88)
        self.assertEqual(self.policy.completed_request("parent", 2).origin_input_ids, array("q", [1, 2]))

    def test_completed_radix_tag_rejects_old_generation_and_native_mode(self):
        req = SimpleNamespace(origin_input_ids=array("q", [1]), output_ids=array("q"),
                              extra_key=None, cache_salt=None, session_generation=1)
        self.policy.remember_request("parent", req, 2)
        self.assertIsNone(self.policy.completed_request("parent", 2))
        self.policy.remember_request("parent", req, 1)
        self.assertIsNone(self.policy.completed_request("parent", 2))
        self.policy.update("parent", LifecycleHint("COMPLETED"))
        self.assertIsNone(self.policy.completed_request("parent", 1))
        self.policy.mode = "native"
        self.policy.remember_request("parent", req, 1)
        self.assertFalse(self.policy.completed_requests)

    def test_hint_uses_completed_explicit_history_without_native_append_nodes(self):
        from sglang.srt.session.session_controller import Session, SessionController
        req = SimpleNamespace(origin_input_ids=array("q", [1, 2]), output_ids=array("q", [3]),
                              extra_key="layout", cache_salt="tenant", session_generation=2)
        self.policy.remember_request("parent", req, 2)
        cache = SimpleNamespace(agent_lifecycle=self.policy, set_session_lifecycle=Mock(),
            get_session_lifecycle_request=lambda sid: self.policy.completed_request(sid, 2),
            demote_agent_session=Mock(), prefetch_session=Mock(return_value=True))
        controller = SessionController(cache)
        controller.sessions["parent"] = Session(capacity_of_str_len=100)
        hint = SimpleNamespace(session_id="parent", state="READY_SOON", expected_resume_at=1003,
                               resume_lead_ms=None, remaining_dependencies=0, confidence=1, reason="test")
        self.assertTrue(controller.update_hint(hint))
        sid, saved = cache.prefetch_session.call_args.args
        self.assertEqual(sid, "parent")
        self.assertEqual(saved.origin_input_ids, array("q", [1, 2]))
        self.assertFalse(controller.sessions["parent"].req_nodes)
        hint.state, hint.remaining_dependencies = "DEPENDENCY_BLOCKED", 2
        controller.update_hint(hint)
        cache.demote_agent_session.assert_called_once()
        self.assertEqual(cache.prefetch_session.call_count, 1)

    def test_radix_close_discards_completed_hint_context(self):
        self.policy.completed_requests["parent"] = object()
        refs = SimpleNamespace(release_radix_session=Mock(return_value=0))
        cache = SimpleNamespace(agent_lifecycle=self.policy, session_refs=refs)
        self.assertEqual(UnifiedRadixCache.release_radix_session(cache, "parent"), 0)
        self.assertNotIn("parent", self.policy.completed_requests)
        refs.release_radix_session.assert_called_once_with("parent")

    def test_consumption_counts_physical_intersection_and_eviction(self):
        self.ready()
        self.policy.request("parent", 3, "host")
        attempt = self.policy.attempts["parent"]
        attempt.update(gpu_indices={10, 11, 12}, resume_at=1001)
        self.policy.record("parent", "GPU_READY")
        self.policy.evicted(torch.tensor([12, 99]))
        self.assertEqual(attempt["evicted_bytes"], 49152)
        self.policy.consume("parent", torch.tensor([10, 11, 99]))
        self.assertEqual(attempt["state"], "CONSUMED")
        self.assertEqual(attempt["useful_bytes"], 2 * 49152)
        self.assertTrue(attempt["ready_before_resume"])


if __name__ == "__main__":
    unittest.main()
