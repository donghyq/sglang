import json
import inspect
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.agent_lifecycle import AgentLifecycle, LifecycleHint
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode
from sglang.srt.session.session_controller import Session, SessionController
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDAGKVPolicy(CustomTestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"SGLANG_AGENT_LIFECYCLE_MODE": "v1"})
        env.start()
        self.addCleanup(env.stop)
        parallel = patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(tp_rank=0),
        )
        parallel.start()
        self.addCleanup(parallel.stop)
        clock = patch(
            "sglang.srt.mem_cache.agent_lifecycle.time.time", return_value=1000.0
        )
        clock.start()
        self.addCleanup(clock.stop)
        self.policy = AgentLifecycle("P", 49152)
        self.ct = ComponentType.FULL

    def _configure(self, mode="eviction", **extra):
        event = dict(
            kind="configure",
            session_id="control",
            workflow_id="control",
            epoch=0,
            seq=0,
            logical_time=1000.0,
            mode=mode,
            **extra,
        )
        return self.policy.handle_dag_event(event, idle=True)

    def _consumer(self, consumer_id="consumer", session_id="session", seq=1, **extra):
        event = dict(
            kind="consumer_upsert",
            workflow_id="workflow",
            epoch=0,
            seq=seq,
            logical_time=1000.0,
            consumer_id=consumer_id,
            session_id=session_id,
            step_id=consumer_id,
            prefix_tokens=8,
            next_use_distance=2,
            status="pending",
            confidence=0.9,
            expected_resume_at=1003.0,
            ttl_ms=60000,
        )
        event.update(extra)
        return self.policy.handle_dag_event(event)

    def _node(self, parent, tokens, slots=None, key_start=0):
        node = UnifiedTreeNode((self.ct,))
        node.parent = parent
        node.key = RadixKey(
            array("q", range(key_start, key_start + tokens)), extra_key="layout"
        )
        node.component_data[self.ct].value = torch.tensor(
            slots if slots is not None else list(range(tokens)), dtype=torch.int64
        )
        if parent is not None:
            parent.children[node.key.child_key(1)] = node
        return node

    def _component(self, leaves=(), root=None):
        component = FullComponent.__new__(FullComponent)
        component.cache = SimpleNamespace(
            session_refs=SimpleNamespace(
                get_session_lifecycle=lambda _session_id: None
            ),
            agent_lifecycle=self.policy,
        )
        component.tree_core = SimpleNamespace(
            eviction_strategy=SimpleNamespace(get_priority=lambda _node: 0),
            evictable_device_leaves=set(leaves),
            root_node=root,
        )
        component.component_type = self.ct
        component._session_leaves = {}
        component.session_ref_eviction_strategy = component._session_ref_eviction_strategy
        nodes = [root] if root is not None else list(leaves)
        total = 0
        while nodes:
            current = nodes.pop()
            value = current.component_data[self.ct].value
            total += len(value) if value is not None else 0
            nodes.extend(current.children.values())
        component.tree_core.component_evictable_size_ = {self.ct: total}
        return component

    def test_off_keeps_bed4_eviction_key_and_rejects_state_events(self):
        root = UnifiedTreeNode((self.ct,))
        node = self._node(root, 2)
        component = self._component([node], root)
        self.assertEqual(
            component._session_ref_eviction_strategy(node), (True, False, 0, 0)
        )
        accepted, reason = self._consumer()
        self.assertFalse(accepted)
        self.assertEqual(reason, "dag_mode_off")

    def test_native_match_observation_uses_actual_device_hit_not_host_hit(self):
        root = UnifiedTreeNode((self.ct,))
        root.key = RadixKey(array("q"), extra_key="layout")
        gpu_node = self._node(root, 2)
        host_node = self._node(gpu_node, 4, key_start=2)
        component = self._component([host_node], root)
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.agent_lifecycle = self.policy
        cache._wait_agent_prefetch_fence = lambda: None
        cache.session = SimpleNamespace(try_match_prefix=lambda _params: None)
        cache.disable = False
        cache.components = {self.ct: component}
        cache._components_tuple = ()
        cache.linker = None
        cache._apply_cache_actions = lambda _actions: None
        nodes = {node.id: node for node in (root, gpu_node, host_node)}
        result = MatchResult(
            device_indices=torch.tensor([0, 1]), last_device_node=gpu_node.id,
            last_host_node=host_node.id, best_match_node=host_node.id,
            host_hit_length=4, full_kv_hit_length=6,
        )
        cache.tree_core = SimpleNamespace(match_prefix=lambda _params: result,
                                         node_by_id=nodes.__getitem__)
        params = MatchPrefixParams(key=RadixKey(array("q", range(6))),
                                   req=SimpleNamespace(session_id="session"))
        with patch.object(self.policy, "emit") as emit:
            matched = inspect.unwrap(UnifiedRadixCache.match_prefix)(cache, params)
        self.assertIs(matched, result)
        self.assertEqual(emit.call_args.kwargs["matched_tokens"], 2)
        self.assertEqual(emit.call_args.kwargs["stable_key"], component._dag_stable_key(gpu_node))
        # A full Host-only match has no installed GPU prefix to report.
        result = result._replace(device_indices=torch.tensor([], dtype=torch.int64),
                                 last_device_node=root.id, full_kv_hit_length=6)
        with patch.object(self.policy, "emit") as emit:
            inspect.unwrap(UnifiedRadixCache.match_prefix)(cache, params)
        emit.assert_not_called()

    def test_configure_only_at_idle_boundary_and_explicit_stable_history_flag(self):
        cache = SimpleNamespace(
            agent_lifecycle=self.policy,
            _all_reduce_storage_admission=lambda _vote: None,
        )
        event = dict(
            kind="configure",
            session_id="control",
            workflow_id="control",
            epoch=0,
            seq=0,
            logical_time=1000.0,
            mode="eviction",
        )
        self.assertTrue(UnifiedRadixCache.dag_configure_event(cache, event, True)[0])
        self.assertEqual(self.policy.dag_mode, "eviction")
        self.assertFalse(self.policy.dag_stable_history_prepare)
        event.update(
            dict(seq=1, mode="eviction_prefetch", stable_history_prepare=True)
        )
        accepted, reason = UnifiedRadixCache.dag_configure_event(cache, event, False)
        self.assertFalse(accepted)
        self.assertEqual(reason, "not_idle_window")
        self.assertEqual(self.policy.dag_mode, "eviction")
        mismatch_cache = SimpleNamespace(
            agent_lifecycle=self.policy,
            _all_reduce_storage_admission=lambda vote: vote.__setitem__(
                1, (int(vote[1]) + 1) % (1 << 62)
            ),
        )
        accepted, reason = UnifiedRadixCache.dag_configure_event(
            mismatch_cache, event, True
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "peer_configure_signature_mismatch")
        self.assertEqual(self.policy.dag_mode, "eviction")
        accepted, reason = UnifiedRadixCache.dag_configure_event(cache, event, True)
        self.assertTrue(accepted)
        self.assertTrue(self.policy.dag_stable_history_prepare)
        self.assertTrue(self._configure("off")[0])
        self.assertEqual(self.policy.dag_mode, "off")
        self.assertFalse(self.policy.dag_stable_history_prepare)

    def test_configure_rejects_empty_req_nodes_when_scheduler_is_busy(self):
        vote_cache = SimpleNamespace(
            agent_lifecycle=self.policy,
            _all_reduce_storage_admission=lambda _vote: None,
        )
        tree_cache = SimpleNamespace(
            agent_lifecycle=self.policy,
            dag_switch_is_safe=lambda: True,
            dag_configure_event=lambda event, idle: UnifiedRadixCache.dag_configure_event(
                vote_cache, event, idle
            ),
        )
        controller = SessionController(
            tree_cache, scheduler_idle=lambda: False
        )
        controller.sessions["empty-history"] = Session(capacity_of_str_len=100)
        self.assertEqual(controller.sessions["empty-history"].req_nodes, {})

        recv_req = SimpleNamespace(
            session_id="control",
            dag_event=dict(
                kind="configure",
                session_id="control",
                workflow_id="control",
                epoch=0,
                seq=20,
                logical_time=1000.0,
                mode="eviction",
            ),
        )
        success, reason, mode, _seq, prefetch_started, _snapshot = (
            controller.handle_dag_event(recv_req)
        )
        self.assertFalse(success)
        self.assertEqual(reason, "not_idle_window")
        self.assertEqual(mode, "off")
        self.assertFalse(prefetch_started)
        self.assertEqual(self.policy.dag_mode, "off")

    def test_parallel_frontier_and_farther_use_evicts_before_nearer_use(self):
        self.assertTrue(self._configure("eviction")[0])
        self.assertTrue(self._consumer("left", "left", 1, next_use_distance=5))
        self.assertTrue(self._consumer("right", "right", 2, next_use_distance=2))
        self.assertEqual(
            {h.consumer_id for h in self.policy.dag_hints_for_session("left")},
            {"left"},
        )
        self.assertEqual(
            {h.consumer_id for h in self.policy.dag_hints_for_session("right")},
            {"right"},
        )

        root = UnifiedTreeNode((self.ct,))
        far = self._node(root, 2, key_start=0)
        near = self._node(root, 2, key_start=2)
        component = self._component([far, near], root)
        component._session_leaves["far"] = {far}
        component._session_leaves["near"] = {near}
        self.assertTrue(self._consumer("far", "far", 3, next_use_distance=8))
        self.assertTrue(self._consumer("near", "near", 4, next_use_distance=1))
        component._evict_device_start(2)
        self.assertEqual(
            component._evict_device_next_node({self.ct: 0}, {}, {}), far.id
        )

    def test_split_and_single_node_prefix_coverage_is_exact(self):
        self.assertTrue(self._configure("eviction")[0])
        self.assertTrue(self._consumer(prefix_tokens=8))
        root = UnifiedTreeNode((self.ct,))
        shared = self._node(root, 8, key_start=0)
        private = self._node(shared, 4, key_start=8)
        component = self._component([private], root)
        component._session_leaves["session"] = {private}
        self.assertEqual(component._dag_hints_for_node(shared)[2], 1)
        self.assertEqual(component._dag_hints_for_node(private)[2], 0)

        single = UnifiedTreeNode((self.ct,))
        node12k = self._node(single, 12, key_start=0)
        component.tree_core.evictable_device_leaves = {node12k}
        component._session_leaves["session"] = {node12k}
        self.assertEqual(component._dag_hints_for_node(node12k)[2], 0)
        self.assertEqual(component._dag_stable_key(private), component._dag_stable_key(node12k))
        before = component._dag_stable_key(private)
        private.hash_value = ["storage-hash-materialized-later"]
        self.assertEqual(component._dag_stable_key(private), before)

    def test_lock_and_inflight_io_are_hard_constraints(self):
        self.assertTrue(self._configure("eviction")[0])
        self.assertTrue(self._consumer())
        root = UnifiedTreeNode((self.ct,))
        locked = self._node(root, 2)
        loading = self._node(root, 2, key_start=2)
        normal = self._node(root, 2, key_start=4)
        locked.component_data[self.ct].lock_ref = 1
        loading.load_back_pending_id = 77
        component = self._component([locked, loading, normal], root)
        component._session_leaves["session"] = {normal}
        component._evict_device_start(3)
        self.assertEqual(
            component._evict_device_next_node({self.ct: 0}, {}, {}), normal.id
        )

    def test_real_victim_order_changes_and_actual_free_is_observed(self):
        root = UnifiedTreeNode((self.ct,))
        hinted = self._node(root, 8, slots=list(range(8)))
        shared = self._node(root, 8, slots=list(range(8, 16)), key_start=8)
        ordinary = self._node(shared, 24, slots=list(range(16, 40)), key_start=16)
        component = self._component([ordinary, hinted], root)
        component._session_leaves["session"] = {hinted}
        component.tree_core.eviction_strategy.get_priority = (
            lambda node: 0 if node is hinted else 1
        )

        component._evict_device_start(2)
        self.assertEqual(
            component._evict_device_next_node({self.ct: 0}, {}, {}), hinted.id
        )

        self.assertTrue(self._configure("eviction")[0])
        self.assertTrue(self._consumer(prefix_tokens=8))
        # This ancestor has a separate, genuinely covering future consumer.
        component._session_leaves["shared-session"] = {shared}
        self.assertTrue(self._consumer(
            "shared-consumer", "shared-session", 2,
            prefix_tokens=8, next_use_distance=2,
        )[0])
        self.policy.emit = Mock()
        component._evict_device_start(2)
        self.assertEqual(
            component._evict_device_next_node({self.ct: 0}, {}, {}), ordinary.id
        )
        plan_call = next(
            call
            for call in self.policy.emit.call_args_list
            if call.args[1] == "NATIVE_EVICTION_PLAN"
        )
        self.assertNotEqual(
            plan_call.kwargs["native_candidate_order"],
            plan_call.kwargs["dag_candidate_order"],
        )
        device_frees = {self.ct: []}
        host_frees = {self.ct: []}
        component.cache._pending_native_victim_frees = []
        component.evict_component(ordinary, device_frees, host_frees)
        # Full defers this clear to the native cascade, after sibling hooks.
        ordinary.component_data[self.ct].value = None
        # After the child enters the native free path, its parent can join the
        # frontier. The parent receives the same shared-consumer coverage and
        # protection-budget accounting, rather than losing all hints.
        component.tree_core.evictable_device_leaves.add(shared)
        component._evict_device_next_node({self.ct: 0}, {}, {})
        self.assertEqual(component._dag_raw_hints_for_node(shared)[0], 2)
        self.assertEqual(component._dag_candidate_tokens, 40)
        self.assertEqual(component._dag_protected_token_budget, 10)
        event_names = [call.args[1] for call in self.policy.emit.call_args_list]
        self.assertIn("NATIVE_VICTIM_SELECTED", event_names)
        self.assertNotIn("NATIVE_VICTIM_FREED", event_names)
        self.assertEqual(len(device_frees[self.ct]), 1)
        free_cache = SimpleNamespace(
            agent_lifecycle=self.policy,
            _pending_native_victim_frees=component.cache._pending_native_victim_frees,
            _drain_device_frees=lambda frees: frees.clear(),
            _drain_host_frees=lambda frees: frees.clear(),
        )
        UnifiedRadixCache._free_values(free_cache, device_frees, host_frees)
        event_names = [call.args[1] for call in self.policy.emit.call_args_list]
        self.assertIn("NATIVE_VICTIM_FREED", event_names)
        self.assertEqual(component.cache._pending_native_victim_frees, [])

    def test_dependency_complete_requires_declared_all_frontier(self):
        self.assertTrue(self._configure("eviction")[0])
        self.assertTrue(
            self._consumer(
                "parent",
                "session",
                1,
                required_dependencies=2,
                completed_dependencies=0,
                dependency_mode="ALL",
            )
        )
        self.assertTrue(
            self.policy.handle_dag_event(
                dict(kind="dependency_complete", workflow_id="workflow", epoch=0,
                     seq=2, logical_time=1000.0, consumer_id="parent", session_id="control")
            )[0]
        )
        hint = self.policy.dag_workflows["workflow"].consumers["parent"]
        self.assertEqual(hint.status, "pending")
        self.assertEqual(hint.completed_dependencies, 1)
        self.assertTrue(
            self.policy.handle_dag_event(
                dict(kind="dependency_complete", workflow_id="workflow", epoch=0,
                     seq=3, logical_time=1000.0, consumer_id="parent", session_id="control")
            )[0]
        )
        self.assertEqual(hint.status, "ready")
        self.assertEqual(hint.completed_dependencies, 2)

        self.assertTrue(self._consumer("child", "session", 4))
        self.assertTrue(
            self.policy.handle_dag_event(
                dict(kind="step_complete", workflow_id="workflow", epoch=0,
                     seq=5, logical_time=1000.0, consumer_id="child", session_id="control")
            )[0]
        )
        self.assertEqual(
            self.policy.dag_workflows["workflow"].consumers["child"].status,
            "completed",
        )
        self.assertEqual(hint.status, "ready")

    def test_cancel_reset_ttl_and_stale_seq_are_idempotent(self):
        self.assertTrue(self._configure("eviction")[0])
        self.assertTrue(self._consumer(seq=2))
        saved = self.policy.dag_workflows["workflow"].consumers["consumer"]
        self.assertFalse(self._consumer(seq=1, next_use_distance=9)[0])
        self.assertFalse(
            self.policy.handle_dag_event(
                dict(kind="consumer_upsert", workflow_id="workflow", epoch=0, seq=2,
                     logical_time=1000.0, consumer_id="consumer", session_id="new", step_id="s",
                     prefix_tokens=4, status="pending")
            )[0]
        )
        self.assertIs(self.policy.dag_workflows["workflow"].consumers["consumer"], saved)
        self.assertTrue(
            self.policy.handle_dag_event(
                dict(kind="branch_cancel", workflow_id="workflow", epoch=0, seq=3,
                     logical_time=1000.0, consumer_id="consumer", session_id="control")
            )[0]
        )
        self.assertFalse(self.policy.dag_hints_for_session("session"))
        self.assertTrue(self._consumer(seq=4, ttl_ms=1))
        self.policy.set_dag_logical_time(dict(logical_time=1002.0))
        self.assertFalse(self.policy.dag_hints_for_session("session"))
        self.assertTrue(self._consumer(seq=5))
        self.assertTrue(
            self.policy.handle_dag_event(
                dict(kind="workflow_reset", workflow_id="workflow", epoch=0, seq=6,
                     logical_time=1002.0, session_id="control")
            )[0]
        )
        self.assertEqual(self.policy.dag_workflows["workflow"].consumers, {})

    def test_stable_history_prepare_has_bounded_predictive_window(self):
        self.assertTrue(self._configure("eviction_prefetch", stable_history_prepare=True)[0])
        self.assertTrue(
            self._consumer(stable_history_prepare=True, expected_resume_at=1000.4)
        )
        self.policy.update(
            "session",
            LifecycleHint(
                state="DEPENDENCY_BLOCKED",
                expected_resume_at=1000.4,
                remaining_dependencies=1,
                confidence=1.0,
            ),
        )
        cost = dict(valid=True, predicted_restore_ms=400, predicted_recompute_ms=520)
        denied, fields = self.policy.admission(
            "session", 8, "host", True, cost, allow_stable_history=False
        )
        self.assertFalse(denied)
        self.assertEqual(fields["reason"], "dependencies_not_converged")
        allowed, fields = self.policy.admission(
            "session", 8, "host", True, cost, allow_stable_history=True
        )
        self.assertTrue(allowed)
        self.assertEqual(fields["reason"], "overlap_reduces_resume_wait")

        self.policy.update(
            "session",
            LifecycleHint(
                state="DEPENDENCY_BLOCKED",
                expected_resume_at=1100.0,
                remaining_dependencies=1,
                confidence=1.0,
            ),
        )
        denied, fields = self.policy.admission(
            "session", 8, "host", True, cost, allow_stable_history=True
        )
        self.assertFalse(denied)
        self.assertEqual(fields["reason"], "too_early_for_predictive_window")

        # A confirmed 8K stable prefix must not promote a completed request's
        # 12K history into predictive preparation.
        root = UnifiedTreeNode((self.ct,))
        node = self._node(root, 8)
        component = self._component([node], root)
        component._session_leaves["session"] = {node}
        cache = SimpleNamespace(
            agent_lifecycle=self.policy,
            enable_storage=True,
            cache_controller=object(),
            session_refs=SimpleNamespace(ensure_session_generation=lambda _sid: 1),
            components={self.ct: component},
            tree_core=component.tree_core,
            _component_available_size=lambda _ct: 0,
            _agent_cost_record=lambda _tokens: None,
            _all_reduce_storage_admission=lambda _vote: None,
        )
        cache._agent_match = lambda _sid: SimpleNamespace(
            device_indices=torch.empty(0), host_hit_length=8, best_match_node=node.id
        )
        cache._agent_dag_victim_plan = lambda needed, sid: (
            UnifiedRadixCache._agent_dag_victim_plan(cache, needed, sid)
        )
        req = SimpleNamespace(
            origin_input_ids=list(range(12)), extra_key="layout", cache_salt=None
        )
        self.assertFalse(
            UnifiedRadixCache._agent_prefetch_candidate(
                cache, "session", req, prefix_tokens=8
            )
        )
        self.assertEqual(len(self.policy.attempts["session"]["tokens"]), 8)

    def test_converged_16k_prefetch_keeps_v1_benefit_decision(self):
        self.assertTrue(self._configure("eviction_prefetch", stable_history_prepare=True)[0])
        self.assertTrue(self._consumer(prefix_tokens=16384, stable_history_prepare=True)[0])
        cost = dict(valid=True, predicted_restore_ms=627.755019813776,
                    predicted_recompute_ms=1326.355554163456)
        # The sealed trace has enough overlap, but it is outside EP's narrow
        # speculative window. Actual dependency convergence must retain V1.
        for restore_ms in (90.42651951313019, 113.42152208089828):
            self.policy.host_samples.clear()
            for _ in range(3):
                self.policy.observe("host", 16384, restore_ms)
            for state in ("READY_SOON", "TOOL_WAITING", "DEPENDENCY_BLOCKED"):
                with self.subTest(restore_ms=restore_ms, state=state):
                    self.policy.update("session", LifecycleHint(
                        state, 1000.3418154716492, remaining_dependencies=0,
                        confidence=0.9))
                    allowed, fields = self.policy.admission(
                        "session", 16384, "host", True, cost)
                    self.assertTrue(allowed)
                    self.assertEqual(fields["reason"], "overlap_reduces_resume_wait")
                    self.policy.update("session", LifecycleHint(
                        state, 1000.3418154716492, remaining_dependencies=1,
                        confidence=0.9))
                    allowed, fields = self.policy.admission(
                        "session", 16384, "host", True, cost,
                        allow_stable_history=True)
                    self.assertFalse(allowed)
                    self.assertEqual(fields["reason"], "too_early_for_predictive_window")

    def test_converged_prefetch_keeps_resource_and_state_rejections(self):
        self.assertTrue(self._configure("eviction_prefetch", stable_history_prepare=True)[0])
        for _ in range(3):
            self.policy.observe("host", 16384, 90.42651951313019)
        cases = (
            ("capacity", {"capacity": False}, "target_capacity"),
            ("copy", {"complete": False}, "incomplete_cold_copy"),
            ("budget", {"inflight": True}, "inflight_budget"),
            ("active", {"state": "ACTIVE"}, "state_not_prefetch_candidate"),
            ("completed", {"state": "COMPLETED"}, "state_not_prefetch_candidate"),
            ("confidence", {"confidence": 0.2}, "low_confidence"),
            ("samples", {"valid": False}, "missing_recompute_samples"),
            ("expired", {"deadline": 999.9}, "no_remaining_wait_benefit"),
            ("cost", {"deadline": 1000.03, "alternate": 50}, "no_remaining_wait_benefit"),
        )
        for name, options, reason in cases:
            with self.subTest(name=name):
                self.policy.attempts.clear()
                if options.get("inflight"):
                    self.policy.attempts["other"] = {"state": "REQUESTED"}
                self.policy.update("session", LifecycleHint(
                    options.get("state", "TOOL_WAITING"),
                    options.get("deadline", 1000.3418154716492),
                    remaining_dependencies=0, confidence=options.get("confidence", 0.9)))
                allowed, fields = self.policy.admission(
                    "session", 16384, "host", options.get("capacity", True),
                    dict(valid=options.get("valid", True), predicted_restore_ms=627.755,
                         predicted_recompute_ms=options.get("alternate", 1326.355)),
                    complete=options.get("complete", True))
                self.assertFalse(allowed)
                self.assertEqual(fields["reason"], reason)

    def test_dependency_ready_is_evaluated_by_existing_controller(self):
        self.assertTrue(self._configure("eviction_prefetch", stable_history_prepare=True)[0])
        self.assertTrue(self._consumer(
            prefix_tokens=16384, stable_history_prepare=True,
            required_dependencies=1, completed_dependencies=0)[0])
        for _ in range(3):
            self.policy.observe("host", 16384, 90.42651951313019)
        self.policy.update("session", LifecycleHint(
            "TOOL_WAITING", 1000.3418154716492,
            remaining_dependencies=0, confidence=0.9))
        decisions = []

        def prefetch(session_id, req, prefix_tokens):
            allowed, fields = self.policy.admission(
                session_id, prefix_tokens, "host", True,
                dict(valid=True, predicted_restore_ms=627.755,
                     predicted_recompute_ms=1326.355))
            decisions.append((prefix_tokens, allowed, fields["reason"]))
            return allowed

        controller = SimpleNamespace(tree_cache=SimpleNamespace(
            agent_lifecycle=self.policy,
            get_session_lifecycle_request=lambda _sid: object(),
            prefetch_session=prefetch))
        controller._drain_dag_prefetch_queue = lambda: (
            SessionController._drain_dag_prefetch_queue(controller))
        result = SessionController.handle_dag_event(controller, SimpleNamespace(
            dag_event=dict(kind="dependency_complete", workflow_id="workflow",
                           epoch=0, seq=2, logical_time=1000.0,
                           consumer_id="consumer", session_id="session")))
        self.assertTrue(result[0])
        self.assertEqual(self.policy.dag_workflows["workflow"].consumers["consumer"].status, "ready")
        self.assertTrue(result[4])
        self.assertEqual(decisions, [(16384, True, "overlap_reduces_resume_wait")])
        # A later management event must not prepare KV for an active request.
        self.policy.update("session", LifecycleHint("ACTIVE", confidence=1.0))
        result = SessionController.handle_dag_event(controller, SimpleNamespace(
            dag_event=dict(kind="tick", workflow_id="workflow", epoch=0, seq=3,
                           logical_time=1000.0, session_id="session")))
        self.assertFalse(result[4])
        self.assertEqual(decisions[-1], (16384, False, "missing_deadline"))

    def test_prefetch_candidate_status_and_expiry(self):
        self.assertTrue(self._configure("eviction_prefetch", stable_history_prepare=True)[0])
        self.assertTrue(self._consumer(prefix_tokens=16384, stable_history_prepare=True)[0])
        workflow = self.policy.dag_workflows["workflow"]
        hint = workflow.consumers["consumer"]
        for state in ("pending", "ready", "active", "cancelled", "completed"):
            with self.subTest(state=state):
                hint.status = state
                workflow.consumers["consumer"] = hint
                candidates = self.policy.dag_prefetch_candidates()
                self.assertEqual(len(candidates), int(state in {"pending", "ready"}))
        hint.status = "ready"
        hint.valid_until = 1000.0
        workflow.consumers["consumer"] = hint
        self.assertEqual(self.policy.dag_prefetch_candidates(), ())

    def test_ep_victim_trial_is_bounded_and_does_not_release_or_sacrifice_closer_use(self):
        self.assertTrue(self._configure("eviction_prefetch", stable_history_prepare=True)[0])
        self.assertTrue(self._consumer("future", "future-session", 1,
                                      next_use_distance=10, prefix_tokens=8,
                                      stable_history_prepare=True))
        self.assertTrue(self._consumer("closer", "victim-session", 2,
                                      next_use_distance=1, prefix_tokens=8))
        root = UnifiedTreeNode((self.ct,))
        victim = self._node(root, 8, slots=list(range(8)))
        component = self._component([victim], root)
        component._session_leaves["victim-session"] = {victim}
        cache = SimpleNamespace(
            agent_lifecycle=self.policy,
            components={self.ct: component},
            tree_core=component.tree_core,
            _component_available_size=lambda _ct: 0,
            _agent_cost_record=lambda _tokens: None,
        )
        plan = UnifiedRadixCache._agent_dag_victim_plan(cache, 8, "future-session")
        self.assertFalse(plan["allowed"])
        self.assertEqual(plan["reason"], "would_sacrifice_closer_consumer")
        self.assertIsNotNone(victim.component_data[self.ct].value)

    def test_ep_trial_counts_exposed_parent_without_real_free(self):
        self.assertTrue(self._configure("eviction_prefetch", stable_history_prepare=True)[0])
        self.assertTrue(self._consumer("future", "future-session", 1,
                                      next_use_distance=10, prefix_tokens=12,
                                      stable_history_prepare=True))
        root = UnifiedTreeNode((self.ct,))
        parent = self._node(root, 8, slots=list(range(8)))
        child = self._node(parent, 4, slots=list(range(8, 12)), key_start=8)
        component = self._component([child], root)
        cache = SimpleNamespace(
            agent_lifecycle=self.policy,
            components={self.ct: component},
            tree_core=component.tree_core,
            _component_available_size=lambda _ct: 0,
            _agent_cost_record=lambda _tokens: None,
        )

        plan = UnifiedRadixCache._agent_dag_victim_plan(cache, 12, "future-session")
        self.assertTrue(plan["allowed"])
        self.assertEqual(plan["trial_reclaimed_tokens"], 12)
        self.assertEqual(plan["victim_count"], 2)
        self.assertEqual({victim["tokens"] for victim in plan["victims"]}, {8, 4})
        # The trial must not release either page.
        self.assertIsNotNone(parent.component_data[self.ct].value)
        self.assertIsNotNone(child.component_data[self.ct].value)

        parent.component_data[self.ct].lock_ref = 1
        locked_plan = UnifiedRadixCache._agent_dag_victim_plan(
            cache, 12, "future-session"
        )
        self.assertFalse(locked_plan["allowed"])
        self.assertEqual(locked_plan["reason"], "insufficient_bounded_trial_capacity")
        self.assertEqual(locked_plan["trial_reclaimed_tokens"], 4)
        self.assertIsNotNone(parent.component_data[self.ct].value)
        self.assertIsNotNone(child.component_data[self.ct].value)

    def test_finite_budget_protects_only_a_bounded_token_share(self):
        self.assertTrue(self._configure("eviction")[0])
        self.assertTrue(self._consumer(next_use_distance=1))
        root = UnifiedTreeNode((self.ct,))
        near_nodes = [self._node(root, 1, key_start=i) for i in range(4)]
        ordinary = self._node(root, 1, key_start=4)
        ordinary.last_access_time = 999.0
        component = self._component(near_nodes + [ordinary], root)
        component._session_leaves["session"] = set(near_nodes)
        component._evict_device_start(5)
        self.assertEqual(len(component._dag_protected_nodes), 1)
        self.assertEqual(component._dag_protected_tokens_used, 1)
        self.assertEqual(len(component._dag_foreground_protected_nodes), 1)
        self.assertEqual(component._dag_foreground_tokens_used, 1)
        protected_near = next(iter(component._dag_protected_nodes))
        self.assertEqual(component._session_ref_eviction_strategy(protected_near)[0], 2)
        unprotected_near = next(n for n in near_nodes if n not in component._dag_protected_nodes)
        self.assertEqual(component._session_ref_eviction_strategy(unprotected_near)[0], 1)
        self.assertEqual(component._session_ref_eviction_strategy(ordinary)[0], 1)

        # Isomorphic TP trees can have different local node creation order.
        # The bounded protection budget must select the same logical prefix.
        other_root = UnifiedTreeNode((self.ct,))
        other_near = [self._node(other_root, 1, key_start=i) for i in reversed(range(4))]
        other_ordinary = self._node(other_root, 1, key_start=4)
        other_ordinary.last_access_time = 999.0
        other = self._component(other_near + [other_ordinary], other_root)
        other._session_leaves["session"] = set(other_near)
        other._evict_device_start(5)
        self.assertEqual(
            {component._dag_stable_key(node) for node in component._dag_protected_nodes},
            {other._dag_stable_key(node) for node in other._dag_protected_nodes},
        )

    def test_decode_role_and_snapshot_use_same_protocol_without_dag_policy(self):
        self.policy.role = "D"
        accepted, reason = self._configure("eviction")
        self.assertTrue(accepted)
        self.assertEqual(reason, "decode_role_dag_disabled")
        self.assertEqual(self.policy.dag_mode, "off")
        accepted, reason = self._consumer()
        self.assertTrue(accepted)
        self.assertEqual(reason, "decode_role_dag_disabled")
        self.assertEqual(self.policy.dag_workflows, {})

        root = UnifiedTreeNode((self.ct,))
        node = self._node(root, 10)
        cache = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(size_full=100),
            _component_available_size=lambda _ct: 40,
            full_evictable_size=lambda: 20,
            full_protected_size=lambda: 10,
            ongoing_load_back={1: SimpleNamespace(node_id=1)},
            tree_core=SimpleNamespace(node_by_id=lambda _node_id: node),
            ongoing_prefetch={},
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    anchor_entry=SimpleNamespace(
                        host_pool=SimpleNamespace(size=80, available_size=lambda: 30)
                    )
                )
            ),
            agent_lifecycle=self.policy,
        )
        snapshot = UnifiedRadixCache.dag_snapshot(cache)
        self.assertEqual(snapshot["gpu_total_tokens"], 100)
        self.assertEqual(snapshot["gpu_used_tokens"], 60)
        self.assertEqual(snapshot["gpu_free_tokens"], 40)
        self.assertEqual(snapshot["gpu_evictable_tokens"], 20)
        self.assertEqual(snapshot["gpu_locked_tokens"], 10)
        self.assertEqual(snapshot["gpu_reserved_tokens"], 10)
        self.assertEqual(snapshot["host_total_tokens"], 80)
        self.assertEqual(snapshot["host_used_tokens"], 50)
        self.assertEqual(snapshot["host_free_tokens"], 30)
        with patch("sglang.srt.mem_cache.agent_lifecycle.logger.info") as logged:
            self.policy.emit(None, "NATIVE_SNAPSHOT", **snapshot)
        emitted = json.loads(logged.call_args.args[1])
        self.assertEqual(emitted["role"], "D")
        self.assertEqual(emitted["mode"], "v1")
        self.assertEqual(emitted["gpu_used_tokens"], 60)


if __name__ == "__main__":
    unittest.main()
