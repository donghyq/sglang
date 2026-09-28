import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.io_struct import UpdateSessionLifecycleReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.session_ref_tracker import (
    SessionLifecycleState,
    UnifiedSessionRefTracker,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.session.session_controller import Session, SessionController
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSessionLifecycle(CustomTestCase):
    def _component(self, tracker: UnifiedSessionRefTracker):
        component = FullComponent.__new__(FullComponent)
        component.cache = SimpleNamespace(session_refs=tracker)
        component.tree_core = SimpleNamespace(
            eviction_strategy=SimpleNamespace(get_priority=lambda _node: 0)
        )
        component.component_type = ComponentType.FULL
        return component

    def _node(self, session_id: str):
        return SimpleNamespace(
            component_data={
                ComponentType.FULL: SimpleNamespace(
                    session_ref=1, session_ids={session_id}
                )
            }
        )

    def test_tool_waiting_session_is_evicted_first(self):
        tracker = UnifiedSessionRefTracker(
            components=(),
            tree_core=SimpleNamespace(),
            enable_session_radix_cache=True,
        )
        tracker.set_session_lifecycle("waiting", SessionLifecycleState.TOOL_WAITING)
        tracker.set_session_lifecycle("active", SessionLifecycleState.ACTIVE)
        component = self._component(tracker)

        waiting_priority = component._session_ref_eviction_strategy(
            self._node("waiting")
        )
        active_priority = component._session_ref_eviction_strategy(self._node("active"))

        self.assertLess(waiting_priority, active_priority)

    def test_ready_soon_keeps_normal_priority(self):
        tracker = UnifiedSessionRefTracker(
            components=(),
            tree_core=SimpleNamespace(),
            enable_session_radix_cache=True,
        )
        tracker.set_session_lifecycle("ready", SessionLifecycleState.READY_SOON)
        tracker.set_session_lifecycle("active", SessionLifecycleState.ACTIVE)
        component = self._component(tracker)

        ready_priority = component._session_ref_eviction_strategy(self._node("ready"))
        active_priority = component._session_ref_eviction_strategy(self._node("active"))

        self.assertEqual(ready_priority, active_priority)

    def test_session_controller_tool_waiting_updates_state(self):
        states = {}

        class FakeTreeCache:
            def set_session_lifecycle(self, session_id, state):
                states[session_id] = state

            def prefetch_session(self, session_id, req):
                return None

        controller = SessionController(FakeTreeCache())
        controller.sessions["session"] = Session(capacity_of_str_len=100)
        controller.mark_tool_waiting("session")

        self.assertEqual(states["session"], SessionLifecycleState.TOOL_WAITING)

    def test_session_controller_ready_soon_prefetches_latest_request(self):
        states = {}
        prefetch_calls = []

        class FakeTreeCache:
            def set_session_lifecycle(self, session_id, state):
                states[session_id] = state

            def prefetch_session(self, session_id, req):
                prefetch_calls.append((session_id, req))
                return True

        controller = SessionController(FakeTreeCache())
        session = Session(capacity_of_str_len=100)
        req = SimpleNamespace(
            last_node=123, origin_input_ids=[1, 2, 3], extra_key=None, cache_salt=None
        )
        session.req_nodes["rid"] = SimpleNamespace(req=req)
        controller.sessions["session"] = session

        result = controller.mark_ready_soon("session")

        self.assertTrue(result)
        self.assertEqual(states["session"], SessionLifecycleState.READY_SOON)
        self.assertEqual(prefetch_calls, [("session", req)])

    def test_ready_soon_reanchors_after_session_leaf_is_evicted(self):
        calls = []
        cache = SimpleNamespace(
            enable_storage=True,
            cache_controller=object(),
            match_prefix=lambda _params: SimpleNamespace(
                host_hit_length=0, device_indices=[]
            ),
            root_node_handle=lambda _extra_key: 0,
            prefetch_from_storage=lambda *args, **kwargs: (
                calls.append((args, kwargs)) or True
            ),
        )
        req = SimpleNamespace(
            last_node=14,
            origin_input_ids=array("q", [1, 2, 3]),
            extra_key=None,
            cache_salt=None,
        )

        self.assertTrue(UnifiedRadixCache.prefetch_session(cache, "session", req))
        self.assertEqual(calls[0][0][1:], (0, array("q", [1, 2, 3])))
        self.assertEqual(calls[0][1]["matched_prefix_tokens"], [])

    def test_ready_soon_reports_async_prefetch_submission(self):
        cache = SimpleNamespace(
            enable_storage=True,
            cache_controller=object(),
            match_prefix=lambda _params: SimpleNamespace(
                host_hit_length=0, device_indices=[]
            ),
            root_node_handle=lambda _extra_key: 0,
            prefetch_from_storage=lambda *_args, **_kwargs: None,
            has_ongoing_prefetch=lambda _handle: True,
        )
        req = SimpleNamespace(
            last_node=14, origin_input_ids=[1, 2, 3], extra_key=None, cache_salt=None
        )

        self.assertTrue(UnifiedRadixCache.prefetch_session(cache, "session", req))

    def test_ready_soon_loads_host_copy_before_storage_query(self):
        loaded_nodes = []
        cache = SimpleNamespace(
            enable_storage=True,
            cache_controller=object(),
            match_prefix=lambda _params: SimpleNamespace(
                host_hit_length=3, best_match_node=15, device_indices=[]
            ),
            load_back=lambda node_id: loaded_nodes.append(node_id) or True,
            prefetch_from_storage=lambda *_args, **_kwargs: self.fail(
                "host copy should be loaded before querying storage"
            ),
        )
        req = SimpleNamespace(
            last_node=14,
            origin_input_ids=array("q", [1, 2, 3]),
            extra_key=None,
            cache_salt=None,
        )

        self.assertTrue(UnifiedRadixCache.prefetch_session(cache, "session", req))
        self.assertEqual(loaded_nodes, [15])

    def test_ready_soon_skips_fully_resident_prefix(self):
        cache = SimpleNamespace(
            enable_storage=True,
            cache_controller=object(),
            match_prefix=lambda _params: SimpleNamespace(
                host_hit_length=0, device_indices=[1, 2, 3]
            ),
            prefetch_from_storage=lambda *_args, **_kwargs: self.fail(
                "fully resident prefix should not query storage"
            ),
        )
        req = SimpleNamespace(
            last_node=14,
            origin_input_ids=array("q", [1, 2, 3]),
            extra_key=None,
            cache_salt=None,
        )

        self.assertFalse(UnifiedRadixCache.prefetch_session(cache, "session", req))

    def test_scheduler_applies_lifecycle_hint_and_acknowledges(self):
        class FakeController(SimpleNamespace):
            def __contains__(self, session_id):
                return session_id == "session"

        states = []
        controller = FakeController(
            tree_cache=SimpleNamespace(set_session_lifecycle=lambda *_: None),
            mark_tool_waiting=lambda sid: states.append((sid, "TOOL_WAITING")),
            mark_ready_soon=lambda sid: states.append((sid, "READY_SOON")) or True,
        )
        scheduler = SimpleNamespace(session_controller=controller)
        parallel = SimpleNamespace(pp_rank=0, tp_rank=0, attn_cp_rank=0)
        with patch("sglang.srt.managers.scheduler.get_parallel", return_value=parallel):
            waiting = Scheduler.update_session_lifecycle(
                scheduler,
                UpdateSessionLifecycleReqInput(
                    rid="waiting", session_id="session", state="TOOL_WAITING"
                ),
            )
            ready = Scheduler.update_session_lifecycle(
                scheduler,
                UpdateSessionLifecycleReqInput(
                    rid="ready", session_id="session", state="READY_SOON"
                ),
            )

        self.assertTrue(waiting.success)
        self.assertIsNone(waiting.prefetch_started)
        self.assertEqual(waiting.rid, "waiting")
        self.assertTrue(ready.success)
        self.assertTrue(ready.prefetch_started)
        self.assertEqual(
            states, [("session", "TOOL_WAITING"), ("session", "READY_SOON")]
        )

    def test_scheduler_rejects_unknown_session_or_state(self):
        class FakeController(SimpleNamespace):
            def __contains__(self, session_id):
                return session_id == "session"

        controller = FakeController(
            tree_cache=SimpleNamespace(set_session_lifecycle=lambda *_: None)
        )
        scheduler = SimpleNamespace(session_controller=controller)
        parallel = SimpleNamespace(pp_rank=0, tp_rank=0, attn_cp_rank=0)
        with patch("sglang.srt.managers.scheduler.get_parallel", return_value=parallel):
            missing = Scheduler.update_session_lifecycle(
                scheduler,
                UpdateSessionLifecycleReqInput(
                    rid="missing", session_id="other", state="TOOL_WAITING"
                ),
            )
            invalid = Scheduler.update_session_lifecycle(
                scheduler,
                UpdateSessionLifecycleReqInput(
                    rid="invalid", session_id="session", state="UNKNOWN"
                ),
            )

        self.assertFalse(missing.success)
        self.assertFalse(invalid.success)


if __name__ == "__main__":
    unittest.main()
