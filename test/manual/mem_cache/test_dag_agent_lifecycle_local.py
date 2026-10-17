"""Local, dependency-free DAG lifecycle checks.

This file intentionally loads ``agent_lifecycle.py`` directly instead of
importing the ``sglang`` package. It is for machines without Triton/GPU
dependencies; the registered real-node regression remains
``test/registered/unit/mem_cache/test_dag_kv_policy.py``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest
from unittest.mock import patch


def _load_agent_lifecycle():
    path = pathlib.Path(__file__).parents[3] / (
        "python/sglang/srt/mem_cache/agent_lifecycle.py"
    )
    spec = importlib.util.spec_from_file_location("agent_lifecycle_local", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_agent_lifecycle()


class PureAgentLifecycleTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(mod.os.environ, {"SGLANG_AGENT_LIFECYCLE_MODE": "v1"}):
            self.policy = mod.AgentLifecycle("P", 49152)
        self.policy.mode = "v1"
        self.policy.emit = lambda *args, **kwargs: None

    def event(self, kind, seq, **fields):
        payload = dict(
            kind=kind,
            workflow_id="wf",
            epoch=0,
            seq=seq,
            session_id="control",
            logical_time=1000.0,
        )
        payload.update(fields)
        with patch.object(mod.time, "time", return_value=1000.0):
            return self.policy.handle_dag_event(payload, idle=True)

    def admission(self, *args, **kwargs):
        with patch.object(mod.time, "time", return_value=1000.0):
            return self.policy.admission(*args, **kwargs)

    def test_configure_and_stable_flag(self):
        self.assertTrue(self.event("configure", 0, mode="eviction")[0])
        self.assertEqual(self.policy.dag_mode, "eviction")
        self.assertFalse(self.policy.dag_stable_history_prepare)
        self.assertTrue(
            self.event(
                "configure",
                1,
                mode="eviction_prefetch",
                stable_history_prepare=True,
            )[0]
        )
        self.assertTrue(self.policy.dag_stable_history_prepare)
        self.assertTrue(self.event("configure", 2, mode="off")[0])
        self.assertEqual(self.policy.dag_mode, "off")

    def test_dependency_counts(self):
        self.event("configure", 3, mode="eviction")
        self.assertTrue(
            self.event(
                "consumer_upsert",
                4,
                consumer_id="c",
                session_id="s",
                step_id="step",
                prefix_tokens=8,
                next_use_distance=2,
                status="pending",
                confidence=1,
                required_dependencies=2,
                completed_dependencies=0,
                dependency_mode="ALL",
            )[0]
        )
        self.assertTrue(self.event("dependency_complete", 5, consumer_id="c")[0])
        hint = self.policy.dag_workflows["wf"].consumers["c"]
        self.assertEqual((hint.status, hint.completed_dependencies), ("pending", 1))
        self.assertTrue(self.event("dependency_complete", 6, consumer_id="c")[0])
        self.assertEqual((hint.status, hint.completed_dependencies), ("ready", 2))

    def test_ttl_is_capped(self):
        self.event("configure", 7, mode="eviction")
        self.assertTrue(
            self.event(
                "consumer_upsert",
                8,
                consumer_id="c",
                session_id="s",
                step_id="step",
                prefix_tokens=8,
                next_use_distance=1,
                status="pending",
                confidence=1,
                ttl_ms=100000000,
            )[0]
        )
        self.assertEqual(
            self.policy.dag_workflows["wf"].consumers["c"].valid_until, 1060.0
        )

    def test_stale_seq_does_not_replace(self):
        self.event("configure", 9, mode="eviction")
        self.assertTrue(
            self.event(
                "consumer_upsert",
                10,
                consumer_id="c",
                session_id="s",
                step_id="step",
                prefix_tokens=8,
                next_use_distance=2,
                status="pending",
                confidence=1,
            )[0]
        )
        saved = self.policy.dag_workflows["wf"].consumers["c"]
        self.assertFalse(
            self.event(
                "consumer_upsert",
                9,
                consumer_id="c",
                session_id="other",
                step_id="step",
                prefix_tokens=4,
                status="pending",
                confidence=1,
            )[0]
        )
        self.assertIs(self.policy.dag_workflows["wf"].consumers["c"], saved)

    def test_readonly_tick_and_snapshot(self):
        self.event(
            "configure",
            11,
            mode="eviction_prefetch",
            stable_history_prepare=True,
        )
        self.assertTrue(
            self.event(
                "consumer_upsert",
                12,
                consumer_id="c",
                session_id="s",
                step_id="step",
                prefix_tokens=8,
                next_use_distance=2,
                status="pending",
                confidence=1,
                stable_history_prepare=True,
                expected_resume_at=1000.4,
                required_dependencies=1,
                completed_dependencies=0,
                dependency_mode="ALL",
            )[0]
        )
        self.assertTrue(self.event("tick", 12)[0])
        self.assertTrue(self.event("snapshot", 12)[0])
        self.assertTrue(self.event("dependency_complete", 13, consumer_id="c")[0])

    def test_stable_history_window(self):
        self.event(
            "configure",
            14,
            mode="eviction_prefetch",
            stable_history_prepare=True,
        )
        self.assertTrue(
            self.event(
                "consumer_upsert",
                15,
                consumer_id="c",
                session_id="s",
                step_id="step",
                prefix_tokens=8,
                next_use_distance=2,
                status="pending",
                confidence=1,
                stable_history_prepare=True,
                expected_resume_at=1000.4,
            )[0]
        )
        self.policy.update(
            "s",
            mod.LifecycleHint(
                state="DEPENDENCY_BLOCKED",
                expected_resume_at=1000.4,
                remaining_dependencies=1,
                confidence=1,
            ),
        )
        cost = dict(
            valid=True, predicted_restore_ms=400, predicted_recompute_ms=520
        )
        allowed, _fields = self.admission(
            "s", 8, "host", True, cost, allow_stable_history=True
        )
        self.assertTrue(allowed)
        self.policy.update(
            "s",
            mod.LifecycleHint(
                state="DEPENDENCY_BLOCKED",
                expected_resume_at=1100.0,
                remaining_dependencies=1,
                confidence=1,
            ),
        )
        denied, fields = self.admission(
            "s", 8, "host", True, cost, allow_stable_history=True
        )
        self.assertFalse(denied)
        self.assertEqual(fields["reason"], "too_early_for_predictive_window")

    def test_decode_role_accepts_but_stays_native(self):
        self.policy.role = "D"
        accepted, reason = self.event("configure", 16, mode="eviction")
        self.assertTrue(accepted)
        self.assertEqual(reason, "decode_role_dag_disabled")
        self.assertEqual(self.policy.dag_mode, "off")


if __name__ == "__main__":
    unittest.main(verbosity=2)
