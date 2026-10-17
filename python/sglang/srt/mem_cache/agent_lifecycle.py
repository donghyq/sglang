"""Bounded performance hints; KV identity and native transfer ownership stay unchanged."""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

logger = logging.getLogger(__name__)
WAIT_STATES = {"TOOL_WAITING", "DEPENDENCY_BLOCKED", "WAITING_INPUT"}
TERMINAL = {"CONSUMED", "CANCELLED", "TIMEOUT", "CAPACITY_REJECT", "EVICTED_BEFORE_USE", "STORAGE_MISS", "REJECTED"}
DAG_MODES = {"off", "eviction", "eviction_prefetch"}
DAG_EVENT_KINDS = {
    "configure", "consumer_upsert", "step_start", "step_complete",
    "dependency_complete", "branch_cancel", "workflow_finish", "workflow_reset",
    "tick", "snapshot",
}
DAG_CONSUMER_STATUSES = {"pending", "ready", "active", "cancelled", "completed"}
DAG_MAX_TTL_MS = 60_000
DAG_NEAR_USE_DISTANCE = 8
DAG_PROTECTED_TOKEN_FRACTION = 0.25
DAG_FOREGROUND_TOKEN_FRACTION = 0.25
DAG_EARLY_WINDOW_FACTOR = 0.25
DAG_MODE_CODES = {"off": 0, "eviction": 1, "eviction_prefetch": 2}


@dataclass(frozen=True)
class LifecycleHint:
    state: str
    expected_resume_at: float | None = None
    resume_lead_ms: float | None = None
    remaining_dependencies: int | None = None
    confidence: float = 0.0
    reason: str | None = None

    def validated(self, now: float):
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and in [0,1]")
        if self.remaining_dependencies is not None and self.remaining_dependencies < 0:
            raise ValueError("remaining_dependencies must be nonnegative")
        for value in (self.expected_resume_at, self.resume_lead_ms):
            if value is not None and not math.isfinite(value):
                raise ValueError("deadline hints must be finite")
        if self.resume_lead_ms is not None and self.resume_lead_ms < 0:
            raise ValueError("resume_lead_ms must be nonnegative")
        if self.expected_resume_at is None and self.resume_lead_ms is not None:
            return LifecycleHint(**dict(asdict(self), expected_resume_at=now + self.resume_lead_ms / 1000))
        return self


@dataclass
class DAGConsumerHint:
    workflow_id: str
    workflow_epoch: int
    consumer_id: str
    session_id: str
    step_id: str
    event_seq: int
    prefix_tokens: int
    next_use_distance: int | None
    status: str
    confidence: float
    expected_resume_at: float | None
    valid_until: float
    stable_history_prepare: bool = False
    allow_stable_history: bool = False
    required_dependencies: int | None = None
    completed_dependencies: int = 0
    dependency_mode: str = "ALL"


@dataclass
class DAGWorkflowState:
    epoch: int
    last_seq: int
    consumers: dict[str, DAGConsumerHint] = field(default_factory=dict)


class AgentLifecycle:
    def __init__(self, role: str, bytes_per_token: int):
        self.mode = os.getenv("SGLANG_AGENT_LIFECYCLE_MODE", "native")
        if self.mode not in {"native", "v0", "v1"}:
            raise ValueError("SGLANG_AGENT_LIFECYCLE_MODE must be native/v0/v1")
        self.role = role
        self.bytes_per_token = bytes_per_token
        self.hints = {}
        self.attempts = {}
        self.host_samples = {}
        self.restore_samples = {}
        self.pd_samples = {}
        self.query_started = {}
        self.loads = {}
        self.host_retention = {}
        self.max_inflight = 1
        self.completed_requests = {}
        self.dag_mode = "off"
        self.dag_stable_history_prepare = False
        self.dag_workflows: dict[str, DAGWorkflowState] = {}
        self.dag_max_workflows = 128
        self.dag_max_consumers = 1024
        self.dag_max_events = 10000
        self.dag_max_ttl_ms = DAG_MAX_TTL_MS
        self.dag_near_use_distance = DAG_NEAR_USE_DISTANCE
        self.dag_protected_token_fraction = DAG_PROTECTED_TOKEN_FRACTION
        self.dag_foreground_token_fraction = DAG_FOREGROUND_TOKEN_FRACTION
        self.dag_early_window_factor = DAG_EARLY_WINDOW_FACTOR
        # Common logical clock advanced only by broadcast management events.
        # Expiry is never decided from independently sampled local clocks.
        self.dag_logical_now = 0.0
        self.dag_event_count = 0
        self.native_consumption_log_budget = 4096
        self.native_consumption_events = 0
        self.native_eviction_log_budget = 4096
        self.native_eviction_events = 0
        self.native_victim_log_budget = 8192
        self.native_victim_events = 0
        self.dag_stats = {
            "events_accepted": 0,
            "events_rejected": 0,
            "eviction_candidate_checks": 0,
            "eviction_cpu_us": 0,
            "metadata_entries": 0,
        }
        self.trace_sessions = set(filter(None, os.getenv("SGLANG_AGENT_LIFECYCLE_TRACE_SESSIONS", "").split(",")))

    def remember_request(self, session_id, req, generation):
        if self.mode != "v1" or session_id is None or generation is None:
            return
        if req.session_generation != generation:
            return
        # Keep only logical identity, never request objects or physical KV handles.
        self.completed_requests[session_id] = (
            generation, req.origin_input_ids[:], req.output_ids[:],
            req.extra_key, req.cache_salt,
        )

    def completed_request(self, session_id, generation):
        saved = self.completed_requests.get(session_id)
        if self.mode != "v1" or saved is None or saved[0] != generation:
            return None
        return SimpleNamespace(origin_input_ids=saved[1][:], output_ids=saved[2][:],
                               extra_key=saved[3], cache_salt=saved[4])

    @staticmethod
    def bucket(tokens):
        return min((512, 8192, 16384), key=lambda n: abs(tokens - n))

    def emit(self, session_id, state, **fields):
        from sglang.srt.runtime_context import get_parallel
        row = dict(session_id=session_id, state=state, role=self.role,
                   rank=get_parallel().tp_rank, wall_time=time.time(), mode=self.mode, **fields)
        logger.info("AGENT_LIFECYCLE %s", json.dumps(row, sort_keys=True, default=str))

    def update(self, session_id, hint):
        hint = hint.validated(time.time())
        self.hints[session_id] = hint
        if hint.state == "COMPLETED":
            self.completed_requests.pop(session_id, None)
        self.emit(session_id, "HINT", hint=asdict(hint))
        return hint

    def dag_metadata_size(self):
        return len(self.dag_workflows) + sum(
            len(state.consumers) for state in self.dag_workflows.values()
        )

    def _dag_reject(self, event, reason):
        self.dag_stats["events_rejected"] += 1
        self.emit(event.get("session_id"), "DAG_EVENT_REJECTED", kind=event.get("kind"), reason=reason)
        return False, reason

    def set_dag_logical_time(self, event):
        value = event.get("logical_time")
        if value is None:
            return "missing_logical_time"
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return "invalid_logical_time"
        self.dag_logical_now = float(value)
        return None

    def consume_native_log_budget(self):
        if self.native_consumption_events >= self.native_consumption_log_budget:
            return False
        self.native_consumption_events += 1
        return True

    def consume_native_eviction_log_budget(self):
        if self.native_eviction_events >= self.native_eviction_log_budget:
            return False
        self.native_eviction_events += 1
        return True

    def consume_native_victim_log_budget(self):
        if self.native_victim_events >= self.native_victim_log_budget:
            return False
        self.native_victim_events += 1
        return True

    def configure_dag(self, event, idle=True):
        mode = event.get("mode")
        if mode not in DAG_MODES:
            return self._dag_reject(event, "invalid_mode")
        if self.role == "D":
            # Decode accepts the broadcast for protocol compatibility but keeps
            # its native path; P-side FULL KV is the only policy surface.
            self.emit(event.get("session_id"), "DAG_CONFIGURE_IGNORED", reason="decode_role", requested_mode=mode)
            return True, "decode_role_dag_disabled"
        if self.mode != "v1":
            return self._dag_reject(event, "lifecycle_mode_not_v1")
        if not idle:
            return self._dag_reject(event, "not_idle_window")
        if self.dag_event_count >= self.dag_max_events:
            return self._dag_reject(event, "event_budget")
        self.dag_mode = mode
        self.dag_stable_history_prepare = bool(event.get("stable_history_prepare", False))
        if mode == "off":
            self.dag_stable_history_prepare = False
        self.dag_workflows.clear()
        self.dag_event_count += 1
        self.native_consumption_events = 0
        self.native_eviction_events = 0
        self.native_victim_events = 0
        self.dag_stats["events_accepted"] += 1
        self.emit(event.get("session_id"), "DAG_CONFIGURED", mode=mode,
                  stable_history_prepare=self.dag_stable_history_prepare,
                  metadata_entries=self.dag_metadata_size())
        return True, None

    def dag_configure_is_valid(self, event):
        mode = event.get("mode")
        if mode not in DAG_MODES:
            return "invalid_mode"
        if self.role != "D" and self.mode != "v1":
            return "lifecycle_mode_not_v1"
        if self.role != "D" and self.dag_event_count >= self.dag_max_events:
            return "event_budget"
        stable = event.get("stable_history_prepare", False)
        if not isinstance(stable, bool):
            return "invalid_stable_history_prepare"
        return None

    def _dag_validate_event(self, event):
        workflow_id = event.get("workflow_id")
        if not isinstance(workflow_id, str) or not workflow_id:
            return "missing_workflow_id"
        epoch = event.get("epoch")
        seq = event.get("seq")
        if not isinstance(epoch, int) or epoch < 0:
            return "invalid_epoch"
        if not isinstance(seq, int) or seq < 0:
            return "invalid_seq"
        kind = event.get("kind")
        if kind == "consumer_upsert":
            required = ("consumer_id", "session_id", "step_id", "prefix_tokens", "status")
            if any(event.get(name) is None for name in required):
                return "missing_consumer_fields"
            consumer_id = event["consumer_id"]
            if not isinstance(consumer_id, str) or not consumer_id:
                return "invalid_consumer_id"
            prefix_tokens = event["prefix_tokens"]
            if not isinstance(prefix_tokens, int) or prefix_tokens <= 0:
                return "invalid_prefix_tokens"
            if event["status"] not in DAG_CONSUMER_STATUSES:
                return "invalid_consumer_status"
            confidence = event.get("confidence", 0.0)
            if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
                return "invalid_confidence"
            distance = event.get("next_use_distance")
            if distance is not None and (not isinstance(distance, int) or distance < 0):
                return "invalid_next_use_distance"
            expected = event.get("expected_resume_at")
            if expected is not None and not math.isfinite(float(expected)):
                return "invalid_expected_resume_at"
            ttl_ms = event.get("ttl_ms", 60_000)
            if not isinstance(ttl_ms, (int, float)) or not math.isfinite(float(ttl_ms)) or ttl_ms <= 0:
                return "invalid_ttl"
            required_dependencies = event.get("required_dependencies")
            completed = event.get("completed_dependencies", 0)
            remaining = event.get("remaining_dependencies")
            dependency_mode = event.get("dependency_mode", "ALL")
            if dependency_mode not in {"ALL"}:
                return "unsupported_dependency_mode"
            if required_dependencies is not None and (
                not isinstance(required_dependencies, int) or required_dependencies < 0
            ):
                return "invalid_required_dependencies"
            if not isinstance(completed, int) or completed < 0:
                return "invalid_completed_dependencies"
            if remaining is not None and (not isinstance(remaining, int) or remaining < 0):
                return "invalid_remaining_dependencies"
            if required_dependencies is None and remaining is not None:
                required_dependencies = completed + remaining
            if required_dependencies is not None and completed > required_dependencies:
                return "completed_exceeds_required"
            if (remaining is not None and required_dependencies is not None
                    and required_dependencies - completed != remaining):
                return "inconsistent_dependency_counts"
        elif kind in {"step_complete", "dependency_complete", "branch_cancel"}:
            consumer_id = event.get("consumer_id")
            if not isinstance(consumer_id, str) or not consumer_id:
                return "missing_consumer_id"
        return None

    def handle_dag_event(self, event, idle=True):
        if not isinstance(event, dict):
            return self._dag_reject({}, "invalid_event")
        kind = event.get("kind")
        if kind not in DAG_EVENT_KINDS:
            return self._dag_reject(event, "invalid_kind")
        reason = self.set_dag_logical_time(event)
        if reason is not None:
            return self._dag_reject(event, reason)
        if kind == "configure":
            return self.configure_dag(event, idle=idle)
        if kind in {"tick", "snapshot"}:
            # Read-only management events do not consume the state-event budget.
            return True, None
        if self.role == "D":
            # Validation still catches protocol drift, but no D-side policy state is kept.
            reason = self._dag_validate_event(event)
            if reason is not None:
                return self._dag_reject(event, reason)
            self.dag_stats["events_accepted"] += 1
            self.emit(event.get("session_id"), "DAG_EVENT_IGNORED", kind=kind, reason="decode_role")
            return True, "decode_role_dag_disabled"
        if self.dag_mode == "off":
            return self._dag_reject(event, "dag_mode_off")
        if self.dag_event_count >= self.dag_max_events:
            return self._dag_reject(event, "event_budget")
        reason = self._dag_validate_event(event)
        if reason is not None:
            return self._dag_reject(event, reason)

        workflow_id = event["workflow_id"]
        epoch = event["epoch"]
        seq = event["seq"]
        state = self.dag_workflows.get(workflow_id)
        created = False
        reset_epoch = False
        if state is None:
            if kind != "consumer_upsert":
                return self._dag_reject(event, "unknown_workflow")
            if len(self.dag_workflows) >= self.dag_max_workflows:
                return self._dag_reject(event, "workflow_budget")
            state = DAGWorkflowState(epoch=epoch, last_seq=-1)
            created = True
        else:
            if state.epoch > epoch:
                return self._dag_reject(event, "stale_epoch")
            if state.epoch < epoch:
                reset_epoch = True
            elif seq <= state.last_seq:
                return self._dag_reject(event, "stale_seq")

        # Validate consumer-specific state before mutating epoch or creating a
        # workflow, so a rejected event cannot pollute or consume a new seq.
        if reset_epoch and kind in {"step_complete", "dependency_complete"}:
            return self._dag_reject(event, "unknown_consumer")
        if kind in {"step_complete", "dependency_complete"}:
            hint = state.consumers.get(event["consumer_id"])
            if hint is None:
                return self._dag_reject(event, "unknown_consumer")
            if kind == "dependency_complete":
                if hint.required_dependencies is None:
                    return self._dag_reject(event, "dependency_count_not_declared")
                if hint.completed_dependencies >= hint.required_dependencies:
                    return self._dag_reject(event, "dependency_count_exceeded")
        if kind == "consumer_upsert":
            prospective_consumers = 0 if reset_epoch else len(state.consumers)
            if (not reset_epoch
                    and self.dag_metadata_size() >= self.dag_max_consumers
                    and event["consumer_id"] not in state.consumers):
                return self._dag_reject(event, "consumer_budget")
            if prospective_consumers >= self.dag_max_consumers:
                return self._dag_reject(event, "consumer_budget")

        if created:
            self.dag_workflows[workflow_id] = state
        if reset_epoch:
            state.epoch = epoch
            state.last_seq = -1
            state.consumers.clear()

        if kind == "consumer_upsert":
            consumer_id = event["consumer_id"]
            prefix_tokens = event["prefix_tokens"]
            distance = event.get("next_use_distance")
            expected = event.get("expected_resume_at")
            ttl_ms = event.get("ttl_ms", 60_000)
            required = event.get("required_dependencies")
            completed = event.get("completed_dependencies", 0)
            remaining = event.get("remaining_dependencies")
            dependency_mode = event.get("dependency_mode", "ALL")
            if required is None and remaining is not None:
                required = completed + remaining
            state.consumers[consumer_id] = DAGConsumerHint(
                workflow_id=event["workflow_id"], workflow_epoch=state.epoch,
                consumer_id=consumer_id, session_id=event["session_id"],
                step_id=event["step_id"], event_seq=event["seq"],
                prefix_tokens=prefix_tokens, next_use_distance=distance,
                status=event["status"], confidence=float(event.get("confidence", 0.0)),
                expected_resume_at=float(expected) if expected is not None else None,
                valid_until=self.dag_logical_now + min(float(ttl_ms), self.dag_max_ttl_ms) / 1000.0,
                stable_history_prepare=bool(event.get("stable_history_prepare", False)),
                allow_stable_history=self.dag_stable_history_prepare,
                required_dependencies=required,
                completed_dependencies=completed,
                dependency_mode=dependency_mode,
            )
        elif kind in {"step_complete", "dependency_complete"}:
            hint = state.consumers[event["consumer_id"]]
            if kind == "step_complete":
                hint.status = "completed"
            else:
                if hint.required_dependencies is None:
                    # One completed dependency is insufficient when the ALL
                    # frontier size was not declared.
                    return self._dag_reject(event, "dependency_count_not_declared")
                if hint.completed_dependencies >= hint.required_dependencies:
                    return self._dag_reject(event, "dependency_count_exceeded")
                hint.completed_dependencies += 1
                if hint.completed_dependencies >= hint.required_dependencies:
                    hint.status = "ready"
        elif kind == "branch_cancel":
            consumer_id = event.get("consumer_id")
            if consumer_id in state.consumers:
                state.consumers[consumer_id].status = "cancelled"
        elif kind == "workflow_finish":
            for hint in state.consumers.values():
                hint.status = "completed"
            state.consumers.clear()
        elif kind == "workflow_reset":
            state.consumers.clear()
        # step_start only advances the event sequence. Parallel frontiers are
        # represented by independent consumer records, never one global step.
        state.last_seq = event["seq"]
        self.dag_event_count += 1
        self.dag_stats["events_accepted"] += 1
        self.dag_stats["metadata_entries"] = self.dag_metadata_size()
        self.emit(event.get("session_id"), "DAG_EVENT", kind=kind,
                  workflow_id=event.get("workflow_id"), epoch=state.epoch, seq=event["seq"],
                  mode=self.dag_mode, metadata_entries=self.dag_metadata_size())
        return True, None

    def dag_cleanup_expired(self, now=None):
        now = self.dag_logical_now if now is None else now
        removed = 0
        for state in self.dag_workflows.values():
            expired = [cid for cid, h in state.consumers.items() if h.valid_until <= now or h.status in {"cancelled", "completed"}]
            for cid in expired:
                state.consumers.pop(cid, None)
                removed += 1
        return removed

    def dag_hints_for_session(self, session_id, now=None):
        if self.dag_mode == "off":
            return ()
        now = self.dag_logical_now if now is None else now
        self.dag_cleanup_expired(now)
        hints = []
        for state in self.dag_workflows.values():
            for hint in state.consumers.values():
                if (hint.session_id == session_id and hint.status in {"pending", "ready"}
                        and hint.valid_until > now and hint.confidence > 0):
                    hints.append(hint)
        return hints

    def dag_hints_by_session(self, now=None):
        if self.dag_mode == "off":
            return {}
        now = self.dag_logical_now if now is None else now
        self.dag_cleanup_expired(now)
        grouped = {}
        for state in self.dag_workflows.values():
            for hint in state.consumers.values():
                if (hint.status in {"pending", "ready"} and hint.valid_until > now
                        and hint.confidence > 0):
                    grouped.setdefault(hint.session_id, []).append(hint)
        return grouped

    def dag_prefetch_candidates(self):
        if self.dag_mode != "eviction_prefetch" or not self.dag_stable_history_prepare:
            return ()
        self.dag_cleanup_expired(self.dag_logical_now)
        candidates = [h for state in self.dag_workflows.values() for h in state.consumers.values()
                      if h.status == "pending" and h.stable_history_prepare and h.expected_resume_at is not None]
        candidates.sort(key=lambda h: (h.expected_resume_at, h.next_use_distance if h.next_use_distance is not None else 1 << 30,
                                      -h.prefix_tokens, h.workflow_id, h.consumer_id))
        return tuple(candidates[:32])

    def dag_prefix_consumed(self, session_id, matched_tokens):
        hints = self.dag_hints_for_session(session_id)
        if not hints:
            return
        self.emit(session_id, "DAG_PREFIX_CONSUMED", matched_tokens=matched_tokens,
                  consumer_ids=[h.consumer_id for h in hints],
                  workflow_ids=sorted({h.workflow_id for h in hints}))

    def diagnose(self, session_id, stage, **fields):
        if session_id in self.trace_sessions:
            self.emit(session_id, "POLICY_CHECK", stage=stage, **fields)

    def record(self, session_id, state, **fields):
        attempt = self.attempts.get(session_id)
        if attempt is None:
            return
        attempt.update(state=state, **fields)
        if state == "ADMITTED":
            attempt["ever_admitted"] = True
        if state == "GPU_READY":
            attempt["gpu_ready_at"] = time.time()
        self.emit(session_id, state, handle=attempt["handle"], attempt_id=attempt["attempt_id"],
                  session_generation=attempt["generation"], **fields)

    def request(self, session_id, tokens, source, generation=0):
        previous = self.attempts.get(session_id)
        if previous is not None and previous["state"] not in TERMINAL:
            return False
        self.attempts[session_id] = dict(handle="_agent_prefetch_" + session_id,
                                      requested_tokens=tokens, source_tier=source, generation=generation,
                                      attempt_id=(generation<<20)+(previous.get("sequence",0)+1 if previous else 1),
                                      sequence=previous.get("sequence",0)+1 if previous else 1,
                                      state="REQUESTED", useful_bytes=0, transferred_bytes=0,
                                      requested_at=time.time())
        self.record(session_id, "REQUESTED", tokens=tokens, source_tier=source)
        return True

    def session_for_handle(self, handle):
        rid = getattr(handle, "rid", "")
        if not rid.startswith("_agent_prefetch_"):
            return None
        sid = rid[len("_agent_prefetch_"):]
        attempt = self.attempts.get(sid)
        if attempt is not None and getattr(handle, "attempt_id", 0) != attempt.get("attempt_id", 0):
            return None
        return sid

    def observe(self, tier, tokens, elapsed_ms, load_bucket="idle"):
        if tokens <= 0 or elapsed_ms <= 0:
            return
        samples = self.pd_samples if tier == "pd_history" else (self.host_samples if tier == "host" else self.restore_samples)
        samples.setdefault((self.bucket(tokens), load_bucket), deque(maxlen=32)).append(elapsed_ms)
        self.emit(None, "COST_OBSERVATION", tokens=tokens, source_tier=tier, elapsed_ms=elapsed_ms,
                  load_bucket=load_bucket, cost_scope="PD_history_ready_path_wall" if tier == "pd_history" else "restore_critical_path_wall")

    def estimate(self, tokens, tier, gate_record=None):
        load_bucket = (gate_record or {}).get("load_bucket", "idle")
        samples = (self.host_samples if tier == "host" else self.restore_samples).get((self.bucket(tokens), load_bucket), ())
        if len(samples) >= 3:
            return max(samples), "observed_restore_critical_path"
        # P request TTFT bounds its existing restore path conservatively. Never
        # substitute P recompute costs for D incoming-KV costs.
        if self.role == "P" and gate_record and gate_record["valid"]:
            return gate_record["predicted_restore_ms"], "calibrated_P_restore_TTFT_upper_bound"
        return None, "missing_restore_samples"

    def admission(self, session_id, tokens, tier, capacity, gate_record=None, complete=True,
                  allow_stable_history=False):
        hint = self.hints.get(session_id)
        if hint is None or hint.expected_resume_at is None:
            return False, {"reason": "missing_deadline"}
        if hint.confidence < 0.8:
            return False, {"reason": "low_confidence"}
        if hint.state in {"ACTIVE", "WAITING_INPUT", "COMPLETED"}:
            return False, {"reason": "state_not_prefetch_candidate"}
        if hint.remaining_dependencies != 0 and not (allow_stable_history and self.dag_stable_history_prepare):
            return False, {"reason": "dependencies_not_converged"}
        if hint.remaining_dependencies != 0 and allow_stable_history:
            stable = next((h for h in self.dag_hints_for_session(session_id)
                           if h.stable_history_prepare and h.prefix_tokens >= tokens), None)
            if stable is None:
                return False, {"reason": "stable_history_not_confirmed"}
        if not complete:
            return False, {"reason": "incomplete_cold_copy"}
        if not capacity:
            return False, {"reason": "target_capacity"}
        inflight = sum(a["state"] not in TERMINAL for sid, a in self.attempts.items() if sid != session_id)
        if inflight >= self.max_inflight:
            return False, {"reason": "inflight_budget"}
        restore, basis = self.estimate(tokens, tier, gate_record)
        if restore is None:
            return False, {"reason": basis}
        alternate = None
        if self.role == "P" and gate_record and gate_record["valid"]:
            alternate = gate_record["predicted_recompute_ms"]
        # Without a measured D retransmit alternative, preserve native decode.
        if self.role == "D":
            samples = self.pd_samples.get((self.bucket(tokens), (gate_record or {}).get("load_bucket", "idle")), ())
            if len(samples) < 3:
                return False, {"reason": "missing_D_history_retransmit_samples", "restore_estimate_ms": restore}
            alternate = min(samples)
        if alternate is None:
            return False, {"reason": "missing_recompute_samples"}
        # EP decisions use the broadcast logical clock. Ordinary V1 keeps its
        # original wall-clock behavior when DAG mode is off.
        now = (
            self.dag_logical_now
            if self.dag_mode == "eviction_prefetch"
            else time.time()
        )
        overlap = (hint.expected_resume_at - now) * 1000
        remaining = max(0.0, restore - max(0.0, overlap))
        fields = dict(restore_estimate_ms=restore, alternate_estimate_ms=alternate,
                      available_overlap_ms=overlap, remaining_restore_wait_ms=remaining,
                      estimate_basis=basis)
        if (self.dag_mode == "eviction_prefetch"
                and overlap > restore * (1.0 + self.dag_early_window_factor)):
            return False, dict(fields, reason="too_early_for_predictive_window")
        if overlap <= 0 or remaining >= min(restore, alternate):
            return False, dict(fields, reason="no_remaining_wait_benefit")
        return True, dict(fields, reason="overlap_reduces_resume_wait")

    def can_retain_host(self, session_ids, node, component_type, pending_writes,
                        recovery_ms=None, details=None):
        def reject(reason, **fields):
            if details is not None:
                details.update(reason=reason, **fields)
            return False

        hints = [self.hints.get(sid) for sid in session_ids]
        if not hints or any(h is None or h.state not in {"TOOL_WAITING", "DEPENDENCY_BLOCKED"} for h in hints):
            return reject("not_waiting_for_dependencies")
        if recovery_ms is None or any(h.confidence < 0.8 or h.expected_resume_at is None or
                                      (h.expected_resume_at-time.time())*1000 <= recovery_ms for h in hints):
            return reject("missing_or_unfavorable_recovery_estimate")
        if node.parent is None:
            return reject("no_cached_prefix")
        pending_nodes = set(pending_writes)
        for entry in pending_writes.values():
            pending_nodes.update(getattr(entry, "node_ids", ()))
            pending_nodes.update(getattr(entry, "publish_node_ids", ()))
            pending_nodes.add(getattr(entry, "node_id", None))
        current = node
        while current.parent is not None:
            cd = current.component_data[component_type]
            if current.id in pending_nodes:
                return reject("write_still_pending", node_id=current.id)
            if cd.host_value is None or len(cd.host_value) != len(current.key):
                return reject("incomplete_host_copy", node_id=current.id)
            current = current.parent
        return True

    def demotable_chain(self, node, component_type):
        # A shared ancestor does not prevent releasing an exclusive suffix.
        # Stop before shared/active/locked KV; native leaf checks still apply.
        chain = []
        while node.parent is not None:
            cd = node.component_data[component_type]
            owners = getattr(cd, "session_ids", None) or ()
            if (getattr(cd, "lock_ref", 0) > 0 or
                getattr(cd, "session_ref", len(owners)) > 1 or
                any(self.hints.get(sid) is None or self.hints[sid].state not in
                    {"TOOL_WAITING", "DEPENDENCY_BLOCKED"} for sid in owners)):
                break
            if getattr(cd, "value", None) is not None:
                chain.append(node.id)
            node = node.parent
        return chain

    def can_demote(self, session_ids, node, component_type, pending_writes, recovery_ms=None):
        if not self.can_retain_host(session_ids, node, component_type, pending_writes, recovery_ms):
            return False
        cd = node.component_data[component_type]
        owners = getattr(cd, "session_ids", None) or ()
        return (getattr(cd, "lock_ref", 0) == 0 and
                getattr(cd, "session_ref", len(owners)) <= 1 and
                all(self.hints.get(sid) is not None and self.hints[sid].state in
                    {"TOOL_WAITING", "DEPENDENCY_BLOCKED"} for sid in owners))

    def consume(self, session_id, device_indices):
        attempt = self.attempts.get(session_id)
        if not attempt or attempt["state"] != "GPU_READY":
            return
        used = attempt.get("gpu_indices", set()) & set(device_indices.cpu().tolist())
        if not used:
            return
        attempt["gpu_indices"] -= used
        useful = attempt["useful_bytes"] + len(used) * self.bytes_per_token
        self.record(session_id, "CONSUMED", useful_bytes=useful,
                    ready_before_resume=attempt.get("gpu_ready_at", float("inf")) <= attempt.get("resume_at", time.time()))

    def evicted(self, device_indices):
        freed = set(device_indices.cpu().tolist())
        for sid, attempt in self.attempts.items():
            unused = attempt.get("gpu_indices", set()) & freed
            if unused:
                attempt["gpu_indices"] -= unused
                lost = attempt.get("evicted_bytes", 0) + len(unused)*self.bytes_per_token
                state = ("CONSUMED" if attempt["useful_bytes"] else "EVICTED_BEFORE_USE") if not attempt["gpu_indices"] else attempt["state"]
                self.record(sid, state, evicted_bytes=lost)
