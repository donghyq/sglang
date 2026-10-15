"""Bounded performance hints; KV identity and native transfer ownership stay unchanged."""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from types import SimpleNamespace

logger = logging.getLogger(__name__)
WAIT_STATES = {"TOOL_WAITING", "DEPENDENCY_BLOCKED", "WAITING_INPUT"}
TERMINAL = {"CONSUMED", "CANCELLED", "TIMEOUT", "CAPACITY_REJECT", "EVICTED_BEFORE_USE", "STORAGE_MISS", "REJECTED"}


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

    def admission(self, session_id, tokens, tier, capacity, gate_record=None, complete=True):
        hint = self.hints.get(session_id)
        if hint is None or hint.expected_resume_at is None:
            return False, {"reason": "missing_deadline"}
        if hint.confidence < 0.8:
            return False, {"reason": "low_confidence"}
        if hint.state in {"ACTIVE", "WAITING_INPUT", "COMPLETED"}:
            return False, {"reason": "state_not_prefetch_candidate"}
        if hint.remaining_dependencies != 0:
            return False, {"reason": "dependencies_not_converged"}
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
        overlap = (hint.expected_resume_at - time.time()) * 1000
        remaining = max(0.0, restore - max(0.0, overlap))
        fields = dict(restore_estimate_ms=restore, alternate_estimate_ms=alternate,
                      available_overlap_ms=overlap, remaining_restore_wait_ms=remaining,
                      estimate_basis=basis)
        if overlap <= 0 or remaining >= min(restore, alternate):
            return False, dict(fields, reason="no_remaining_wait_benefit")
        return True, dict(fields, reason="overlap_reduces_resume_wait")

    def can_demote(self, session_ids, node, component_type, pending_writes, recovery_ms=None):
        hints = [self.hints.get(sid) for sid in session_ids]
        if not hints or any(h is None or h.state not in {"TOOL_WAITING", "DEPENDENCY_BLOCKED"} for h in hints):
            return False
        if recovery_ms is None or any(h.confidence < 0.8 or h.expected_resume_at is None or
                                      (h.expected_resume_at-time.time())*1000 <= recovery_ms for h in hints):
            return False
        pending_nodes = set(pending_writes)
        for entry in pending_writes.values():
            pending_nodes.update(getattr(entry, "node_ids", ()))
            pending_nodes.update(getattr(entry, "publish_node_ids", ()))
            pending_nodes.add(getattr(entry, "node_id", None))
        current = node
        while current.parent is not None:
            cd = current.component_data[component_type]
            owners = getattr(cd, "session_ids", None) or ()
            if getattr(cd, "session_ref", len(owners)) > 1:
                return False
            if any(self.hints.get(sid) is None or self.hints[sid].state not in {"TOOL_WAITING", "DEPENDENCY_BLOCKED"} for sid in owners):
                return False
            if current.id in pending_nodes or cd.host_value is None or len(cd.host_value) != len(current.key):
                return False
            current = current.parent
        return True

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
