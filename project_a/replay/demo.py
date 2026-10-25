"""Run the extracted control queue with a synthetic dependency graph.

This example uses no model, GPU, service or network. The delays represent local
control acknowledgements only and must not be reported as benchmark results.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from types import SimpleNamespace

from .control_dispatch import workflow_control_queue


class SyntheticRunner:
    def __init__(self, entry):
        self.started = time.monotonic()
        self.spec = {"entries": [entry]}
        self.events = []
        self.decisions = []
        self.decision_writer = SimpleNamespace(write=self.decisions.append)
        self.dependency_locks = {}
        self.completed_dependencies = {}
        self.convergence_events = {entry["workflow_id"]: asyncio.Event()}
        self.expected_resume_by_workflow = {entry["workflow_id"]: time.time() + .05}

    def elapsed(self):
        return time.monotonic() - self.started

    async def tool_delay(self, seconds):
        await asyncio.sleep(seconds)

    def workflow_event(self, event, **fields):
        self.events.append({"event": event, "elapsed_s": self.elapsed(), **fields})


async def demonstrate(control_delay_ms=20.0, native=False):
    entry = {
        "workflow_id": "example-workflow", "session_id": "example-session",
        "prefix_tokens": 8192, "required_dependencies": 2,
        "actual_tool_seconds": {"availability-a": .01, "availability-b": .02},
    }
    runner = SyntheticRunner(entry)
    acknowledgements = []

    async def lifecycle(session_id, state, **fields):
        if native:
            return [{"sent": False, "suppressed": True}]
        for role in ("P", "D"):
            await asyncio.sleep(control_delay_ms / 1000)
            acknowledgements.append({"role": role, "kind": "lifecycle", "state": state})
        return [{"success": True}]

    async def dag(session_id, state, event, *, workflow_id=None):
        if native:
            return [{"sent": False, "suppressed": True}]
        for role in ("P", "D"):
            await asyncio.sleep(control_delay_ms / 1000)
            acknowledgements.append({"role": role, "kind": event["kind"], "state": state})
        return [{"success": True}]

    run_tool, hint, dag_event, metadata, drain = workflow_control_queue(runner, lifecycle, dag)
    await asyncio.gather(*(run_tool(entry, name) for name in entry["actual_tool_seconds"]))
    await runner.convergence_events[entry["workflow_id"]].wait()
    ready = runner.elapsed()
    queued = await hint(entry["session_id"], "ACTIVE", confidence=1.0,
                        reason="actual ALL dependencies converged")
    # In a real adapter the model request is submitted here. No model is called.
    submitted = runner.elapsed()
    runner.workflow_event("synthetic_resume_submitted")
    await hint(entry["session_id"], "COMPLETED", confidence=1.0,
               reason="synthetic model step completed")
    await dag_event(entry["session_id"], "COMPLETED", {"kind": "workflow_finish"},
                    workflow_id=entry["workflow_id"])
    outcome = await drain(entry["workflow_id"])
    lifecycle_states = [a["state"] for a in acknowledgements if a["kind"] == "lifecycle"]
    if not native:
        assert lifecycle_states == ["TOOL_WAITING"] * 4 + ["ACTIVE"] * 2 + ["COMPLETED"] * 2
    assert metadata["workers_started"][entry["workflow_id"]] == 1
    assert metadata["completed_queue_items"] == 3
    assert metadata["control_failures"] == 0
    assert outcome["status"] == "complete"
    return {
        "classification": "synthetic_dispatch_demo_no_inference_no_gpu_no_network",
        "native_suppression": native,
        "ready_to_synthetic_submit_ms": (submitted - ready) * 1000,
        "enqueue_success_scope": queued[0]["success_scope"],
        "server_kv_ready_proven": False,
        "metadata": metadata, "drain": outcome,
        "control_acknowledgements": acknowledgements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-delay-ms", type=float, default=20.0)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if args.control_delay_ms < 0 or args.control_delay_ms > 500:
        parser.error("control delay must be between 0 and 500 ms")
    print(json.dumps(asyncio.run(demonstrate(args.control_delay_ms, args.native)), indent=2))


if __name__ == "__main__":
    main()
