"""Ordered lifecycle hint dispatch, extracted from the measured client repair.

The queue implementation is preserved from the archived client. Private
harness loading, deployment addresses and experiment inputs are omitted.
Queued work is not proof of server-side KV readiness.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable

CONTROL_DRAIN_TIMEOUT_S = 10.0
CONTROL_CANCEL_WAIT_S = 0.25

def workflow_control_queue(
    runner: Any,
    base_lifecycle_hint: Callable[..., Awaitable[Any]],
    base_dag_event: Callable[..., Awaitable[Any]],
) -> tuple[Any, Any, Any, dict[str, Any], Any]:
    """Install one ordered background control worker per workflow.

    The queued sequence is every actual dependency completion, followed by
    ACTIVE.  Calls before tool startup remain synchronous.  The worker calls
    the frozen wrapper methods, so Native retains its original suppression.
    """
    queues: dict[str, deque[tuple[str, dict[str, Any]]]] = defaultdict(deque)
    worker_state: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "worker": None,
        "wake": asyncio.Event(),
        "active_queued": False,
        "closed": False,
        "uncertain_in_flight_items": 0,
    })
    metadata: dict[str, Any] = {
        "queued_control_calls": 0,
        "completed_queue_items": 0,
        "successful_control_calls": 0,
        "failed_control_calls": 0,
        "control_failures": 0,
        "active_after_pending": {},
        "post_active_updates_suppressed": 0,
        "workers_started": {},
        "uncertain_in_flight_items": 0,
        "io_thread_cancellation_limitation": (
            "Cancelling the asyncio task cannot stop an already-running "
            "requests thread; the original HTTP timeout remains the I/O bound"
        ),
    }

    def workflow_for_session(session_id: str) -> str:
        # spec['entries'] is a list of entry dictionaries.  It must not be
        # unpacked as (entry_id, entry).
        for entry in runner.spec.get("entries", []):
            if entry.get("session_id") == session_id:
                return entry["workflow_id"]
        return session_id

    def start_worker(workflow_id: str) -> None:
        state = worker_state[workflow_id]
        if state["worker"] is None:
            state["worker"] = asyncio.create_task(
                publish_worker(workflow_id),
                name=f"projecta-control-fifo-{workflow_id}",
            )
            metadata["workers_started"][workflow_id] = 1
        state["wake"].set()

    def record_failure(
        workflow_id: str, kind: str, stage: str, error: Exception
    ) -> None:
        metadata["control_failures"] += 1
        metadata["failed_control_calls"] += 1
        runner.workflow_event(
            "control_dispatch_error",
            workflow_id=workflow_id,
            kind=kind,
            stage=stage,
            error=repr(error),
        )

    async def call_control(
        workflow_id: str, kind: str, stage: str,
        call: Callable[[], Awaitable[Any]],
    ) -> None:
        try:
            await call()
            metadata["successful_control_calls"] += 1
        except asyncio.CancelledError:
            raise
        except Exception as error:
            record_failure(workflow_id, kind, stage, error)

    async def publish_worker(workflow_id: str) -> None:
        queue = queues[workflow_id]
        state = worker_state[workflow_id]
        while True:
            while queue:
                kind, fields = queue.popleft()
                stage = "not_started"
                try:
                    if kind == "dependency":
                        stage = "lifecycle_hint"
                        await call_control(
                            workflow_id, kind, stage,
                            lambda: base_lifecycle_hint(
                                fields["session_id"], "TOOL_WAITING",
                                **fields["lifecycle"],
                            ),
                        )
                        stage = "dag_event"
                        await call_control(
                            workflow_id, kind, stage,
                            lambda: base_dag_event(
                                fields["session_id"], "TOOL_WAITING",
                                fields["dag"], workflow_id=workflow_id,
                            ),
                        )
                    elif kind == "active":
                        stage = "lifecycle_hint"
                        await call_control(
                            workflow_id, kind, stage,
                            lambda: base_lifecycle_hint(
                                fields["session_id"], "ACTIVE",
                                **fields["lifecycle"],
                            ),
                        )
                    metadata["completed_queue_items"] += 1
                except asyncio.CancelledError:
                    if stage == "not_started":
                        queue.appendleft((kind, fields))
                    else:
                        state["uncertain_in_flight_items"] += 1
                        metadata["uncertain_in_flight_items"] += 1
                        runner.workflow_event(
                            "control_dispatch_cancelled_in_flight",
                            workflow_id=workflow_id,
                            kind=kind,
                            stage=stage,
                            pending_items=len(queue),
                            io_thread_may_continue=True,
                        )
                    raise
            if state["active_queued"] or state["closed"]:
                break
            await state["wake"].wait()
            state["wake"].clear()

    async def queued_lifecycle_hint(
        session_id: str, state: str, **fields: Any
    ) -> list[dict[str, Any]]:
        workflow_id = workflow_for_session(session_id)
        if state != "ACTIVE" or fields.get("reason") != "actual ALL dependencies converged":
            if state == "COMPLETED":
                outcome = await drain(workflow_id)
                if outcome["status"] != "complete":
                    runner.workflow_event(
                        "workflow_control_fence_failed",
                        workflow_id=workflow_id, **outcome,
                    )
                    raise RuntimeError(
                        "control FIFO did not finish before COMPLETED: "
                        f"{outcome['status']}"
                    )
            return await base_lifecycle_hint(session_id, state, **fields)

        queue_state = worker_state[workflow_id]
        duplicate = queue_state["active_queued"]
        if not duplicate:
            queues[workflow_id].append(("active", {
                "session_id": session_id,
                "lifecycle": fields,
            }))
            queue_state["active_queued"] = True
            metadata["queued_control_calls"] += 1
            metadata["active_after_pending"][workflow_id] = len(queues[workflow_id]) - 1
            start_worker(workflow_id)
        row = {
            "event": "native_lifecycle_hint_queued", "role": "P,D",
            "session_id": session_id, "state": state, "fields": fields,
            "success": True, "queued": True, "duplicate": duplicate,
            "success_scope": "enqueue_only", "awaited_by_model": False,
            "final_ack_pending": True, "product_ack_success": None,
            "result": {"queued": True, "final_ack_pending": True},
            "started_wall": time.time(), "ended_wall": time.time(),
        }
        runner.decision_writer.write(row)
        return [row]

    async def queued_dag_event(
        session_id: str, state: str, event: dict[str, Any],
        *, workflow_id: str | None = None,
    ) -> list[dict[str, Any]]:
        resolved_workflow_id = (
            workflow_id or event.get("workflow_id") or workflow_for_session(session_id)
        )
        if state == "COMPLETED" and event.get("kind") == "workflow_finish":
            outcome = await drain(resolved_workflow_id)
            runner.workflow_event(
                "workflow_control_fence", workflow_id=resolved_workflow_id, **outcome
            )
            if outcome["status"] != "complete":
                raise RuntimeError(
                    "control FIFO did not finish before workflow_finish: "
                    f"{outcome['status']}"
                )
        return await base_dag_event(
            session_id, state, event, workflow_id=workflow_id
        )

    async def run_tool(
        entry: dict[str, Any], tool_id: str, *,
        cancel: bool = False, remaining_after: int | None = None,
    ) -> dict[str, Any]:
        actual = entry["actual_tool_seconds"][tool_id]
        tool_service_started_s = runner.elapsed()
        runner.workflow_event(
            "tool_started", workflow_id=entry["workflow_id"], tool_id=tool_id,
            planned_duration_s=actual, actual_duration_s=actual,
            cancelled=cancel, tool_service_started_s=tool_service_started_s,
        )
        if cancel:
            runner.workflow_event(
                "tool_cancelled", workflow_id=entry["workflow_id"],
                tool_id=tool_id, planned_duration_s=actual,
            )
            return {"tool_id": tool_id, "status": "cancelled", "actual_duration_s": 0.0}

        await runner.tool_delay(actual)
        finished_s = runner.elapsed()
        runner.workflow_event(
            "tool_completed", workflow_id=entry["workflow_id"], tool_id=tool_id,
            planned_duration_s=actual,
            actual_duration_s=finished_s - tool_service_started_s,
            finished_s=finished_s,
        )
        workflow_id = entry["workflow_id"]
        async with runner.dependency_locks.setdefault(workflow_id, asyncio.Lock()):
            done = runner.completed_dependencies.setdefault(workflow_id, set())
            done.add(tool_id)
            remaining = entry["required_dependencies"] - len(done)
            if remaining <= 0 and workflow_id in runner.convergence_events:
                runner.convergence_events[workflow_id].set()

            if not worker_state[workflow_id]["active_queued"]:
                queues[workflow_id].append(("dependency", {
                    "session_id": entry["session_id"],
                    "lifecycle": {
                        "expected_resume_at": runner.expected_resume_by_workflow[workflow_id],
                        "remaining_dependencies": remaining,
                        "confidence": 0.9,
                        "reason": "actual dependency completed",
                    },
                    "dag": {
                        "kind": "dependency_complete",
                        "consumer_id": f"{workflow_id}-resume",
                        "step_id": "resume",
                        "prefix_tokens": entry["prefix_tokens"],
                        "next_use_distance": 0,
                        "confidence": 1.0,
                    },
                }))
                metadata["queued_control_calls"] += 1
                start_worker(workflow_id)
            else:
                metadata["post_active_updates_suppressed"] += 1
        return {
            "tool_id": tool_id, "status": "completed",
            "actual_duration_s": finished_s - tool_service_started_s,
            "finished_s": finished_s,
        }

    async def drain(workflow_id: str, timeout_s: float = CONTROL_DRAIN_TIMEOUT_S) -> dict[str, Any]:
        started = time.monotonic()
        state = worker_state[workflow_id]
        state["closed"] = True
        state["wake"].set()
        worker = state["worker"]
        status = "complete"
        timeout = False
        try:
            if worker is None:
                if queues[workflow_id]:
                    status = "worker_missing"
            elif not worker.done():
                _, pending = await asyncio.wait({worker}, timeout=timeout_s)
                if pending:
                    raise asyncio.TimeoutError
        except asyncio.TimeoutError:
            timeout = True
            worker.cancel()
            try:
                _, pending = await asyncio.wait(
                    {worker}, timeout=CONTROL_CANCEL_WAIT_S
                )
            except asyncio.TimeoutError:
                status = "timeout_worker_uncancellable"
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            else:
                if pending:
                    status = "timeout_worker_uncancellable"
                else:
                    status = (
                        "complete"
                        if worker.done() and not worker.cancelled()
                        else "timeout_cancelled"
                    )
        except asyncio.CancelledError:
            worker.cancel()
            try:
                _, pending = await asyncio.wait(
                    {worker}, timeout=CONTROL_CANCEL_WAIT_S
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                status = "cancelled_worker_uncancellable"
            else:
                status = "cancelled"
            raise

        if worker is not None and worker.done():
            if worker.cancelled():
                status = "cancelled"
            elif worker.exception() is not None:
                status = "error"
            elif queues[workflow_id]:
                status = "incomplete"
        elif worker is not None:
            status = "timeout_worker_uncancellable"
        if state["uncertain_in_flight_items"]:
            status = "incomplete_after_uncertain_io"

        async_task_remaining = worker is not None and not worker.done()
        return {
            "status": status,
            "timeout": timeout,
            "elapsed_s": time.monotonic() - started,
            "pending_after_drain": len(queues[workflow_id]),
            "uncertain_after_drain": state["uncertain_in_flight_items"],
            "active_queued": state["active_queued"],
            "async_task_remaining": async_task_remaining,
            "io_thread_may_continue": bool(
                state["uncertain_in_flight_items"] or async_task_remaining
            ),
            "control_failures": metadata["control_failures"],
            "all_control_calls_successful": metadata["control_failures"] == 0,
        }

    return run_tool, queued_lifecycle_hint, queued_dag_event, metadata, drain
