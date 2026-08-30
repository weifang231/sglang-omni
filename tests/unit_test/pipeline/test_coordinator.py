# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import gc

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.config import PipelineConfig, ProcessConfig, QueueControlConfig
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.replicas import ReplicaTopology, expand_replica_stages
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage
from sglang_omni.runtime_queue import (
    FIRST_OUTPUT_DEADLINE_METADATA_KEY,
    REQUEST_CLASS_METADATA_KEY,
)
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane
from tests.unit_test.pipeline.helpers import stage


def test_coordinator_multi_terminal_completion_and_abort_contracts() -> None:
    """Preserves multi-terminal completion and abort cancellation semantics."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", {"text": "hello"})
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert not coordinator._completion_futures["req-1"].done()
        await coordinator._handle_completion(
            CompleteMessage("req-1", "code2wav", True, result={"audio": "ok"})
        )
        assert coordinator._completion_futures["req-1"].result() == {
            "decode": {"text": "hi"},
            "code2wav": {"audio": "ok"},
        }

        await coordinator._submit_request("req-2", "hello")
        future = coordinator._completion_futures["req-2"]
        assert await coordinator.abort("req-2") is True
        assert control_plane.aborts[0].request_id == "req-2"
        with pytest.raises(asyncio.CancelledError):
            await future

    asyncio.run(_run())


def test_coordinator_resolves_active_terminal_subset_per_request() -> None:
    async def _run() -> None:
        def terminal_stages(request: OmniRequest) -> list[str]:
            assert isinstance(request, OmniRequest)
            if request.metadata.get("audio"):
                return ["decode", "code2wav"]
            return ["decode"]

        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=terminal_stages,
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request(
            "text-req",
            OmniRequest(inputs="hello", metadata={"audio": False}),
        )
        await coordinator._handle_completion(
            CompleteMessage("text-req", "decode", True, result={"text": "hi"})
        )
        assert coordinator._completion_futures["text-req"].result() == {"text": "hi"}

        await coordinator._submit_request("raw-text-req", "hello")
        await coordinator._handle_completion(
            CompleteMessage("raw-text-req", "decode", True, result={"text": "raw"})
        )
        assert coordinator._completion_futures["raw-text-req"].result() == {
            "text": "raw"
        }

        await coordinator._submit_request(
            "audio-req",
            OmniRequest(inputs="hello", metadata={"audio": True}),
        )
        await coordinator._handle_completion(
            CompleteMessage("audio-req", "decode", True, result={"text": "hi"})
        )
        assert not coordinator._completion_futures["audio-req"].done()
        await coordinator._handle_completion(
            CompleteMessage(
                "audio-req",
                "code2wav",
                True,
                result={"audio": "ok"},
            )
        )
        assert coordinator._completion_futures["audio-req"].result() == {
            "decode": {"text": "hi"},
            "code2wav": {"audio": "ok"},
        }

    asyncio.run(_run())


def test_coordinator_rejects_invalid_resolved_terminal_subset() -> None:
    async def _run() -> None:
        for resolved, error in (
            ([], "no terminal stages"),
            (["decode", "missing"], "outside the static terminal stages"),
            ("decode", "must return a sequence"),
        ):
            coordinator = Coordinator(
                "inproc://complete",
                "inproc://abort",
                entry_stage="preprocess",
                terminal_stages=["decode", "code2wav"],
                terminal_stages_resolver=lambda request, resolved=resolved: resolved,
            )
            coordinator.control_plane = RecordingCoordinatorControlPlane()
            coordinator.register_stage("preprocess", "inproc://preprocess")

            with pytest.raises(ValueError, match=error):
                await coordinator._submit_request("req-1", OmniRequest(inputs="hello"))
            assert coordinator._requests == {}
            assert coordinator.control_plane.submitted == []

    asyncio.run(_run())


def test_coordinator_stream_cleans_queue_when_terminal_resolver_rejects() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=lambda request: [],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", OmniRequest(inputs="hello"))
        with pytest.raises(ValueError, match="no terminal stages"):
            await stream.__anext__()
        await stream.aclose()

        assert coordinator._stream_queues == {}
        assert coordinator._completion_futures == {}
        assert coordinator.control_plane.submitted == []

    asyncio.run(_run())


def test_coordinator_stream_uses_request_terminal_subset_after_cleanup() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=lambda request: ["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        events = []

        async def _consume() -> None:
            async for event in coordinator.stream("req-1", OmniRequest(inputs="hello")):
                events.append(event)

        task = asyncio.create_task(_consume())
        for _ in range(10):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        await asyncio.wait_for(task, timeout=1)

        assert [event.from_stage for event in events] == ["decode"]

    asyncio.run(_run())


def test_coordinator_stream_received_event_pairs_terminal_chunk(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.pipeline.coordinator._emit_event",
        lambda **kwargs: events.append(kwargs),
    )

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        queue: asyncio.Queue = asyncio.Queue()
        coordinator._stream_queues["req-1"] = queue

        await coordinator._handle_stream(
            StreamMessage(
                request_id="req-1",
                from_stage="decode",
                chunk={"text": "hi"},
                modality="text",
                chunk_id=1,
            )
        )

        routed = queue.get_nowait()
        assert routed.chunk_id == 1

    asyncio.run(_run())

    receive_events = [
        event
        for event in events
        if event["event_name"] == "stage_stream_chunk_received"
    ]
    assert len(receive_events) == 1
    assert receive_events[0]["stage"] == "coordinator"
    assert receive_events[0]["metadata"] == {
        "from_stage": "decode",
        "chunk_id": 1,
        "modality": "text",
    }


def test_stream_message_round_trips_terminal_chunk_id() -> None:
    msg = StreamMessage(
        request_id="req-1",
        from_stage="decode",
        chunk={"text": "hi"},
        modality="text",
        chunk_id=3,
    )

    round_trip = StreamMessage.from_dict(msg.to_dict())

    assert round_trip.chunk_id == 3
    assert round_trip.modality == "text"


def test_coordinator_failure_completion_fails_fast_and_cleans_state() -> None:
    """Preserves fail-fast behavior and cleanup after any terminal failure."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", "hello")
        future = coordinator._completion_futures["req-1"]
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert coordinator._partial_results["req-1"] == {"decode": {"text": "hi"}}

        await coordinator._handle_completion(
            CompleteMessage("req-1", "code2wav", False, error="boom")
        )

        with pytest.raises(RuntimeError, match="boom"):
            await future
        assert "req-1" not in coordinator._requests
        assert "req-1" not in coordinator._partial_results
        assert control_plane.aborts[-1].request_id == "req-1"

    asyncio.run(_run())


def test_coordinator_fail_pending_requests_resolves_waiters() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", "hello")
        future = coordinator._completion_futures["req-1"]

        await coordinator.fail_pending_requests(RuntimeError("stage died"))

        with pytest.raises(RuntimeError, match="stage died"):
            await future
        assert coordinator._requests == {}
        assert coordinator._partial_results == {}

    asyncio.run(_run())


async def _drive_stream_until_registered(coordinator: Coordinator, request_id: str):
    """Start consuming a stream and return (task, error_sink, future) once the
    request's completion future has been created."""
    error_sink: list[str] = []

    async def _consume() -> None:
        try:
            async for _msg in coordinator.stream(request_id, "hello"):
                pass
        except RuntimeError as exc:
            error_sink.append(str(exc))

    task = asyncio.create_task(_consume())
    for _ in range(100):
        if request_id in coordinator._completion_futures:
            break
        await asyncio.sleep(0)
    future = coordinator._completion_futures[request_id]
    return task, error_sink, future


def test_coordinator_stream_early_close_aborts_and_cleans_state() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", OmniRequest(inputs="hello"))
        first_chunk = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req-1" in coordinator._stream_queues:
                break
            await asyncio.sleep(0)
        await coordinator._handle_stream(
            StreamMessage(
                request_id="req-1",
                from_stage="decode",
                chunk={"text": "hello"},
                modality="text",
            )
        )
        await first_chunk
        await stream.aclose()

        assert [msg.request_id for msg in control_plane.aborts] == ["req-1"]
        assert "req-1" not in coordinator._requests
        assert "req-1" not in coordinator._stream_queues
        assert "req-1" not in coordinator._completion_futures

    asyncio.run(_run())


def test_stream_close_after_one_terminal_aborts_remaining_terminal_work() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", "hello")
        first_terminal = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "done"})
        )
        assert (await first_terminal).from_stage == "decode"
        assert coordinator._partial_results["req-1"] == {"decode": {"text": "done"}}

        await stream.aclose()

        assert [msg.request_id for msg in control_plane.aborts] == ["req-1"]
        assert coordinator._requests == {}
        assert coordinator._partial_results == {}
        assert coordinator._completion_futures == {}
        assert coordinator._stream_queues == {}

    asyncio.run(_run())


def test_coordinator_stream_natural_completion_does_not_abort() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        async def _consume() -> list[CompleteMessage | StreamMessage]:
            return [
                message
                async for message in coordinator.stream(
                    "req-1", OmniRequest(inputs="hello")
                )
            ]

        task = asyncio.create_task(_consume())
        for _ in range(100):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)
        await coordinator._handle_completion(
            CompleteMessage(
                request_id="req-1",
                from_stage="decode",
                success=True,
                result={"text": "hello"},
            )
        )
        messages = await task

        assert len(messages) == 1
        assert control_plane.aborts == []
        assert "req-1" not in coordinator._requests
        assert "req-1" not in coordinator._stream_queues
        assert "req-1" not in coordinator._completion_futures

    asyncio.run(_run())


def test_duplicate_stream_preserves_existing_non_stream_request() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", "original")
        original_request = coordinator._requests["req-1"]
        original_future = coordinator._completion_futures["req-1"]

        duplicate = coordinator.stream("req-1", "duplicate")
        with pytest.raises(ValueError, match="already exists"):
            await anext(duplicate)

        assert coordinator._requests["req-1"] is original_request
        assert coordinator._completion_futures["req-1"] is original_future
        assert "req-1" not in coordinator._stream_queues
        assert control_plane.aborts == []

        assert await coordinator.abort("req-1") is True
        with pytest.raises(asyncio.CancelledError):
            await original_future

    asyncio.run(_run())


def test_completed_stream_allows_request_id_reuse_after_owner_closes() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", "original")
        terminal_event = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "done"})
        )
        assert (await terminal_event).result == {"text": "done"}

        old_future = coordinator._completion_futures["req-1"]
        old_queue = coordinator._stream_queues["req-1"]
        assert "req-1" not in coordinator._requests

        with pytest.raises(ValueError, match="already exists"):
            await coordinator._submit_request("req-1", "replacement")
        assert coordinator._completion_futures["req-1"] is old_future
        assert coordinator._stream_queues["req-1"] is old_queue

        await stream.aclose()
        assert "req-1" not in coordinator._completion_futures
        assert "req-1" not in coordinator._stream_queues
        await coordinator._submit_request("req-1", "replacement")
        assert coordinator._requests["req-1"].request_id == "req-1"

    asyncio.run(_run())


def test_stream_abort_reserves_request_id_while_broadcast_is_in_flight() -> None:
    class BlockingAbortControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.abort_started = asyncio.Event()
            self.release_abort = asyncio.Event()

        async def broadcast_abort(self, msg) -> None:
            self.aborts.append(msg)
            self.abort_started.set()
            await self.release_abort.wait()

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = BlockingAbortControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", "original")
        first_chunk = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)
        await coordinator._handle_stream(
            StreamMessage(
                request_id="req-1",
                from_stage="decode",
                chunk={"text": "partial"},
                modality="text",
            )
        )
        await first_chunk

        close_task = asyncio.create_task(stream.aclose())
        await control_plane.abort_started.wait()

        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "done"})
        )
        assert "req-1" not in coordinator._requests
        assert "req-1" in coordinator._abort_tasks

        with pytest.raises(ValueError, match="already exists"):
            await coordinator._submit_request("req-1", "replacement")

        control_plane.release_abort.set()
        await close_task
        assert coordinator._abort_tasks == {}
        assert coordinator._completion_futures == {}
        assert coordinator._stream_queues == {}

    asyncio.run(_run())


def test_stream_cancellation_is_preserved_after_abort_cleanup() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        next_event = asyncio.create_task(anext(coordinator.stream("req-1", "hello")))
        for _ in range(100):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)

        next_event.cancel()
        with pytest.raises(asyncio.CancelledError):
            await next_event

        assert [msg.request_id for msg in control_plane.aborts] == ["req-1"]
        assert coordinator._requests == {}
        assert coordinator._completion_futures == {}
        assert coordinator._stream_queues == {}
        assert coordinator._abort_tasks == {}

    asyncio.run(_run())


def test_coordinator_stream_abort_failure_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingAbortControlPlane(RecordingCoordinatorControlPlane):
        async def broadcast_abort(self, msg) -> None:
            self.aborts.append(msg)
            raise RuntimeError("abort transport unavailable")

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = FailingAbortControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", OmniRequest(inputs="hello"))
        first_chunk = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req-1" in coordinator._stream_queues:
                break
            await asyncio.sleep(0)
        await coordinator._handle_stream(
            StreamMessage(
                request_id="req-1",
                from_stage="decode",
                chunk={"text": "hello"},
                modality="text",
            )
        )
        await first_chunk
        await stream.aclose()

        assert "req-1" not in coordinator._stream_queues
        assert "req-1" not in coordinator._completion_futures

    with caplog.at_level("WARNING"):
        asyncio.run(_run())
    assert "Failed to abort request req-1" in caplog.text


def test_coordinator_stream_abort_cancels_future_without_unretrieved_exception() -> (
    None
):
    """Aborting a streaming request cancels its completion future instead of
    setting an exception no one retrieves, so the event loop never reports a
    'Future exception was never retrieved' error."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        loop = asyncio.get_running_loop()
        handler_contexts: list = []
        loop.set_exception_handler(
            lambda _loop, context: handler_contexts.append(context)
        )

        task, error_sink, future = await _drive_stream_until_registered(
            coordinator, "req-1"
        )

        assert await coordinator.abort("req-1") is True
        await asyncio.wait_for(task, timeout=1)

        # Stream terminated via its queue; the future is cancelled rather than
        # carrying an un-retrieved exception.
        assert error_sink == ["aborted"]
        assert future.cancelled() is True
        assert "req-1" not in coordinator._completion_futures

        # Dropping the future must not trip the loop's exception handler.
        del future
        gc.collect()
        assert not any(
            "never retrieved" in str(ctx.get("message", "")) for ctx in handler_contexts
        )

    asyncio.run(_run())


def test_coordinator_stream_fail_pending_requests_cancels_future() -> None:
    """A coordinator failure reaches the stream without leaving an exception
    on its unused completion future."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        loop = asyncio.get_running_loop()
        handler_contexts: list = []
        loop.set_exception_handler(
            lambda _loop, context: handler_contexts.append(context)
        )

        task, error_sink, future = await _drive_stream_until_registered(
            coordinator, "req-1"
        )

        await coordinator.fail_pending_requests(RuntimeError("stage died"))
        await asyncio.wait_for(task, timeout=1)

        assert error_sink == ["stage died"]
        assert future.cancelled() is True
        assert "req-1" not in coordinator._completion_futures

        del future
        gc.collect()
        assert not any(
            "never retrieved" in str(ctx.get("message", "")) for ctx in handler_contexts
        )

    asyncio.run(_run())


def test_coordinator_stream_stage_failure_cancels_future() -> None:
    """A stage failure on a streaming request cancels the completion future
    (which the stream consumer never awaits) rather than setting an exception
    that would be reported as never retrieved."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        task, error_sink, future = await _drive_stream_until_registered(
            coordinator, "req-1"
        )

        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", False, error="boom")
        )
        await asyncio.wait_for(task, timeout=1)

        assert error_sink == ["boom"]
        assert future.cancelled() is True
        assert "req-1" not in coordinator._completion_futures

    asyncio.run(_run())


def test_coordinator_rejects_submit_when_in_flight_cap_is_reached() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            max_in_flight=1,
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", "hello")
        with pytest.raises(QueueFullError, match="The request queue is full"):
            await coordinator._submit_request("req-2", "hello")

        assert [msg.request_id for _, _, msg in control_plane.submitted] == ["req-1"]
        assert list(coordinator._requests) == ["req-1"]

        await coordinator._handle_completion(
            CompleteMessage("req-1", "preprocess", True, result={"ok": True})
        )
        await coordinator._submit_request("req-2", "hello")
        assert [msg.request_id for _, _, msg in control_plane.submitted] == [
            "req-1",
            "req-2",
        ]

    asyncio.run(_run())


def test_runtime_waiters_do_not_consume_native_dispatch_capacity() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            max_in_flight=2,
            queue_control=QueueControlConfig(max_active_requests=1),
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("active", "one")
        await coordinator._submit_request("waiting-1", "two")
        await coordinator._submit_request("waiting-2", "three")

        assert [msg.request_id for _, _, msg in control_plane.submitted] == ["active"]
        snapshot = coordinator.health()["queue_control"]
        assert snapshot["max_active_requests"] == 1
        assert snapshot["max_waiting_requests"] == 2
        assert snapshot["active_requests"] == 1
        assert snapshot["waiting_requests"] == 2

        with pytest.raises(QueueFullError, match="The request queue is full"):
            await coordinator._submit_request("rejected", "four")
        assert "rejected" not in coordinator._requests
        assert "rejected" not in coordinator._completion_futures
        assert coordinator.health()["queue_control"]["waiting_rejected_total"] == 1

        await coordinator._handle_completion(
            CompleteMessage("active", "preprocess", True, result={})
        )
        assert [msg.request_id for _, _, msg in control_plane.submitted] == [
            "active",
            "waiting-1",
        ]
        await coordinator.abort("waiting-1")
        await coordinator.abort("waiting-2")

    asyncio.run(_run())


def test_runtime_active_limit_is_clamped_to_native_dispatch_capacity() -> None:
    coordinator = Coordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="preprocess",
        max_in_flight=2,
        queue_control=QueueControlConfig(
            max_active_requests=8,
            max_waiting_requests=4,
        ),
    )

    snapshot = coordinator.health()["queue_control"]
    assert snapshot["max_active_requests"] == 2
    assert snapshot["max_waiting_requests"] == 4

    snapshot = asyncio.run(
        coordinator.update_queue_control(
            QueueControlConfig(
                max_active_requests=16,
                max_waiting_requests=8,
            )
        )
    )
    assert snapshot["max_active_requests"] == 2
    assert snapshot["max_waiting_requests"] == 8


def test_coordinator_runtime_queue_enforces_class_credits_and_edf() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            queue_control=QueueControlConfig(
                max_active_requests=2,
                class_limits={"text": 1, "speech": 1},
                discipline="edf",
                trust_request_metadata=True,
            ),
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        def request(request_class: str, deadline: float) -> OmniRequest:
            return OmniRequest(
                inputs="hello",
                metadata={
                    REQUEST_CLASS_METADATA_KEY: request_class,
                    FIRST_OUTPUT_DEADLINE_METADATA_KEY: deadline,
                },
            )

        await coordinator._submit_request("text-active", request("text", 1.0))
        await coordinator._submit_request("text-wait", request("text", 2.0))
        await coordinator._submit_request("speech-active", request("speech", 9.0))
        await coordinator._submit_request("speech-late", request("speech", 8.0))
        await coordinator._submit_request("speech-early", request("speech", 3.0))
        assert [msg.request_id for _, _, msg in control_plane.submitted] == [
            "text-active",
            "speech-active",
        ]

        await coordinator._handle_completion(
            CompleteMessage("speech-active", "preprocess", True, result={})
        )
        assert [msg.request_id for _, _, msg in control_plane.submitted] == [
            "text-active",
            "speech-active",
            "speech-early",
        ]

        await coordinator._handle_completion(
            CompleteMessage("text-active", "preprocess", True, result={})
        )
        assert control_plane.submitted[-1][2].request_id == "text-wait"

        for request_id in ("text-wait", "speech-early", "speech-late"):
            if request_id in coordinator._requests:
                await coordinator.abort(request_id)

    asyncio.run(_run())


def test_coordinator_queue_update_and_waiting_abort_are_runtime_native() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            queue_control=QueueControlConfig(
                max_active_requests=1,
                trust_request_metadata=True,
            ),
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("r1", "one")
        await coordinator._submit_request("r2", "two")
        await coordinator._submit_request("r3", "three")
        assert [msg.request_id for _, _, msg in control_plane.submitted] == ["r1"]

        assert await coordinator.abort("r2") is True
        assert control_plane.aborts == []
        snapshot = await coordinator.update_queue_control(
            QueueControlConfig(max_active_requests=2)
        )
        assert snapshot["active_requests"] == 2
        assert [msg.request_id for _, _, msg in control_plane.submitted] == [
            "r1",
            "r3",
        ]

        await coordinator.abort("r1")
        await coordinator.abort("r3")

    asyncio.run(_run())


def test_coordinator_accepts_class_only_runtime_queue_update() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            queue_control=QueueControlConfig(
                max_active_requests=1,
                trust_request_metadata=True,
            ),
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        snapshot = await coordinator.update_queue_control(
            {"discipline": "edf", "class_limits": {" gold ": 1}}
        )

        assert snapshot["max_active_requests"] is None
        assert snapshot["class_limits"] == {"gold": 1}

    asyncio.run(_run())


def test_coordinator_failed_active_abort_does_not_orphan_next_credit() -> None:
    class FailingAbortControlPlane(RecordingCoordinatorControlPlane):
        async def broadcast_abort(self, msg) -> None:
            del msg
            raise RuntimeError("abort transport failed")

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            queue_control=QueueControlConfig(max_active_requests=1),
        )
        control_plane = FailingAbortControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("r1", "one")
        await coordinator._submit_request("r2", "two")
        with pytest.raises(RuntimeError, match="abort transport failed"):
            await coordinator.abort("r1")

        snapshot = coordinator.health()["queue_control"]
        assert snapshot["active_requests"] == 1
        assert snapshot["waiting_requests"] == 1
        assert [msg.request_id for _, _, msg in control_plane.submitted] == ["r1"]

    asyncio.run(_run())


def test_coordinator_cancelled_dispatch_rolls_back_credit() -> None:
    class BlockingSubmitControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def submit_to_stage(self, stage, endpoint, msg) -> None:
            del stage, endpoint, msg
            self.entered.set()
            await self.release.wait()

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            queue_control=QueueControlConfig(max_active_requests=1),
        )
        control_plane = BlockingSubmitControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        submit = asyncio.create_task(coordinator.submit("r1", "one"))
        await control_plane.entered.wait()
        submit.cancel()
        await asyncio.sleep(0)
        control_plane.release.set()
        with pytest.raises(asyncio.CancelledError):
            await submit

        snapshot = coordinator.health()["queue_control"]
        assert snapshot["active_requests"] == 0
        assert snapshot["waiting_requests"] == 0
        assert coordinator._requests == {}
        assert coordinator._completion_futures == {}
        assert [msg.request_id for msg in control_plane.aborts] == ["r1"]

    asyncio.run(_run())


def test_coordinator_cancelled_dispatch_retains_credit_when_abort_is_uncertain() -> None:
    class BlockingSubmitAndFailingAbort(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def submit_to_stage(self, stage, endpoint, msg) -> None:
            del stage, endpoint, msg
            self.entered.set()
            await self.release.wait()

        async def broadcast_abort(self, msg) -> None:
            del msg
            raise RuntimeError("abort transport failed")

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            queue_control=QueueControlConfig(max_active_requests=1),
        )
        control_plane = BlockingSubmitAndFailingAbort()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        submit = asyncio.create_task(coordinator.submit("r1", "one"))
        await control_plane.entered.wait()
        submit.cancel()
        await asyncio.sleep(0)
        control_plane.release.set()
        with pytest.raises(asyncio.CancelledError):
            await submit

        snapshot = coordinator.health()["queue_control"]
        assert snapshot["active_requests"] == 1
        assert snapshot["waiting_requests"] == 0
        assert list(coordinator._requests) == ["r1"]
        assert coordinator._completion_futures == {}

    asyncio.run(_run())


def test_coordinator_cancelled_waiter_is_removed_without_orphaning_queue() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            queue_control=QueueControlConfig(max_active_requests=1),
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("active", "one")
        waiting = asyncio.create_task(coordinator.submit("waiting", "two"))
        await asyncio.sleep(0)
        assert coordinator.health()["queue_control"]["waiting_requests"] == 1

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        snapshot = coordinator.health()["queue_control"]
        assert snapshot["active_requests"] == 1
        assert snapshot["waiting_requests"] == 0
        assert list(coordinator._requests) == ["active"]
        assert [msg.request_id for _, _, msg in control_plane.submitted] == ["active"]
        await coordinator.abort("active")

    asyncio.run(_run())


def test_coordinator_serializes_pregranted_dispatch_batches() -> None:
    class BlockingOneSubmitControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.a_entered = asyncio.Event()
            self.release_a = asyncio.Event()

        async def submit_to_stage(self, stage, endpoint, msg) -> None:
            self.submitted.append((stage, endpoint, msg))
            if msg.request_id == "a":
                self.a_entered.set()
                await self.release_a.wait()

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            queue_control=QueueControlConfig(max_active_requests=1),
        )
        control_plane = BlockingOneSubmitControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("x", "active")
        await coordinator._submit_request("a", "first waiter")
        await coordinator._submit_request("b", "second waiter")
        await coordinator._submit_request("d", "third waiter")

        update = asyncio.create_task(
            coordinator.update_queue_control(
                QueueControlConfig(max_active_requests=3)
            )
        )
        await control_plane.a_entered.wait()
        completion = asyncio.create_task(
            coordinator._handle_completion(
                CompleteMessage("x", "preprocess", True, result={})
            )
        )
        await asyncio.sleep(0)
        control_plane.release_a.set()
        await update
        await completion

        assert [msg.request_id for _, _, msg in control_plane.submitted] == [
            "x",
            "a",
            "b",
            "d",
        ]

        for request_id in ("a", "b", "d"):
            await coordinator.abort(request_id)

    asyncio.run(_run())


def test_cancelled_queue_update_does_not_abort_dispatched_user_request() -> None:
    class BlockingSubmitControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def submit_to_stage(self, stage, endpoint, msg) -> None:
            self.submitted.append((stage, endpoint, msg))
            if msg.request_id == "waiting":
                self.entered.set()
                await self.release.wait()

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            queue_control=QueueControlConfig(max_active_requests=1),
        )
        control_plane = BlockingSubmitControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("active", "one")
        await coordinator._submit_request("waiting", "two")
        update = asyncio.create_task(
            coordinator.update_queue_control(
                QueueControlConfig(max_active_requests=2)
            )
        )
        await control_plane.entered.wait()
        update.cancel()
        await asyncio.sleep(0)
        control_plane.release.set()
        with pytest.raises(asyncio.CancelledError):
            await update

        assert [msg.request_id for _, _, msg in control_plane.submitted] == [
            "active",
            "waiting",
        ]
        assert control_plane.aborts == []
        assert coordinator.health()["queue_control"]["active_requests"] == 2
        await coordinator.abort("active")
        await coordinator.abort("waiting")

    asyncio.run(_run())


def test_coordinator_runtime_credit_telemetry_balances_after_drain(
    monkeypatch,
) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.pipeline.coordinator._emit_event",
        lambda **kwargs: events.append(kwargs),
    )

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            queue_control=QueueControlConfig(max_active_requests=1),
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("r1", "one")
        await coordinator._submit_request("r2", "two")
        await coordinator._handle_completion(
            CompleteMessage("r1", "preprocess", True, result={})
        )
        await coordinator._handle_completion(
            CompleteMessage("r2", "preprocess", True, result={})
        )

        assert coordinator.health()["queue_control"]["active_requests"] == 0
        assert coordinator.health()["queue_control"]["waiting_requests"] == 0

    asyncio.run(_run())

    acquired = [
        event for event in events if event["event_name"] == "runtime_credit_acquired"
    ]
    released = [
        event for event in events if event["event_name"] == "runtime_credit_released"
    ]
    assert [event["request_id"] for event in acquired] == ["r1", "r2"]
    assert [event["request_id"] for event in released] == ["r1", "r2"]
    assert all(
        {
            "scope",
            "request_class",
            "active_requests",
            "waiting_requests",
        }
        <= event["metadata"].keys()
        for event in acquired + released
    )


def test_admin_resolves_logical_replica_target_to_all_instances() -> None:
    coordinator = Coordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="preprocess",
        replica_topology=ReplicaTopology(
            replicas={"talker_ar": ("talker_ar@r0", "talker_ar@r1")}
        ),
    )
    coordinator.control_plane = RecordingCoordinatorControlPlane()
    coordinator.register_stage("talker_ar@r0", "inproc://t0")
    coordinator.register_stage("talker_ar@r1", "inproc://t1")
    coordinator.register_stage("thinker", "inproc://thinker")

    assert coordinator._resolve_admin_stages(["talker_ar"]) == [
        "talker_ar@r0",
        "talker_ar@r1",
    ]
    assert coordinator._resolve_admin_stages(
        ["talker_ar", "talker_ar@r0", "thinker"]
    ) == ["talker_ar@r0", "talker_ar@r1", "thinker"]
    assert coordinator._resolve_admin_stages(None) == [
        "talker_ar@r0",
        "talker_ar@r1",
        "thinker",
    ]
    with pytest.raises(ValueError, match="Unknown admin target"):
        coordinator._resolve_admin_stages(["nope"])


def test_coordinator_normalizes_replica_instance_name_on_stream_chunk() -> None:
    async def _run() -> None:
        logical_plan, replica_topology = _multi_terminal_replica_runtime()
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["code2wav"],
            replica_topology=replica_topology,
            logical_process_plan=logical_plan,
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")
        queue: asyncio.Queue = asyncio.Queue()
        await coordinator._submit_request("req-1", "hello", stream_queue=queue)
        bindings = coordinator.control_plane.submitted[0][2].replica_bindings
        instance = replica_topology.resolve("code2wav", bindings["code2wav"])

        await coordinator._handle_stream(
            StreamMessage(
                request_id="req-1",
                from_stage=instance,
                chunk={"audio": "x"},
                stage_name=instance,
                modality="audio",
                chunk_id=0,
            )
        )

        routed = queue.get_nowait()
        assert routed.from_stage == "code2wav"
        assert routed.stage_name == "code2wav"

    asyncio.run(_run())


@pytest.mark.parametrize("success", [True, False])
def test_coordinator_normalizes_replica_instance_name_on_completion(
    success: bool,
) -> None:
    async def _run() -> None:
        logical_plan, replica_topology = _multi_terminal_replica_runtime()
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["code2wav"],
            replica_topology=replica_topology,
            logical_process_plan=logical_plan,
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        queue: asyncio.Queue = asyncio.Queue()
        await coordinator._submit_request(
            "req-1", {"text": "hello"}, stream_queue=queue
        )
        bindings = coordinator.control_plane.submitted[0][2].replica_bindings
        instance = replica_topology.resolve("code2wav", bindings["code2wav"])

        await coordinator._handle_completion(
            CompleteMessage(
                "req-1",
                instance,
                success,
                result={"audio": "ok"} if success else None,
                error=None if success else "boom",
            )
        )

        assert queue.get_nowait().from_stage == "code2wav"

    asyncio.run(_run())


def _compile_replica_runtime(stages, **replicas: int):
    config = PipelineConfig(
        stages=stages,
        model_path="dummy",
        processes={
            process: ProcessConfig(num_replicas=count)
            for process, count in replicas.items()
            if count > 1
        },
    )
    logical_plan, compiled_stages = compile_logical_processes(config)
    _, replica_topology = expand_replica_stages(compiled_stages, logical_plan)
    return logical_plan, replica_topology


def _linear_replica_runtime(**replicas: int):
    return _compile_replica_runtime(
        [
            stage("normalize", process="front", next="decode"),
            stage("decode", process="tail", next="postprocess"),
            stage("postprocess", process="tail", terminal=True),
        ],
        **replicas,
    )


def _multi_terminal_replica_runtime():
    return _compile_replica_runtime(
        [
            stage(
                "preprocess",
                process="front",
                next=["decode", "code2wav"],
            ),
            stage("decode", process="text", terminal=True),
            stage("code2wav", process="audio", terminal=True),
        ],
        audio=2,
    )


def test_coordinator_projects_one_process_choice_onto_member_stages() -> None:
    async def _run() -> None:
        logical_plan, replica_topology = _linear_replica_runtime(tail=2)
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="normalize",
            replica_topology=replica_topology,
            logical_process_plan=logical_plan,
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("normalize", "inproc://normalize")

        await coordinator._submit_request("req-0", "hello")
        await coordinator._submit_request("req-1", "hello")

        bindings = [
            msg.replica_bindings for _stage, _ep, msg in control_plane.submitted
        ]
        assert bindings == [
            {"decode": 0, "postprocess": 0},
            {"decode": 1, "postprocess": 1},
        ]
        assert [stage for stage, _ep, _msg in control_plane.submitted] == [
            "normalize",
            "normalize",
        ]

    asyncio.run(_run())


def test_binding_validation_precedes_request_registration() -> None:
    class FailOnceBindingPolicy:
        def __init__(self) -> None:
            self.calls = 0

        def bind(self, process_name, num_replicas, request_id):
            del process_name, request_id
            self.calls += 1
            return num_replicas if self.calls == 1 else 0

    async def _run() -> None:
        logical_plan, replica_topology = _linear_replica_runtime(tail=2)
        policy = FailOnceBindingPolicy()
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="normalize",
            replica_topology=replica_topology,
            logical_process_plan=logical_plan,
            binding_policy=policy,
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("normalize", "inproc://normalize")

        with pytest.raises(ValueError, match="selected replica 2"):
            await coordinator._submit_request("req-retry", "hello")

        assert coordinator._requests == {}
        assert coordinator._completion_futures == {}

        await coordinator._submit_request("req-retry", "hello")
        assert control_plane.submitted[0][2].replica_bindings == {
            "decode": 0,
            "postprocess": 0,
        }

    asyncio.run(_run())


def test_coordinator_submits_to_the_bound_entry_replica() -> None:
    async def _run() -> None:
        logical_plan, replica_topology = _linear_replica_runtime(front=2)
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="normalize",
            replica_topology=replica_topology,
            logical_process_plan=logical_plan,
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("normalize@r0", "inproc://n0")
        coordinator.register_stage("normalize@r1", "inproc://n1")

        await coordinator._submit_request("req-0", "hello")
        await coordinator._submit_request("req-1", "hello")

        assert [stage for stage, _ep, _msg in control_plane.submitted] == [
            "normalize@r0",
            "normalize@r1",
        ]
        assert [endpoint for _stage, endpoint, _msg in control_plane.submitted] == [
            "inproc://n0",
            "inproc://n1",
        ]

    asyncio.run(_run())


def test_coordinator_without_replicas_sends_no_bindings() -> None:
    async def _run() -> None:
        logical_plan, replica_topology = _linear_replica_runtime()
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="normalize",
            replica_topology=replica_topology,
            logical_process_plan=logical_plan,
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("normalize", "inproc://normalize")

        await coordinator._submit_request("req-0", "hello")

        assert control_plane.submitted[0][2].replica_bindings is None

    asyncio.run(_run())
