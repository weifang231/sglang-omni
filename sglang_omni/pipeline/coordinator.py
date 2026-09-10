# SPDX-License-Identifier: Apache-2.0
"""Coordinator for managing the multi-stage pipeline."""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from sglang_omni.admission import QueueFullError
from sglang_omni.config.topology import LogicalProcessPlan
from sglang_omni.pipeline.control_plane import CoordinatorControlPlane
from sglang_omni.pipeline.replicas import (
    BindingPolicy,
    ReplicaTopology,
    RoundRobinBindingPolicy,
    assign_replica_bindings,
)
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.utils.runtime_policy import PolicyAdmissionRejected, create_runtime_policy
from sglang_omni.proto import (
    AbortMessage,
    AdminMessage,
    AdminOperation,
    AdminResult,
    AdminResultMessage,
    CompleteMessage,
    OmniRequest,
    RequestInfo,
    RequestState,
    StageInfo,
    StagePayload,
    StreamMessage,
    SubmitMessage,
    is_update_action,
)
from sglang_omni.runtime_queue import (
    RuntimeCreditQueue,
    RuntimeQueueAdmissionRejection,
    RuntimeQueueItem,
)
from sglang_omni.utils.runtime_state import RuntimeStateChannel

logger = logging.getLogger(__name__)


@dataclass
class _AdminPendingOperation:
    expected_stages: set[str]
    action: str
    results: dict[str, AdminResult] = field(default_factory=dict)
    future: asyncio.Future | None = None


@dataclass(frozen=True)
class _PendingRequestDispatch:
    entry_instance: str
    entry_endpoint: str
    message: SubmitMessage


class Coordinator:
    """Central coordinator for the multi-stage pipeline.

    Responsibilities:
    - Register stages
    - Submit requests to entry stage
    - Track request state
    - Handle completions
    - Broadcast abort signals
    """

    def __init__(
        self,
        completion_endpoint: str,
        abort_endpoint: str,
        entry_stage: str,
        terminal_stages: list[str] | None = None,
        terminal_stages_resolver: (
            Callable[[OmniRequest], list[str] | None] | None
        ) = None,
        replica_topology: ReplicaTopology | None = None,
        logical_process_plan: LogicalProcessPlan | None = None,
        binding_policy: BindingPolicy | None = None,
        max_in_flight: int | None = None,
        queue_control: Any | None = None,
    ):
        """Initialize coordinator.

        Args:
            completion_endpoint: ZMQ endpoint to receive completions
            abort_endpoint: ZMQ endpoint for abort broadcasts
            entry_stage: Logical name of the entry stage for new requests
            terminal_stages: Terminal stage names. When multiple are given,
                the coordinator waits for all to complete before resolving.
            replica_topology: Logical stage to expanded instance mapping.
            logical_process_plan: Compiled Process topology; the coordinator
                selects one replica per replicated Process from it.
            max_in_flight: Native generation capacity
                (max_running_requests + max_queued_requests). Without runtime
                queue control, reject new submits once this many requests are
                tracked. With runtime queue control, bound active dispatch and
                provide the default waiting-queue limit.
            queue_control: Optional runtime-owned FIFO/EDF queue with global
                and per-class WIP credits. Credits are held from entry-stage
                dispatch until the full request reaches a terminal state.
        """
        self.entry_stage = entry_stage
        self._terminal_stages: set[str] = (
            set(terminal_stages) if terminal_stages else set()
        )
        self._terminal_stages_resolver = terminal_stages_resolver
        self._partial_results: dict[str, dict[str, Any]] = {}
        self._replica_topology = replica_topology or ReplicaTopology()
        self._logical_process_plan = logical_process_plan or LogicalProcessPlan(
            processes=(), stage_to_process={}
        )
        self._binding_policy = binding_policy or RoundRobinBindingPolicy()
        if max_in_flight is None:
            self.max_in_flight = None
        else:
            value = int(max_in_flight)
            if value < 0:
                raise ValueError("max_in_flight must be >= 0")
            self.max_in_flight = value
        self._runtime_queue: RuntimeCreditQueue[_PendingRequestDispatch] | None = None
        if queue_control is not None:
            values = (
                queue_control.model_dump()
                if hasattr(queue_control, "model_dump")
                else dict(queue_control)
            )
            # The native generation limit protects dispatched work. Runtime
            # waiters live before that boundary and therefore need a separate
            # bound instead of consuming max_in_flight slots.
            if values.get("max_waiting_requests") is None:
                values["max_waiting_requests"] = self.max_in_flight
            configured_active = values.get("max_active_requests")
            if self.max_in_flight is not None and (
                configured_active is None or configured_active > self.max_in_flight
            ):
                if configured_active is not None:
                    logger.warning(
                        "Clamping queue_control.max_active_requests from %s to "
                        "native max_in_flight=%s",
                        configured_active,
                        self.max_in_flight,
                    )
                values["max_active_requests"] = self.max_in_flight
            self._runtime_queue = RuntimeCreditQueue.from_config(values)
        self._runtime_state_channel = RuntimeStateChannel(
            engine="sglang-omni",
            component="coordinator_queue",
            stage_id="pipeline",
        )
        self._runtime_dispatch_lock = asyncio.Lock()
        self._runtime_control_task: asyncio.Task[None] | None = None

        # Control plane
        self.control_plane = CoordinatorControlPlane(
            completion_endpoint=completion_endpoint,
            abort_endpoint=abort_endpoint,
        )

        # Stage registry
        self._stages: dict[str, StageInfo] = {}

        # Request tracking
        self._requests: dict[str, RequestInfo] = {}
        self._completion_futures: dict[str, asyncio.Future] = {}
        self._stream_queues: dict[
            str, asyncio.Queue[CompleteMessage | StreamMessage]
        ] = {}
        # Abort messages carry only the request ID. A strongly held task keeps
        # local admission closed and lets the broadcast survive caller cancellation.
        self._abort_tasks: dict[str, asyncio.Task[bool]] = {}
        self._admin_ops: dict[str, _AdminPendingOperation] = {}
        self._admin_lock = asyncio.Lock()

        # State
        self._running = False
        self._fatal_error: str | None = None
        self._runtime_policy = create_runtime_policy(
            role="coordinator", entry_stage=entry_stage,
            terminal_stages=terminal_stages, queue_control=queue_control,
        )

    def register_stage(self, name: str, endpoint: str) -> None:
        """Register a stage.

        Args:
            name: Stage name
            endpoint: ZMQ endpoint for the stage
        """
        if self._runtime_policy is not None:
            self._runtime_policy.register_stage(name, self._stages)
        self._stages[name] = StageInfo(name=name, control_endpoint=endpoint)
        logger.info("Coordinator registered stage: %s at %s", name, endpoint)

    async def start(self) -> None:
        """Start the coordinator."""
        await self.control_plane.start()
        self._running = True
        if (
            self._runtime_queue is not None
            and self._runtime_state_channel.enabled
            and self._runtime_control_task is None
        ):
            self._runtime_control_task = asyncio.create_task(
                self._runtime_control_loop(),
                name="coordinator-runtime-control-loop",
            )
        logger.info("Coordinator started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        self._running = False
        if self._runtime_control_task is not None:
            self._runtime_control_task.cancel()
            try:
                await self._runtime_control_task
            except asyncio.CancelledError:
                pass
            self._runtime_control_task = None
        self.control_plane.close()
        if self._runtime_policy is not None:
            await asyncio.to_thread(self._runtime_policy.close)
        logger.info("Coordinator stopped")

    async def fail_pending_requests(self, error: BaseException | str) -> None:
        """Fail all requests currently owned by the coordinator."""
        self._running = False
        message = str(error)
        self._fatal_error = message
        for request_id, info in list(self._requests.items()):
            info.state = RequestState.FAILED
            info.error = message
            self._reject_completion_future(request_id, RuntimeError(message))
            queue = self._stream_queues.get(request_id)
            if queue is not None:
                await queue.put(
                    CompleteMessage(
                        request_id=request_id,
                        from_stage="coordinator",
                        success=False,
                        error=message,
                    )
                )
        self._requests.clear()
        self._partial_results.clear()
        if self._runtime_queue is not None:
            self._runtime_queue.clear()

    async def shutdown_stages(self) -> None:
        """Send shutdown signal to all registered stages."""
        for name, info in self._stages.items():
            try:
                await self.control_plane.send_shutdown(name, info.control_endpoint)
                logger.info("Sent shutdown to stage: %s", name)
            except Exception as e:
                logger.warning("Failed to send shutdown to stage %s: %s", name, e)

    async def admin(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Run an administrative operation against one or more stages."""
        if not self._running:
            raise RuntimeError("Coordinator is not running")

        target_stages = self._resolve_admin_stages(stages)
        if not target_stages:
            raise ValueError("No stages registered for admin operation")

        op_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        pending = _AdminPendingOperation(
            expected_stages=set(target_stages),
            action=action,
            future=loop.create_future(),
        )
        operation = AdminOperation(
            op_id=op_id,
            action=action,
            payload=dict(payload or {}),
            target_stages=list(target_stages),
            timeout_s=timeout_s,
        )

        async with self._admin_lock:
            self._admin_ops[op_id] = pending
            try:
                for stage_name in target_stages:
                    info = self._stages[stage_name]
                    await self.control_plane.send_admin(
                        stage_name,
                        info.control_endpoint,
                        AdminMessage(operation=operation),
                    )

                assert pending.future is not None
                results = await asyncio.wait_for(pending.future, timeout=timeout_s)
            finally:
                self._admin_ops.pop(op_id, None)

        return self._aggregate_admin_results(
            op_id=op_id,
            action=action,
            results=list(results.values()),
        )

    async def model_info(
        self,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "model_info",
            stages=stages,
            timeout_s=timeout_s,
        )

    async def pause_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "pause_generation",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def continue_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "continue_generation",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def update_weights_from_disk(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "update_weights_from_disk",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def init_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "init_weights_update_group",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def destroy_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "destroy_weights_update_group",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def update_weights_from_distributed(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "update_weights_from_distributed",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def weights_checker(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "weights_checker",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def submit(self, request_id: str, request: OmniRequest | Any) -> Any:
        """Submit a request to the pipeline and wait for completion."""
        try:
            await self._submit_request(request_id, request)
            future = self._completion_futures[request_id]
            return await future
        except asyncio.CancelledError:
            if request_id in self._requests:
                abort_task = asyncio.create_task(self.abort(request_id))
                try:
                    await asyncio.shield(abort_task)
                except asyncio.CancelledError:
                    await abort_task
                except Exception:
                    logger.exception(
                        "Coordinator failed to abort cancelled submit req=%s",
                        request_id,
                    )
            raise
        finally:
            self._completion_futures.pop(request_id, None)

    async def stream(
        self, request_id: str, request: OmniRequest | Any
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        """Submit a request and yield stream events until completion."""
        queue: asyncio.Queue[CompleteMessage | StreamMessage] = asyncio.Queue()

        try:
            await self._submit_request(request_id, request, stream_queue=queue)
            expected_terminal_stages = self._expected_terminal_stages(request_id)

            completed_stages: set[str] = set()
            while True:
                msg = await queue.get()
                if isinstance(msg, CompleteMessage):
                    if not msg.success:
                        raise QueueFullError.from_message(msg.error)
                    yield msg
                    completed_stages.add(
                        self._replica_topology.logical_name(msg.from_stage)
                    )
                    if (
                        not expected_terminal_stages
                        or completed_stages >= expected_terminal_stages
                    ):
                        return
                else:
                    yield msg
        finally:
            if self._stream_queues.get(request_id) is queue:
                try:
                    if request_id in self._requests:
                        try:
                            await self.abort(request_id)
                        except Exception:
                            # The coordinator-owned abort task logs its own failure.
                            # Do not replace the exception already leaving the stream.
                            pass
                finally:
                    if self._stream_queues.get(request_id) is queue:
                        self._stream_queues.pop(request_id, None)
                        self._completion_futures.pop(request_id, None)

    async def _submit_request(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        stream_queue: asyncio.Queue[CompleteMessage | StreamMessage] | None = None,
    ) -> None:
        """Submit a request without waiting for completion."""
        if self._fatal_error is not None:
            raise RuntimeError(self._fatal_error)
        if self._request_id_is_reserved(request_id):
            raise ValueError(f"Request {request_id} already exists")

        if (
            self._runtime_queue is None
            and self.max_in_flight is not None
            and len(self._requests) >= self.max_in_flight
        ):
            logger.warning(
                "Rejecting request %s before pipeline submit: in-flight cap "
                "(max_in_flight=%s)",
                request_id,
                self.max_in_flight,
            )
            raise QueueFullError()

        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)

        replica_bindings = assign_replica_bindings(
            self._logical_process_plan, self._binding_policy, request_id
        )
        bindings = replica_bindings or {}
        entry_instance = (
            self._replica_topology.resolve(self.entry_stage, bindings[self.entry_stage])
            if self._replica_topology.is_replicated(self.entry_stage)
            else self.entry_stage
        )
        if entry_instance not in self._stages:
            raise ValueError(f"Entry stage {entry_instance} not registered")
        entry_info = self._stages[entry_instance]

        if self._runtime_policy is not None:
            self._runtime_policy.validate_stages(self._stages)
            if not self._runtime_policy.admit(request_id, request):
                decision = self._runtime_policy.receipts[request_id]["decision"]
                raise PolicyAdmissionRejected(
                    f"D1 {decision['status']}: {decision['reason']}",
                    self._runtime_policy.message_metadata(request_id),
                )

        # Track request
        self._requests[request_id] = RequestInfo(
            request_id=request_id,
            state=RequestState.PENDING,
            current_stage=self.entry_stage,
            terminal_stages=self._resolve_terminal_stages(request),
        )

        # Create future for completion
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._completion_futures[request_id] = future
        if stream_queue is not None:
            self._stream_queues[request_id] = stream_queue

        payload = StagePayload(
            request_id=request_id,
            request=request,
            data={"raw_inputs": request.inputs},
        )

        pending = _PendingRequestDispatch(
            entry_instance=entry_instance,
            entry_endpoint=entry_info.control_endpoint,
            message=SubmitMessage(
                request_id=request_id,
                data=payload,
                replica_bindings=replica_bindings,
            ),
        )

        request_class = None
        first_output_deadline_unix_s = None
        if self._runtime_queue is not None:
            try:
                request_class, first_output_deadline_unix_s = (
                    self._runtime_queue.attributes(request.metadata)
                )
            except Exception:
                self._requests.pop(request_id, None)
                self._completion_futures.pop(request_id, None)
                self._stream_queues.pop(request_id, None)
                raise
            info = self._requests.get(request_id)
            if info is not None:
                info.request_class = request_class
                info.first_output_deadline_unix_s = first_output_deadline_unix_s

        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="request_admission",
            metadata={
                "entry_stage": self.entry_stage,
                "request_class": request_class,
                "first_output_deadline_unix_s": first_output_deadline_unix_s,
            },
        )

        if self._runtime_queue is None:
            await self._dispatch_submission(pending)
            return

        try:
            dispatched = self._runtime_queue.enqueue(
                request_id,
                pending,
                request.metadata,
            )
        except QueueFullError:
            self._requests.pop(request_id, None)
            self._completion_futures.pop(request_id, None)
            self._stream_queues.pop(request_id, None)
            snapshot = self._runtime_queue.snapshot()
            _emit_event(
                request_id=request_id,
                stage="coordinator",
                event_name="runtime_queue_rejected",
                metadata={
                    "scope": "pipeline",
                    "request_class": request_class,
                    "reason": self._runtime_queue.last_rejection_reason
                    or "waiting_queue_full",
                    "active_requests": snapshot["active_requests"],
                    "waiting_requests": snapshot["waiting_requests"],
                    "max_waiting_requests": snapshot["max_waiting_requests"],
                },
            )
            self._publish_runtime_queue_snapshot(force=True)
            raise
        snapshot = self._runtime_queue.snapshot()
        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="runtime_queue_enter",
            metadata={
                "scope": "pipeline",
                "request_class": request_class,
                "first_output_deadline_unix_s": first_output_deadline_unix_s,
                "active_requests": snapshot["active_requests"],
                "waiting_requests": snapshot["waiting_requests"],
            },
        )
        await self._dispatch_runtime_items(dispatched)
        self._publish_runtime_queue_snapshot()

    async def _dispatch_submission(self, pending: _PendingRequestDispatch) -> None:
        await self.control_plane.submit_to_stage(
            pending.entry_instance,
            pending.entry_endpoint,
            pending.message,
        )
        request_id = pending.message.request_id
        info = self._requests.get(request_id)
        if info is not None:
            info.state = RequestState.RUNNING
        logger.info(
            "Coordinator submitted req=%s to %s at %s bindings=%s",
            request_id,
            pending.entry_instance,
            pending.entry_endpoint,
            pending.message.replica_bindings,
        )

    async def _dispatch_runtime_items(
        self,
        items: Sequence[RuntimeQueueItem[_PendingRequestDispatch]],
    ) -> None:
        if not items:
            return
        # Queue transitions grant credits synchronously, before network sends.
        # Serialize and shield the full granted batch so a concurrent release or
        # caller cancellation cannot reorder or orphan already-active items.
        # A control-plane caller (for example, a live policy update) does not own
        # the user requests it happens to dispatch, so its cancellation must not
        # cancel a stage send or abort an unrelated request.
        dispatch_task = asyncio.create_task(
            self._dispatch_runtime_items_locked(items),
            name="coordinator-runtime-queue-dispatch",
        )
        try:
            await asyncio.shield(dispatch_task)
        except asyncio.CancelledError:
            await dispatch_task
            raise

    async def _dispatch_runtime_items_locked(
        self,
        items: Sequence[RuntimeQueueItem[_PendingRequestDispatch]],
    ) -> None:
        async with self._runtime_dispatch_lock:
            await self._dispatch_runtime_items_unlocked(items)

    async def _dispatch_runtime_items_unlocked(
        self,
        items: Sequence[RuntimeQueueItem[_PendingRequestDispatch]],
    ) -> None:
        pending_items = list(items)
        cancellation: asyncio.CancelledError | None = None
        while pending_items:
            item = pending_items.pop(0)
            request_id = item.request_id
            if request_id not in self._requests:
                released = self._runtime_queue.release(request_id)
                if released is not None:
                    pending_items.extend(released[1])
                continue
            snapshot = self._runtime_queue.snapshot()
            _emit_event(
                request_id=request_id,
                stage="coordinator",
                event_name="runtime_credit_acquired",
                metadata={
                    "scope": "pipeline",
                    "request_class": item.request_class,
                    "first_output_deadline_unix_s": item.deadline_unix_s,
                    "queue_wait_ms": (time.monotonic_ns() - item.enqueued_ns) / 1e6,
                    "active_requests": snapshot["active_requests"],
                    "waiting_requests": snapshot["waiting_requests"],
                },
            )
            try:
                await self._dispatch_submission(item.value)
            except asyncio.CancelledError as exc:
                # A cancelled send is ambiguous: the stage may have received
                # the request before cancellation reached this task. Abort it
                # before releasing the credit, then continue draining every
                # item already granted by the same queue transition.
                abort_confirmed = True
                try:
                    await self.control_plane.broadcast_abort(
                        AbortMessage(request_id=request_id)
                    )
                except Exception:
                    abort_confirmed = False
                    logger.exception(
                        "Coordinator could not confirm rollback for cancelled "
                        "dispatch req=%s; retaining its credit",
                        request_id,
                    )
                if abort_confirmed:
                    future = self._completion_futures.pop(request_id, None)
                    if future is not None and not future.done():
                        future.cancel()
                    self._stream_queues.pop(request_id, None)
                    self._requests.pop(request_id, None)
                    self._partial_results.pop(request_id, None)
                    released = self._runtime_queue.release(request_id)
                    if released is not None:
                        self._emit_runtime_credit_released(
                            released[0], status="dispatch_cancelled"
                        )
                        pending_items.extend(released[1])
                cancellation = exc
            except Exception as exc:
                logger.exception(
                    "Coordinator failed to dispatch queued req=%s", request_id
                )
                info = self._requests.get(request_id)
                if info is not None:
                    info.state = RequestState.FAILED
                    info.error = str(exc)
                self._reject_completion_future(request_id, exc)
                stream_queue = self._stream_queues.get(request_id)
                if stream_queue is not None:
                    await stream_queue.put(
                        CompleteMessage(
                            request_id=request_id,
                            from_stage="coordinator",
                            success=False,
                            error=str(exc),
                        )
                    )
                self._requests.pop(request_id, None)
                self._partial_results.pop(request_id, None)
                released = self._runtime_queue.release(request_id)
                if released is not None:
                    self._emit_runtime_credit_released(
                        released[0], status="dispatch_error"
                    )
                    pending_items.extend(released[1])
        self._publish_runtime_queue_snapshot()
        if cancellation is not None:
            raise cancellation

    def _emit_runtime_credit_released(
        self,
        item: RuntimeQueueItem[_PendingRequestDispatch],
        *,
        status: str,
    ) -> None:
        if self._runtime_queue is None:
            return
        snapshot = self._runtime_queue.snapshot()
        _emit_event(
            request_id=item.request_id,
            stage="coordinator",
            event_name="runtime_credit_released",
            metadata={
                "scope": "pipeline",
                "request_class": item.request_class,
                "status": status,
                "active_requests": snapshot["active_requests"],
                "waiting_requests": snapshot["waiting_requests"],
            },
        )

    async def _release_runtime_credit(self, request_id: str, *, status: str) -> None:
        if self._runtime_queue is None:
            return
        released = self._runtime_queue.release(request_id)
        if released is None:
            return
        item, dispatched = released
        self._emit_runtime_credit_released(item, status=status)
        await self._dispatch_runtime_items(dispatched)
        self._publish_runtime_queue_snapshot()

    async def update_queue_control(self, queue_control: Any) -> dict[str, Any]:
        """Replace live queue limits without preempting active requests."""
        if self._runtime_queue is None:
            raise RuntimeError("pipeline queue_control was not configured")
        values = (
            queue_control.model_dump()
            if hasattr(queue_control, "model_dump")
            else dict(queue_control)
        )
        values.setdefault(
            "max_active_requests", self._runtime_queue.max_active_requests
        )
        values.setdefault("class_limits", self._runtime_queue.class_limits)
        values.setdefault("discipline", self._runtime_queue.discipline)
        values.setdefault("class_metadata_key", self._runtime_queue.class_metadata_key)
        values.setdefault(
            "deadline_metadata_key", self._runtime_queue.deadline_metadata_key
        )
        if values.get("max_waiting_requests") is None:
            values["max_waiting_requests"] = self._runtime_queue.max_waiting_requests
        # Trust is a startup-time boundary, not a live scheduling knob.
        values["trust_request_metadata"] = self._runtime_queue.trust_request_metadata
        values.setdefault("class_limit_mode", self._runtime_queue.class_limit_mode)
        values.setdefault("admission", self._runtime_queue.admission)
        values.setdefault("online_allocator", self._runtime_queue.online_allocator)
        configured_active = values.get("max_active_requests")
        if self.max_in_flight is not None and (
            configured_active is None or configured_active > self.max_in_flight
        ):
            if configured_active is not None:
                logger.warning(
                    "Clamping live queue-control max_active_requests from %s to "
                    "native max_in_flight=%s",
                    configured_active,
                    self.max_in_flight,
                )
            values["max_active_requests"] = self.max_in_flight
        replacement = RuntimeCreditQueue.from_config(values)
        if (
            replacement.class_metadata_key != self._runtime_queue.class_metadata_key
            or replacement.deadline_metadata_key
            != self._runtime_queue.deadline_metadata_key
            or replacement.trust_request_metadata
            != self._runtime_queue.trust_request_metadata
        ):
            raise ValueError(
                "queue-control metadata trust and keys cannot change at runtime"
            )
        dispatched = self._runtime_queue.update(
            max_active_requests=replacement.max_active_requests,
            max_waiting_requests=replacement.max_waiting_requests,
            class_limits=replacement.class_limits,
            discipline=replacement.discipline,
            class_limit_mode=replacement.class_limit_mode,
            admission=replacement.admission,
            online_allocator=replacement.online_allocator,
        )
        rejections = self._runtime_queue.recheck_admission()
        await self._reject_runtime_queue_rechecks(rejections)
        await self._dispatch_runtime_items(dispatched)
        self._publish_runtime_queue_snapshot(force=True)
        return self._runtime_queue.snapshot()

    def _publish_runtime_queue_snapshot(self, *, force: bool = False) -> None:
        if self._runtime_queue is None:
            return
        self._runtime_state_channel.maybe_publish(
            self._runtime_queue.snapshot,
            force=force,
        )

    async def _runtime_control_loop(self) -> None:
        while self._running:
            start_ns = self._runtime_state_channel.timing_start_ns()
            try:
                await self._apply_runtime_control_file_once()
                self._publish_runtime_queue_snapshot()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Coordinator ignored invalid runtime control update",
                    exc_info=True,
                )
            finally:
                self._runtime_state_channel.record_timing_elapsed_ns(
                    "coordinator_control_loop_ns",
                    start_ns,
                )
            await asyncio.sleep(0.05)

    async def _apply_runtime_control_file_once(self) -> None:
        if self._runtime_queue is None:
            return
        control = self._runtime_state_channel.read_control_if_changed()
        if control is None:
            return
        payload = self._queue_control_payload_from_runtime_control(control)
        if payload is None:
            return
        await self.update_queue_control(payload)

    @staticmethod
    def _queue_control_payload_from_runtime_control(
        control: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if "pipeline_queue_control" in control:
            payload = control["pipeline_queue_control"]
        elif (
            "queue_control" in control
            and control.get("scope", "pipeline") == "pipeline"
        ):
            payload = control["queue_control"]
        elif "generation_cap" in control or "pp_max_micro_batch_size" in control:
            payload = {
                "max_active_requests": control.get(
                    "generation_cap",
                    control.get("pp_max_micro_batch_size"),
                )
            }
        elif any(
            key in control
            for key in (
                "discipline",
                "max_active_requests",
                "max_waiting_requests",
                "class_limits",
                "class_limit_mode",
                "admission",
                "online_allocator",
            )
        ):
            payload = control
        else:
            return None
        if payload is False or payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise ValueError("runtime queue-control payload must be an object")
        return dict(payload)

    async def _reject_runtime_queue_rechecks(
        self,
        rejections: Sequence[RuntimeQueueAdmissionRejection[_PendingRequestDispatch]],
    ) -> None:
        if self._runtime_queue is None:
            return
        for rejection in rejections:
            request_id = rejection.item.request_id
            info = self._requests.get(request_id)
            if info is not None:
                info.state = RequestState.FAILED
                info.error = rejection.decision.reason
            self._reject_completion_future(request_id, QueueFullError())
            stream_queue = self._stream_queues.get(request_id)
            if stream_queue is not None:
                await stream_queue.put(
                    CompleteMessage(
                        request_id=request_id,
                        from_stage="coordinator",
                        success=False,
                        error=QueueFullError.MESSAGE,
                    )
                )
            self._requests.pop(request_id, None)
            self._completion_futures.pop(request_id, None)
            self._stream_queues.pop(request_id, None)
            self._partial_results.pop(request_id, None)
            snapshot = self._runtime_queue.snapshot()
            _emit_event(
                request_id=request_id,
                stage="coordinator",
                event_name="runtime_queue_rejected",
                metadata={
                    "scope": "pipeline",
                    "request_class": rejection.item.request_class,
                    "reason": rejection.decision.reason,
                    "phase": rejection.decision.phase,
                    "active_requests": snapshot["active_requests"],
                    "waiting_requests": snapshot["waiting_requests"],
                },
            )

    def _request_id_is_reserved(self, request_id: str) -> bool:
        """Return whether any coordinator owner still holds this request ID."""
        return (
            request_id in self._requests
            or request_id in self._completion_futures
            or request_id in self._stream_queues
            or request_id in self._abort_tasks
        )

    def _reject_completion_future(
        self,
        request_id: str,
        exc: BaseException,
    ) -> None:
        # Note: (Akazaakane) Non-streaming callers await the completion future,
        # so errors must be propagated with set_exception(). Streaming callers
        # receive errors through the stream queue and never await that future;
        # cancel it instead to avoid "Future exception was never retrieved".
        future = self._completion_futures.get(request_id)
        if future is None or future.done():
            return
        if request_id in self._stream_queues:
            future.cancel()
        else:
            future.set_exception(exc)

    async def abort(self, request_id: str) -> bool:
        """Abort a request.

        Args:
            request_id: Request to abort

        Returns:
            True if aborted, False if not found
        """
        abort_task = self._abort_tasks.get(request_id)
        if abort_task is not None:
            return await asyncio.shield(abort_task)

        info = self._requests.get(request_id)
        if info is None:
            return False

        if info.state in (
            RequestState.COMPLETED,
            RequestState.FAILED,
            RequestState.ABORTED,
        ):
            return False

        abort_task = asyncio.create_task(
            self._run_abort(request_id),
            name=f"coordinator-abort-{request_id}",
        )
        self._abort_tasks[request_id] = abort_task
        abort_task.add_done_callback(
            lambda done, rid=request_id: self._on_abort_task_done(rid, done)
        )
        return await asyncio.shield(abort_task)

    async def _run_abort(
        self,
        request_id: str,
    ) -> bool:
        if self._runtime_policy is not None:
            self._runtime_policy.aborted(request_id)
        cancellation = None
        queued_only = (
            self._runtime_queue is not None
            and self._runtime_queue.is_waiting(request_id)
        )
        if queued_only:
            cancellation = self._runtime_queue.cancel(request_id)
        else:
            # A successful PUB send proves transport acceptance, not that every
            # stage has quiesced. We retain the existing logical-credit behavior
            # here; deployments requiring a strict physical-WIP bound across
            # cancellation need an explicit stage-termination acknowledgement.
            # If publishing itself fails, retain the credit rather than orphaning
            # the slot and dispatching a successor immediately.
            await self.control_plane.broadcast_abort(
                AbortMessage(request_id=request_id)
            )
            if self._runtime_queue is not None:
                cancellation = self._runtime_queue.cancel(request_id)

        info = self._requests.get(request_id)
        if info is None:
            return False

        info.state = RequestState.ABORTED
        self._reject_completion_future(
            request_id, asyncio.CancelledError(f"Request {request_id} aborted")
        )
        stream_queue = self._stream_queues.get(request_id)
        if stream_queue is not None:
            await stream_queue.put(
                CompleteMessage(
                    request_id=request_id,
                    from_stage="coordinator",
                    success=False,
                    error="aborted",
                )
            )

        self._requests.pop(request_id, None)
        self._partial_results.pop(request_id, None)

        if cancellation is not None:
            if cancellation.item is not None:
                event_name = (
                    "runtime_credit_released"
                    if cancellation.was_active
                    else "runtime_queue_cancelled"
                )
                snapshot = self._runtime_queue.snapshot()
                _emit_event(
                    request_id=request_id,
                    stage="coordinator",
                    event_name=event_name,
                    metadata={
                        "scope": "pipeline",
                        "request_class": cancellation.item.request_class,
                        "status": "aborted",
                        "active_requests": snapshot["active_requests"],
                        "waiting_requests": snapshot["waiting_requests"],
                    },
                )
            await self._dispatch_runtime_items(cancellation.dispatched)
            self._publish_runtime_queue_snapshot()

        logger.info("Coordinator aborted req=%s", request_id)
        return True

    def _on_abort_task_done(
        self,
        request_id: str,
        task: asyncio.Task[bool],
    ) -> None:
        if self._abort_tasks.get(request_id) is task:
            self._abort_tasks.pop(request_id, None)
        if task.cancelled():
            logger.warning("Coordinator abort task cancelled for req=%s", request_id)
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "Failed to abort request %s",
                request_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def run_completion_loop(self) -> None:
        """Run the completion receiving loop.

        This should be run as a background task.
        """
        try:
            while self._running:
                msg = await self.control_plane.recv_event()
                if isinstance(msg, StreamMessage):
                    await self._handle_stream(msg)
                elif isinstance(msg, AdminResultMessage):
                    self._handle_admin_result(msg.result)
                else:
                    await self._handle_completion(msg)
        except asyncio.CancelledError:
            logger.info("Coordinator completion loop cancelled")
        except Exception as e:
            logger.error("Coordinator completion loop error: %s", e)
            raise

    async def _handle_completion(self, msg: CompleteMessage) -> None:
        """Handle a completion message from a stage."""
        request_id = msg.request_id
        if self._runtime_policy is not None:
            self._runtime_policy.completed(request_id, msg.from_stage)
            msg = replace(msg, metadata={**msg.metadata,
                                         **self._runtime_policy.message_metadata(request_id)})
        logger.debug(
            "Coordinator received completion: req=%s from %s success=%s",
            request_id,
            msg.from_stage,
            msg.success,
        )
        event_name = "terminal_response"
        if (self._runtime_policy is not None and msg.success
                and self._replica_topology.logical_name(msg.from_stage)
                not in self._expected_terminal_stages(request_id)):
            event_name = "intermediate_stage_completion"
        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name=event_name,
            metadata={
                "from_stage": msg.from_stage,
                "success": msg.success,
            },
        )

        if request_id not in self._requests:
            logger.debug(
                "Coordinator ignored completion for inactive req=%s from %s",
                request_id,
                msg.from_stage,
            )
            return

        info = self._requests[request_id]

        # Note (wenyao): the client reads ``from_stage`` off the completion.
        # Observability emits above keep the instance name.
        from_stage = self._replica_topology.logical_name(msg.from_stage)
        if from_stage != msg.from_stage:
            msg = replace(msg, from_stage=from_stage)

        # Fail-fast: any terminal failure -> fail entire request
        if not msg.success:
            info.state = RequestState.FAILED
            info.error = msg.error
            await self.control_plane.broadcast_abort(
                AbortMessage(request_id=request_id)
            )
            self._partial_results.pop(request_id, None)
            self._reject_completion_future(
                request_id, QueueFullError.from_message(msg.error)
            )
            stream_queue = self._stream_queues.get(request_id)
            if stream_queue is not None:
                await stream_queue.put(msg)
            self._requests.pop(request_id, None)
            await self._release_runtime_credit(request_id, status="failed")
            return

        expected_terminal_stages = self._expected_terminal_stages(request_id)
        if expected_terminal_stages and from_stage not in expected_terminal_stages:
            logger.debug(
                "Coordinator ignoring completion from inactive terminal: "
                "req=%s stage=%s expected=%s",
                request_id,
                msg.from_stage,
                sorted(expected_terminal_stages),
            )
            return

        # Single active terminal (original behavior) or no terminal_stages configured
        if len(expected_terminal_stages) <= 1:
            info.state = RequestState.COMPLETED
            info.result = msg.result
            if request_id in self._completion_futures:
                future = self._completion_futures[request_id]
                if not future.done():
                    future.set_result(msg.result)
            if request_id in self._stream_queues:
                await self._stream_queues[request_id].put(msg)
            self._requests.pop(request_id, None)
            await self._release_runtime_credit(request_id, status="completed")
            return

        # Multi-terminal: collect partial results
        partials = self._partial_results.setdefault(request_id, {})
        partials[from_stage] = msg.result

        # Forward stream completion per-stage
        if request_id in self._stream_queues:
            await self._stream_queues[request_id].put(msg)

        if set(partials) < expected_terminal_stages:
            return  # still waiting

        # All terminal stages done -> merge and resolve
        merged = dict(partials)
        self._partial_results.pop(request_id)
        info.state = RequestState.COMPLETED
        info.result = merged

        if request_id in self._completion_futures:
            future = self._completion_futures[request_id]
            if not future.done():
                future.set_result(merged)
        self._requests.pop(request_id, None)
        await self._release_runtime_credit(request_id, status="completed")

    async def _handle_stream(self, msg: StreamMessage) -> None:
        """Handle a stream chunk from a stage."""
        request_id = msg.request_id
        if self._runtime_policy is not None:
            msg = replace(msg, metadata={**msg.metadata,
                                         **self._runtime_policy.stream_metadata(request_id)})
        if request_id not in self._stream_queues:
            return
        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="coordinator_stream_received",
            metadata={
                "from_stage": msg.from_stage,
                "chunk_id": msg.chunk_id,
                "modality": msg.modality,
            },
        )
        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="stage_stream_chunk_received",
            metadata={
                "from_stage": msg.from_stage,
                "chunk_id": msg.chunk_id,
                "modality": msg.modality,
            },
        )
        # Note (wenyao): normalize both fields -- the client falls back to
        # ``stage_name or from_stage``. Observability emits above keep the
        # instance name.
        logical = self._replica_topology.logical_name(msg.from_stage)
        stage_name = (
            self._replica_topology.logical_name(msg.stage_name)
            if msg.stage_name is not None
            else msg.stage_name
        )
        if logical != msg.from_stage or stage_name != msg.stage_name:
            msg = replace(msg, from_stage=logical, stage_name=stage_name)
        await self._stream_queues[request_id].put(msg)

    def _handle_admin_result(self, result: AdminResult) -> None:
        pending = self._admin_ops.get(result.op_id)
        if pending is None:
            logger.warning(
                "Coordinator received admin result for unknown op=%s stage=%s",
                result.op_id,
                result.stage,
            )
            return
        pending.results[result.stage] = result
        if (
            pending.future is not None
            and pending.results.keys() >= pending.expected_stages
        ):
            if not pending.future.done():
                pending.future.set_result(dict(pending.results))

    def _resolve_admin_stages(self, stages: Sequence[str] | None) -> list[str]:
        if stages is None:
            return sorted(self._stages)
        # Note (wenyao): dedup preserving order so a caller passing both a
        # logical name and one of its instances does not double-send admin ops.
        resolved: list[str] = []
        for name in stages:
            for instance in self._replica_topology.instances(name):
                if instance not in resolved:
                    resolved.append(instance)
        unknown = sorted(set(resolved) - set(self._stages))
        if unknown:
            raise ValueError(f"Unknown admin target stage(s): {unknown}")
        return resolved

    def _aggregate_admin_results(
        self,
        *,
        op_id: str,
        action: str,
        results: list[AdminResult],
    ) -> dict[str, Any]:
        updated_results = [
            item
            for item in results
            if not item.data.get("skipped") and not item.data.get("unsupported")
        ]
        if is_update_action(action):
            success = bool(updated_results) and all(
                item.success for item in updated_results
            )
        else:
            success = all(item.success for item in results)

        errors = [item.error for item in results if item.error]
        if success:
            message = "ok"
        elif errors:
            message = "; ".join(errors)
        else:
            message = "admin operation did not complete successfully"

        return {
            "op_id": op_id,
            "action": action,
            "success": success,
            "message": message,
            "results": [item.to_dict() for item in results],
        }

    def get_request_info(self, request_id: str) -> RequestInfo | None:
        """Get info about a request."""
        return self._requests.get(request_id)

    def _resolve_terminal_stages(self, request: OmniRequest) -> set[str]:
        if self._terminal_stages_resolver is None:
            return set(self._terminal_stages)
        resolved = self._terminal_stages_resolver(request)
        if resolved is None:
            return set(self._terminal_stages)
        if isinstance(resolved, str) or not isinstance(resolved, Sequence):
            raise ValueError(
                "terminal_stages_resolver must return a sequence of terminal "
                "stage names or None"
            )
        if not all(isinstance(stage, str) for stage in resolved):
            raise ValueError(
                "terminal_stages_resolver must return terminal stage names"
            )
        resolved_stages = set(resolved)
        if not resolved_stages:
            raise ValueError("terminal_stages_resolver returned no terminal stages")
        unknown = resolved_stages - self._terminal_stages
        if unknown:
            raise ValueError(
                "terminal_stages_resolver returned stages outside the static "
                f"terminal stages: {sorted(unknown)}. Allowed terminal stages: "
                f"{sorted(self._terminal_stages)}"
            )
        return resolved_stages

    def _expected_terminal_stages(self, request_id: str) -> set[str]:
        info = self._requests.get(request_id)
        if info is None or info.terminal_stages is None:
            return set(self._terminal_stages)
        return info.terminal_stages

    def health(self) -> dict[str, Any]:
        """Return health status."""
        state_counts = {}
        for info in self._requests.values():
            state = info.state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        health = {
            "running": self._running,
            "stages": list(self._stages.keys()),
            "entry_stage": self.entry_stage,
            "total_requests": len(self._requests),
            "pending_completions": len(self._completion_futures),
            "request_states": state_counts,
        }
        if self._runtime_queue is not None:
            health["queue_control"] = self._runtime_queue.snapshot()
        if self._runtime_policy is not None:
            health["runtime_policy"] = self._runtime_policy.snapshot()
        return health
