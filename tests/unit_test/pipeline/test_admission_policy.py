# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import types

import pytest

from sglang_omni.admission import (
    AdmissionRejectedError,
    QueueFullError,
    load_admission_policy,
)
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, OmniRequest
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


class RecordingPolicy:
    """Admits unless the request ID is listed in ``reject``; records every call."""

    def __init__(self, reject: set[str] | None = None) -> None:
        self.reject = reject or set()
        self.admitted: list[str] = []
        self.completed_ids: list[str] = []
        self.aborted_ids: list[str] = []
        self.closed = False

    def admit(self, request_id: str, request: OmniRequest) -> bool:
        assert isinstance(request, OmniRequest)
        if request_id in self.reject:
            return False
        else:
            pass
        self.admitted.append(request_id)
        return True

    def completed(self, request_id: str) -> None:
        self.completed_ids.append(request_id)

    def aborted(self, request_id: str) -> None:
        self.aborted_ids.append(request_id)

    def close(self) -> None:
        self.closed = True


def make_coordinator(
    policy, **kwargs
) -> tuple[Coordinator, RecordingCoordinatorControlPlane]:
    coordinator = Coordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="preprocess",
        admission_policy=policy,
        **kwargs,
    )
    control_plane = RecordingCoordinatorControlPlane()
    coordinator.control_plane = control_plane
    coordinator.register_stage("preprocess", "inproc://preprocess")
    return coordinator, control_plane


def test_policy_rejection_is_429_and_leaves_no_state() -> None:
    async def run() -> None:
        policy = RecordingPolicy(reject={"req-2"})
        coordinator, control_plane = make_coordinator(policy)

        await coordinator.submit_request("req-1", "hello")
        with pytest.raises(AdmissionRejectedError, match="Admission rejected"):
            await coordinator.submit_request("req-2", "hello")

        assert [msg.request_id for _, _, msg in control_plane.submitted] == ["req-1"]
        assert list(coordinator.requests) == ["req-1"]
        assert not coordinator.request_id_is_reserved("req-2")
        assert policy.admitted == ["req-1"]
        assert coordinator.admitted == {"req-1"}

    asyncio.run(run())


def test_policy_rejection_is_not_a_queue_full_error() -> None:
    assert not QueueFullError.matches(AdmissionRejectedError())
    assert AdmissionRejectedError.matches(RuntimeError("Admission rejected. capacity"))
    assert AdmissionRejectedError("capacity").reason == "capacity"
    assert not AdmissionRejectedError.matches(QueueFullError())


def test_admitted_requests_are_released_once_on_every_exit_path() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        coordinator, control_plane = make_coordinator(policy)

        await coordinator.submit_request("done", "hello")
        await coordinator.handle_completion(
            CompleteMessage("done", "preprocess", True, result={"ok": True})
        )
        await coordinator.handle_completion(  # duplicate completion: ignored
            CompleteMessage("done", "preprocess", True, result={"ok": True})
        )

        await coordinator.submit_request("failed", "hello")
        await coordinator.handle_completion(
            CompleteMessage("failed", "preprocess", False, error="boom")
        )

        await coordinator.submit_request("aborted", "hello")
        assert await coordinator.abort("aborted") is True

        await coordinator.submit_request("pending", "hello")
        await coordinator.fail_pending_requests("shutting down")

        assert policy.admitted == ["done", "failed", "aborted", "pending"]
        assert policy.completed_ids == ["done"]
        assert policy.aborted_ids == ["failed", "aborted", "pending"]
        assert coordinator.admitted == set()
        assert len(control_plane.aborts) >= 2

    asyncio.run(run())


def test_multi_terminal_request_is_released_after_the_last_terminal() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        coordinator, _ = make_coordinator(
            policy, terminal_stages=["decode", "code2wav"]
        )
        await coordinator.submit_request("req-1", {"text": "hello"})
        await coordinator.handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert policy.completed_ids == []
        await coordinator.handle_completion(
            CompleteMessage("req-1", "code2wav", True, result={"audio": "ok"})
        )
        assert policy.completed_ids == ["req-1"]

    asyncio.run(run())


def test_submit_failure_releases_the_admission() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        coordinator, control_plane = make_coordinator(policy)

        async def failing_submit(stage, endpoint, msg):
            raise RuntimeError("control plane down")

        control_plane.submit_to_stage = failing_submit
        with pytest.raises(RuntimeError, match="control plane down"):
            await coordinator.submit_request("req-1", "hello")
        assert policy.aborted_ids == ["req-1"]
        assert coordinator.admitted == set()

    asyncio.run(run())


def test_bypass_flag_and_in_flight_cap_take_precedence() -> None:
    async def run() -> None:
        policy = RecordingPolicy(reject={"bypassed"})
        coordinator, _ = make_coordinator(policy, max_in_flight=1)

        await coordinator.submit_request(
            "bypassed", "hello", should_bypass_admission=True
        )
        assert policy.admitted == []
        with pytest.raises(QueueFullError):
            await coordinator.submit_request("capped", "hello")
        assert policy.admitted == []  # the cap rejected before the policy ran

    asyncio.run(run())


def test_stop_releases_and_closes_the_policy() -> None:
    async def run() -> None:
        policy = RecordingPolicy()
        coordinator, _ = make_coordinator(policy)
        await coordinator.submit_request("req-1", "hello")
        await coordinator.stop()
        assert policy.aborted_ids == ["req-1"]
        assert policy.closed is True

    asyncio.run(run())


def test_without_a_policy_nothing_changes() -> None:
    async def run() -> None:
        coordinator, control_plane = make_coordinator(None)
        await coordinator.submit_request("req-1", "hello")
        await coordinator.handle_completion(
            CompleteMessage("req-1", "preprocess", True, result={"ok": True})
        )
        assert [msg.request_id for _, _, msg in control_plane.submitted] == ["req-1"]
        assert coordinator.admitted == set()

    asyncio.run(run())


# --- loader -----------------------------------------------------------------


def make_policy(*, config):
    return RecordingPolicy(reject=set(getattr(config, "reject_ids", ())))


def make_none(*, config):
    return None


def make_incomplete(*, config):
    return object()


NOT_CALLABLE = "not a factory"


def config_with(spec: str | None, **extra) -> types.SimpleNamespace:
    return types.SimpleNamespace(admission_policy=spec, **extra)


def test_loader_builds_the_named_policy_or_keeps_native() -> None:
    here = __name__
    assert load_admission_policy(config_with(None)) is None
    assert load_admission_policy(config_with(f"{here}.make_none")) is None
    policy = load_admission_policy(config_with(f"{here}.make_policy", reject_ids=["x"]))
    assert isinstance(policy, RecordingPolicy) and policy.reject == {"x"}


def test_loader_rejects_misconfiguration_at_startup() -> None:
    here = __name__
    with pytest.raises(TypeError, match="must be callable"):
        load_admission_policy(config_with(f"{here}.NOT_CALLABLE"))
    with pytest.raises(TypeError, match="without callable admit"):
        load_admission_policy(config_with(f"{here}.make_incomplete"))
    with pytest.raises(ImportError):
        load_admission_policy(config_with(f"{here}.does_not_exist"))


class ConcurrencyCap:
    """The documented minimal policy: a static cap expressed through the hook."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.in_flight: set[str] = set()

    def admit(self, request_id: str, request: OmniRequest) -> bool:
        if len(self.in_flight) >= self.limit:
            return False
        else:
            pass
        self.in_flight.add(request_id)
        return True

    def completed(self, request_id: str) -> None:
        self.in_flight.discard(request_id)

    aborted = completed


def test_documented_concurrency_cap_behaves_like_the_in_flight_cap() -> None:
    async def run() -> None:
        coordinator, control_plane = make_coordinator(ConcurrencyCap(limit=1))

        await coordinator.submit_request("req-1", "hello")
        with pytest.raises(AdmissionRejectedError):
            await coordinator.submit_request("req-2", "hello")

        await coordinator.handle_completion(
            CompleteMessage("req-1", "preprocess", True, result={"ok": True})
        )
        await coordinator.submit_request("req-2", "hello")
        assert await coordinator.abort("req-2") is True
        await coordinator.submit_request("req-3", "hello")
        assert [msg.request_id for _, _, msg in control_plane.submitted] == [
            "req-1",
            "req-2",
            "req-3",
        ]

    asyncio.run(run())
