# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.runtime_queue import (
    FIRST_OUTPUT_DEADLINE_METADATA_KEY,
    REQUEST_CLASS_METADATA_KEY,
    RuntimeCreditQueue,
)


def _metadata(request_class: str, deadline: float | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {REQUEST_CLASS_METADATA_KEY: request_class}
    if deadline is not None:
        metadata[FIRST_OUTPUT_DEADLINE_METADATA_KEY] = deadline
    return metadata


def test_fifo_credit_queue_holds_and_releases_requests() -> None:
    queue = RuntimeCreditQueue[str](max_active_requests=1)

    assert [item.request_id for item in queue.enqueue("r1", "one", {})] == ["r1"]
    assert queue.enqueue("r2", "two", {}) == ()
    released = queue.release("r1")

    assert released is not None
    assert released[0].request_id == "r1"
    assert [item.request_id for item in released[1]] == ["r2"]
    assert queue.snapshot()["active_requests"] == 1
    assert queue.snapshot()["waiting_requests"] == 0


def test_edf_orders_waiters_and_skips_classes_without_credit() -> None:
    queue = RuntimeCreditQueue[str](
        max_active_requests=2,
        class_limits={"text": 1, "speech": 1},
        discipline="edf",
        trust_request_metadata=True,
    )

    assert queue.enqueue("t0", "text-active", _metadata("text", 1.0))
    assert queue.enqueue("t1", "text-wait", _metadata("text", 2.0)) == ()
    dispatched = queue.enqueue("s0", "speech-active", _metadata("speech", 9.0))
    assert [item.request_id for item in dispatched] == ["s0"]
    assert queue.enqueue("s-late", "speech-late", _metadata("speech", 8.0)) == ()
    assert queue.enqueue("s-early", "speech-early", _metadata("speech", 3.0)) == ()

    released = queue.release("s0")
    assert released is not None
    assert [item.request_id for item in released[1]] == ["s-early"]

    released = queue.release("t0")
    assert released is not None
    assert [item.request_id for item in released[1]] == ["t1"]


def test_limit_decrease_is_non_preemptive_and_increase_drains() -> None:
    queue = RuntimeCreditQueue[str](max_active_requests=2)
    assert len(queue.enqueue("r1", "one", {})) == 1
    assert len(queue.enqueue("r2", "two", {})) == 1

    assert queue.update(max_active_requests=1, class_limits={}, discipline="fifo") == ()
    assert queue.enqueue("r3", "three", {}) == ()
    assert queue.release("r1")[1] == ()
    released = queue.release("r2")
    assert released is not None
    assert [item.request_id for item in released[1]] == ["r3"]


def test_aborting_active_request_dispatches_waiter() -> None:
    queue = RuntimeCreditQueue[str](max_active_requests=1)
    queue.enqueue("r1", "one", {})
    queue.enqueue("r2", "two", {})

    cancelled = queue.cancel("r1")
    assert cancelled.was_active is True
    assert [item.request_id for item in cancelled.dispatched] == ["r2"]


def test_runtime_queue_snapshot_schema() -> None:
    queue = RuntimeCreditQueue[str](
        max_active_requests=2,
        class_limits={"text": 1},
        discipline="edf",
        trust_request_metadata=True,
    )
    queue.enqueue("active", "one", _metadata("text", 1.0))
    queue.enqueue("waiting", "two", _metadata("text", 2.0))

    assert queue.snapshot() == {
        "discipline": "edf",
        "max_active_requests": 2,
        "max_waiting_requests": None,
        "class_limits": {"text": 1},
        "trust_request_metadata": True,
        "active_requests": 1,
        "waiting_requests": 1,
        "active_by_class": {"text": 1},
        "waiting_by_class": {"text": 1},
        "waiting_rejected_total": 0,
    }


def test_waiting_limit_excludes_active_requests_and_rejects_only_new_waiters() -> None:
    queue = RuntimeCreditQueue[str](
        max_active_requests=1,
        max_waiting_requests=2,
    )

    assert [item.request_id for item in queue.enqueue("active", "one", {})] == [
        "active"
    ]
    assert queue.enqueue("waiting-1", "two", {}) == ()
    assert queue.enqueue("waiting-2", "three", {}) == ()
    with pytest.raises(QueueFullError):
        queue.enqueue("rejected", "four", {})

    snapshot = queue.snapshot()
    assert snapshot["active_requests"] == 1
    assert snapshot["waiting_requests"] == 2
    assert snapshot["waiting_rejected_total"] == 1


def test_full_waiting_queue_allows_an_immediately_dispatchable_class() -> None:
    queue = RuntimeCreditQueue[str](
        max_active_requests=2,
        max_waiting_requests=1,
        class_limits={"text": 1, "speech": 1},
        trust_request_metadata=True,
    )

    assert queue.enqueue("text-active", "one", _metadata("text"))
    assert queue.enqueue("text-waiting", "two", _metadata("text")) == ()
    dispatched = queue.enqueue("speech-active", "three", _metadata("speech"))

    assert [item.request_id for item in dispatched] == ["speech-active"]
    assert queue.snapshot()["waiting_requests"] == 1


def test_lowering_waiting_limit_does_not_evict_accepted_requests() -> None:
    queue = RuntimeCreditQueue[str](
        max_active_requests=1,
        max_waiting_requests=2,
    )
    queue.enqueue("active", "one", {})
    queue.enqueue("waiting-1", "two", {})
    queue.enqueue("waiting-2", "three", {})

    assert (
        queue.update(
            max_active_requests=1,
            max_waiting_requests=1,
            class_limits={},
            discipline="fifo",
        )
        == ()
    )
    assert queue.snapshot()["waiting_requests"] == 2
    with pytest.raises(QueueFullError):
        queue.enqueue("rejected", "four", {})


def test_class_limit_keys_are_normalized_like_request_metadata() -> None:
    queue = RuntimeCreditQueue[str](
        class_limits={" gold ": 1}, trust_request_metadata=True
    )

    assert queue.class_limits == {"gold": 1}
    assert queue.enqueue("r1", "one", _metadata(" gold "))
    assert queue.enqueue("r2", "two", _metadata("gold")) == ()


def test_boolean_request_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        RuntimeCreditQueue[str](max_active_requests=False)
    with pytest.raises(ValueError, match="non-negative integer"):
        RuntimeCreditQueue[str](max_active_requests=1, max_waiting_requests=False)


def test_untrusted_request_metadata_is_ignored_by_default() -> None:
    queue = RuntimeCreditQueue[str](
        max_active_requests=1,
        class_limits={"default": 1},
    )

    dispatched = queue.enqueue("r1", "one", _metadata("gold", -1.0))

    assert [item.request_class for item in dispatched] == ["default"]
    assert dispatched[0].deadline_unix_s is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"discipline": "edf"}, "EDF requires"),
        ({"class_limits": {"gold": 1}}, "non-default class limits require"),
    ],
)
def test_untrusted_metadata_rejects_ineffective_policies(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        RuntimeCreditQueue[str](max_active_requests=1, **kwargs)


@pytest.mark.parametrize("deadline", [True, float("inf"), "not-a-number"])
def test_invalid_deadline_metadata_is_rejected(deadline: object) -> None:
    queue = RuntimeCreditQueue[str](
        max_active_requests=1,
        discipline="edf",
        trust_request_metadata=True,
    )
    with pytest.raises(ValueError, match="finite Unix timestamp"):
        queue.enqueue(
            "r1",
            "one",
            {FIRST_OUTPUT_DEADLINE_METADATA_KEY: deadline},
        )
