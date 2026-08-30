# SPDX-License-Identifier: Apache-2.0
"""Runtime-owned FIFO/EDF queues with global and per-class WIP credits."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

REQUEST_CLASS_METADATA_KEY = "sglang_omni.request_class"
FIRST_OUTPUT_DEADLINE_METADATA_KEY = "sglang_omni.first_output_deadline_unix_s"
DEFAULT_REQUEST_CLASS = "default"

QueueDiscipline = Literal["fifo", "edf"]
ItemT = TypeVar("ItemT")


@dataclass
class RuntimeQueueItem(Generic[ItemT]):
    """One accepted request waiting for or holding a runtime credit."""

    request_id: str
    value: ItemT
    request_class: str
    deadline_unix_s: float | None
    sequence: int
    enqueued_ns: int


@dataclass(frozen=True)
class RuntimeQueueCancellation(Generic[ItemT]):
    """Result of removing a waiting or active request."""

    item: RuntimeQueueItem[ItemT] | None
    was_active: bool
    dispatched: tuple[RuntimeQueueItem[ItemT], ...]


class RuntimeCreditQueue(Generic[ItemT]):
    """Non-preemptive WIP-credit queue.

    A request acquires one global credit and, when configured, one credit for
    its class. Decreasing a limit never preempts active work; it only prevents
    new dispatches until the active count falls below the new limit.
    """

    def __init__(
        self,
        *,
        max_active_requests: int | None = None,
        class_limits: Mapping[str, int] | None = None,
        discipline: QueueDiscipline = "fifo",
        trust_request_metadata: bool = False,
        class_metadata_key: str = REQUEST_CLASS_METADATA_KEY,
        deadline_metadata_key: str = FIRST_OUTPUT_DEADLINE_METADATA_KEY,
    ) -> None:
        normalized_class_limits = self._normalize_limits(
            max_active_requests,
            class_limits or {},
        )
        if discipline not in {"fifo", "edf"}:
            raise ValueError("discipline must be 'fifo' or 'edf'")
        if not isinstance(trust_request_metadata, bool):
            raise ValueError("trust_request_metadata must be a boolean")
        if not trust_request_metadata and discipline == "edf":
            raise ValueError(
                "EDF requires trust_request_metadata=true because deadlines "
                "otherwise remain untrusted"
            )
        if not trust_request_metadata and any(
            request_class != DEFAULT_REQUEST_CLASS
            for request_class in normalized_class_limits
        ):
            raise ValueError(
                "non-default class limits require trust_request_metadata=true"
            )
        if not class_metadata_key.strip():
            raise ValueError("class_metadata_key must not be empty")
        if not deadline_metadata_key.strip():
            raise ValueError("deadline_metadata_key must not be empty")

        self.max_active_requests = max_active_requests
        self.class_limits = normalized_class_limits
        self.discipline: QueueDiscipline = discipline
        self.trust_request_metadata = trust_request_metadata
        self.class_metadata_key = class_metadata_key
        self.deadline_metadata_key = deadline_metadata_key
        self._waiting: list[RuntimeQueueItem[ItemT]] = []
        self._active: dict[str, RuntimeQueueItem[ItemT]] = {}
        self._active_by_class: Counter[str] = Counter()
        self._sequence = 0

    @classmethod
    def from_config(cls, config: Any) -> RuntimeCreditQueue[Any]:
        """Build from a Pydantic config object or an equivalent mapping."""
        values = config.model_dump() if hasattr(config, "model_dump") else dict(config)
        return cls(**values)

    @staticmethod
    def _normalize_limits(
        max_active_requests: int | None,
        class_limits: Mapping[str, int],
    ) -> dict[str, int]:
        if isinstance(max_active_requests, bool) or (
            max_active_requests is not None
            and (
                not isinstance(max_active_requests, int)
                or max_active_requests < 0
            )
        ):
            raise ValueError("max_active_requests must be a non-negative integer")
        normalized: dict[str, int] = {}
        for request_class, limit in class_limits.items():
            if not isinstance(request_class, str) or not request_class.strip():
                raise ValueError("class limit keys must be non-empty strings")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError("class limits must be non-negative integers")
            request_class = request_class.strip()
            if request_class in normalized:
                raise ValueError(
                    "class limit keys must remain unique after trimming whitespace"
                )
            normalized[request_class] = limit
        return normalized

    def enqueue(
        self,
        request_id: str,
        value: ItemT,
        metadata: Mapping[str, Any] | None,
        *,
        enqueued_ns: int | None = None,
    ) -> tuple[RuntimeQueueItem[ItemT], ...]:
        if request_id in self._active or any(
            queued.request_id == request_id for queued in self._waiting
        ):
            raise ValueError(f"request {request_id!r} is already queued or active")

        request_class, deadline_unix_s = self._parse_metadata(metadata)
        item = RuntimeQueueItem(
            request_id=request_id,
            value=value,
            request_class=request_class,
            deadline_unix_s=deadline_unix_s,
            sequence=self._sequence,
            enqueued_ns=time.monotonic_ns() if enqueued_ns is None else enqueued_ns,
        )
        self._sequence += 1
        self._waiting.append(item)
        return self._drain()

    def attributes(
        self, metadata: Mapping[str, Any] | None
    ) -> tuple[str, float | None]:
        """Return the normalized class and absolute deadline metadata."""
        return self._parse_metadata(metadata)

    def release(
        self, request_id: str
    ) -> tuple[RuntimeQueueItem[ItemT], tuple[RuntimeQueueItem[ItemT], ...]] | None:
        item = self._active.pop(request_id, None)
        if item is None:
            return None
        self._active_by_class[item.request_class] -= 1
        if self._active_by_class[item.request_class] <= 0:
            self._active_by_class.pop(item.request_class, None)
        return item, self._drain()

    def cancel(self, request_id: str) -> RuntimeQueueCancellation[ItemT]:
        active = self._active.pop(request_id, None)
        if active is not None:
            self._active_by_class[active.request_class] -= 1
            if self._active_by_class[active.request_class] <= 0:
                self._active_by_class.pop(active.request_class, None)
            return RuntimeQueueCancellation(active, True, self._drain())

        for index, item in enumerate(self._waiting):
            if item.request_id == request_id:
                self._waiting.pop(index)
                return RuntimeQueueCancellation(item, False, self._drain())
        return RuntimeQueueCancellation(None, False, ())

    def update(
        self,
        *,
        max_active_requests: int | None,
        class_limits: Mapping[str, int],
        discipline: QueueDiscipline,
    ) -> tuple[RuntimeQueueItem[ItemT], ...]:
        normalized_class_limits = self._normalize_limits(
            max_active_requests,
            class_limits,
        )
        if discipline not in {"fifo", "edf"}:
            raise ValueError("discipline must be 'fifo' or 'edf'")
        if not self.trust_request_metadata and discipline == "edf":
            raise ValueError(
                "EDF requires trust_request_metadata=true because deadlines "
                "otherwise remain untrusted"
            )
        if not self.trust_request_metadata and any(
            request_class != DEFAULT_REQUEST_CLASS
            for request_class in normalized_class_limits
        ):
            raise ValueError(
                "non-default class limits require trust_request_metadata=true"
            )
        self.max_active_requests = max_active_requests
        self.class_limits = normalized_class_limits
        self.discipline = discipline
        return self._drain()

    def clear(self) -> tuple[RuntimeQueueItem[ItemT], ...]:
        items = tuple(self._active.values()) + tuple(self._waiting)
        self._active.clear()
        self._active_by_class.clear()
        self._waiting.clear()
        return items

    def is_active(self, request_id: str) -> bool:
        return request_id in self._active

    def is_waiting(self, request_id: str) -> bool:
        return any(item.request_id == request_id for item in self._waiting)

    def snapshot(self) -> dict[str, Any]:
        waiting_by_class = Counter(item.request_class for item in self._waiting)
        return {
            "discipline": self.discipline,
            "max_active_requests": self.max_active_requests,
            "class_limits": dict(self.class_limits),
            "trust_request_metadata": self.trust_request_metadata,
            "active_requests": len(self._active),
            "waiting_requests": len(self._waiting),
            "active_by_class": dict(self._active_by_class),
            "waiting_by_class": dict(waiting_by_class),
        }

    def _drain(self) -> tuple[RuntimeQueueItem[ItemT], ...]:
        dispatched: list[RuntimeQueueItem[ItemT]] = []
        while self._global_credit_available():
            index = self._next_eligible_index()
            if index is None:
                break
            item = self._waiting.pop(index)
            self._active[item.request_id] = item
            self._active_by_class[item.request_class] += 1
            dispatched.append(item)
        return tuple(dispatched)

    def _global_credit_available(self) -> bool:
        return (
            self.max_active_requests is None
            or len(self._active) < self.max_active_requests
        )

    def _class_credit_available(self, request_class: str) -> bool:
        limit = self.class_limits.get(request_class)
        return limit is None or self._active_by_class[request_class] < limit

    def _next_eligible_index(self) -> int | None:
        eligible = [
            (index, item)
            for index, item in enumerate(self._waiting)
            if self._class_credit_available(item.request_class)
        ]
        if not eligible:
            return None
        if self.discipline == "fifo":
            return min(eligible, key=lambda pair: pair[1].sequence)[0]
        return min(
            eligible,
            key=lambda pair: (
                math.inf
                if pair[1].deadline_unix_s is None
                else pair[1].deadline_unix_s,
                pair[1].sequence,
            ),
        )[0]

    def _parse_metadata(
        self, metadata: Mapping[str, Any] | None
    ) -> tuple[str, float | None]:
        metadata = metadata or {}
        if not self.trust_request_metadata:
            return DEFAULT_REQUEST_CLASS, None
        raw_class = metadata.get(self.class_metadata_key, DEFAULT_REQUEST_CLASS)
        if not isinstance(raw_class, str) or not raw_class.strip():
            raise ValueError(
                f"request metadata {self.class_metadata_key!r} must be a "
                "non-empty string"
            )
        request_class = raw_class.strip()

        raw_deadline = metadata.get(self.deadline_metadata_key)
        if raw_deadline is None:
            return request_class, None
        if isinstance(raw_deadline, bool):
            raise ValueError(
                f"request metadata {self.deadline_metadata_key!r} must be a "
                "finite Unix timestamp in seconds"
            )
        try:
            deadline_unix_s = float(raw_deadline)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"request metadata {self.deadline_metadata_key!r} must be a "
                "finite Unix timestamp in seconds"
            ) from exc
        if not math.isfinite(deadline_unix_s):
            raise ValueError(
                f"request metadata {self.deadline_metadata_key!r} must be a "
                "finite Unix timestamp in seconds"
            )
        return request_class, deadline_unix_s
