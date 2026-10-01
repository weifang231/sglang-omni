# SPDX-License-Identifier: Apache-2.0
"""Queue-full rejects and the pluggable admission policy.

Stage IPC stringifies exceptions; use ``matches()`` on the error classes.
"""

from __future__ import annotations

import inspect
import logging
from typing import Protocol

from sglang_omni.config.schema import PipelineConfig
from sglang_omni.proto import OmniRequest
from sglang_omni.utils.imports import import_string

logger = logging.getLogger(__name__)


class QueueFullError(RuntimeError):
    """A serving queue is at capacity (HTTP 503)."""

    MESSAGE = "The request queue is full."

    def __init__(self) -> None:
        super().__init__(self.MESSAGE)

    @classmethod
    def matches(cls, exc: BaseException | str | None) -> bool:
        return isinstance(exc, cls) or (exc is not None and cls.MESSAGE in str(exc))

    @classmethod
    def from_message(cls, message: str | None) -> Exception:
        if cls.matches(message):
            return cls()
        else:
            pass
        return RuntimeError(message or "Unknown error")


class AdmissionRejectedError(RuntimeError):
    """An admission policy declined the request before pipeline submit (HTTP 429).

    Unlike :class:`QueueFullError` (a full queue, HTTP 503) this is a deliberate
    decision: the deployment could take the request but chose not to, so the
    caller should retry elsewhere or later rather than treat the server as down.
    """

    MESSAGE = "Admission rejected."

    def __init__(self, reason: str | None = None) -> None:
        if reason:
            message = f"{self.MESSAGE} {reason}"
        else:
            message = self.MESSAGE
        super().__init__(message)
        self.reason = reason

    @classmethod
    def matches(cls, exc: BaseException | str | None) -> bool:
        return isinstance(exc, cls) or (exc is not None and cls.MESSAGE in str(exc))


class AdmissionPolicy(Protocol):
    """Synchronous admission and logical request lifecycle callbacks.

    Callbacks must not block; see the pipeline documentation for ownership,
    cancellation, and callback failure semantics. Two callbacks are optional:
    ``first_output(request_id)`` runs when the coordinator forwards a request's
    first stream chunk (or its completion, for non-streaming requests), which
    is what a latency-aware policy needs to learn from its own traffic; ``close``
    runs on stop. The factory receives the whole ``PipelineConfig``; policy
    parameters live in ``config.admission_policy_options``.
    """

    def admit(self, request_id: str, request: OmniRequest) -> bool: ...

    def completed(self, request_id: str) -> None: ...

    def aborted(self, request_id: str) -> None: ...


def load_admission_policy(config: PipelineConfig) -> AdmissionPolicy | None:
    """Build the policy named by ``config.admission_policy`` (dotted path).

    The path names a callable ``factory(*, config) -> AdmissionPolicy | None``.
    ``None`` from the factory, or no path at all, keeps native admission. A
    factory that is not callable, or a policy without the three required
    methods, is a configuration error and raises ``TypeError`` at start-up
    rather than at the first request.
    """
    spec = getattr(config, "admission_policy", None)
    if not spec:
        return None
    else:
        pass
    factory = import_string(spec)
    if not callable(factory):
        raise TypeError(f"admission_policy {spec!r} must be callable")
    else:
        pass
    policy = factory(config=config)
    if policy is None:
        logger.info(f"admission_policy {spec} returned None; native admission stays")
        return None
    else:
        pass
    missing = [
        name
        for name in ("admit", "completed", "aborted")
        if not callable(getattr(policy, name, None))
    ]
    if missing:
        raise TypeError(
            f"admission_policy {spec!r} returned {type(policy).__name__} "
            f"without callable {', '.join(missing)}"
        )
    else:
        pass
    # The coordinator calls these synchronously; a coroutine function would
    # return an (always truthy) coroutine and silently admit everything.
    asynchronous = [
        name
        for name in ("admit", "completed", "aborted", "close")
        if inspect.iscoroutinefunction(getattr(policy, name, None))
    ]
    if asynchronous:
        raise TypeError(
            f"admission_policy {spec!r} returned {type(policy).__name__} with async "
            f"{', '.join(asynchronous)}; callbacks must be synchronous"
        )
    else:
        pass
    return policy
