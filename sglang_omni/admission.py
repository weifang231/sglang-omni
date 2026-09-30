# SPDX-License-Identifier: Apache-2.0
"""Queue-full rejects and the pluggable admission policy.

Stage IPC stringifies exceptions; use ``matches()`` on the error classes.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

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
    """Coordinator-side admission hook.

    ``admit`` runs synchronously on the coordinator's event loop for every
    request that passed the in-flight cap and was not marked
    ``should_bypass_admission``. Returning ``False`` (or raising
    :class:`AdmissionRejectedError`) rejects the request with HTTP 429 before
    anything is submitted to a stage. Every admitted request is released
    exactly once: ``completed`` when its terminal stage(s) succeed, ``aborted``
    when it fails, is aborted, or the coordinator shuts down. ``close`` is
    optional and runs when the coordinator stops.
    """

    def admit(self, request_id: str, request: Any) -> bool: ...

    def completed(self, request_id: str) -> None: ...

    def aborted(self, request_id: str) -> None: ...


def load_admission_policy(config: Any) -> AdmissionPolicy | None:
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
        logger.info("admission_policy %s returned None; native admission stays", spec)
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
    return policy
