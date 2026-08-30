# SPDX-License-Identifier: Apache-2.0
"""Opt-in file channel for lightweight runtime metrics and controls.

The channel is dependency-free and fail-open: malformed control files or
metrics I/O failures never stop a serving loop. Both features are disabled
unless their corresponding environment variable is set.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUNTIME_METRICS_DIR_ENV = "OMNI_RUNTIME_METRICS_DIR"
RUNTIME_CONTROL_FILE_ENV = "OMNI_RUNTIME_CONTROL_FILE"

_DEFAULT_PUBLISH_INTERVAL_S = 0.25
_DEFAULT_CONTROL_POLL_INTERVAL_S = 0.10
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def bounded_int(value: Any, *, minimum: int, maximum: int) -> int | None:
    """Return an integral control value clamped to a safe closed interval."""
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def bounded_float(value: Any, *, minimum: float, maximum: float) -> float | None:
    """Return a finite numeric control value clamped to a safe interval."""
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed):
        return None
    return min(max(parsed, minimum), maximum)


class RuntimeStateChannel:
    """Publish atomic component snapshots and poll a shared JSON control file."""

    def __init__(
        self,
        *,
        engine: str,
        component: str,
        stage_id: str | int | None = None,
        publish_interval_s: float = _DEFAULT_PUBLISH_INTERVAL_S,
        control_poll_interval_s: float = _DEFAULT_CONTROL_POLL_INTERVAL_S,
    ) -> None:
        metrics_dir = os.environ.get(RUNTIME_METRICS_DIR_ENV)
        control_file = os.environ.get(RUNTIME_CONTROL_FILE_ENV)
        self._metrics_dir = Path(metrics_dir).expanduser() if metrics_dir else None
        self._control_file = Path(control_file).expanduser() if control_file else None
        self._engine = str(engine)
        self._component = str(component)
        self._stage_id: str | int = "unknown" if stage_id is None else stage_id
        self._publish_interval_s = max(float(publish_interval_s), 0.0)
        self._control_poll_interval_s = max(float(control_poll_interval_s), 0.0)
        self._last_publish_s = float("-inf")
        self._last_control_poll_s = float("-inf")
        self._last_control_signature: tuple[int, int, int] | None = None
        self._lock = threading.Lock()
        self._warned_failures: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._metrics_dir is not None or self._control_file is not None

    @property
    def metrics_enabled(self) -> bool:
        return self._metrics_dir is not None

    def set_stage_id(self, stage_id: str | int) -> None:
        self._stage_id = stage_id

    def _safe_name(self, value: Any) -> str:
        normalized = _SAFE_FILENAME_RE.sub("_", str(value)).strip("._")
        return normalized or "unknown"

    @property
    def snapshot_path(self) -> Path | None:
        if self._metrics_dir is None:
            return None
        filename = (
            f"{self._safe_name(self._engine)}."
            f"{self._safe_name(self._component)}."
            f"stage-{self._safe_name(self._stage_id)}.pid-{os.getpid()}.json"
        )
        return self._metrics_dir / filename

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if key in self._warned_failures:
            return
        self._warned_failures.add(key)
        logger.warning(message, *args)

    def read_control_if_changed(self, *, force: bool = False) -> dict[str, Any] | None:
        """Return a new valid control object, or ``None`` when unchanged."""
        path = self._control_file
        if path is None:
            return None
        now = time.monotonic()
        if (
            not force
            and now - self._last_control_poll_s < self._control_poll_interval_s
        ):
            return None
        self._last_control_poll_s = now
        try:
            stat = path.stat()
            signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if not force and signature == self._last_control_signature:
                return None
            with path.open("r", encoding="utf-8") as handle:
                control = json.load(handle)
            if not isinstance(control, dict):
                raise ValueError("runtime control JSON must be an object")
        except FileNotFoundError:
            had_control = self._last_control_signature is not None
            self._last_control_signature = None
            return {} if had_control else None
        except Exception:
            self._warn_once(
                "control-read",
                "Ignoring invalid runtime control file %s",
                path,
            )
            return None
        self._warned_failures.discard("control-read")
        self._last_control_signature = signature
        return control

    def maybe_publish(
        self,
        fields_factory: Callable[[], Mapping[str, Any]],
        *,
        force: bool = False,
    ) -> bool:
        """Atomically publish a snapshot when the component interval is due."""
        destination = self.snapshot_path
        if destination is None:
            return False
        now = time.monotonic()
        if not force and now - self._last_publish_s < self._publish_interval_s:
            return False
        self._last_publish_s = now
        try:
            fields = dict(fields_factory())
            payload = {
                **fields,
                "timestamp_s": time.time(),
                "engine": self._engine,
                "component": self._component,
                "stage_id": self._stage_id,
                "pid": os.getpid(),
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=destination.parent,
                )
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        json.dump(
                            payload,
                            handle,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        handle.write("\n")
                    os.replace(temporary, destination)
                finally:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
        except Exception:
            self._warn_once(
                "metrics-write",
                "Failed to publish runtime metrics snapshot %s",
                destination,
            )
            return False
        self._warned_failures.discard("metrics-write")
        return True
