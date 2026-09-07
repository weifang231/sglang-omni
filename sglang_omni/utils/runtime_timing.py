# SPDX-License-Identifier: Apache-2.0
"""Opt-in timing samples for runtime control overhead measurements."""

from __future__ import annotations

import atexit
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

RUNTIME_TIMING_DIR_ENV = "SGLANG_OMNI_RUNTIME_TIMING_DIR"

_DEFAULT_FLUSH_INTERVAL_S = 2.0
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_RECORDERS: list["RuntimeTimingRecorder"] = []
_RECORDERS_LOCK = threading.Lock()


def _safe_name(value: Any) -> str:
    normalized = _SAFE_FILENAME_RE.sub("_", str(value)).strip("._")
    return normalized or "unknown"


class RuntimeTimingRecorder:
    """Per-process timing sink disabled unless ``RUNTIME_TIMING_DIR_ENV`` is set."""

    def __init__(
        self,
        *,
        engine: str = "sglang-omni",
        component: str,
        stage_id: str | int | None = None,
        flush_interval_s: float = _DEFAULT_FLUSH_INTERVAL_S,
    ) -> None:
        timing_dir = os.environ.get(RUNTIME_TIMING_DIR_ENV)
        self._timing_dir = Path(timing_dir).expanduser() if timing_dir else None
        self._engine = str(engine)
        self._component = str(component)
        self._stage_id: str | int = "unknown" if stage_id is None else stage_id
        self._flush_interval_s = max(float(flush_interval_s), 0.0)
        self._last_flush_s = time.monotonic()
        self._samples: dict[str, list[int]] = {}
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        if self.enabled:
            with _RECORDERS_LOCK:
                _RECORDERS.append(self)

    @property
    def enabled(self) -> bool:
        return self._timing_dir is not None

    def start_ns(self) -> int | None:
        if not self.enabled:
            return None
        return time.perf_counter_ns()

    def record_elapsed_ns(self, family: str, start_ns: int | None) -> None:
        if start_ns is None:
            return
        self.record_ns(family, time.perf_counter_ns() - start_ns)

    def record_ns(self, family: str, duration_ns: int) -> None:
        if not self.enabled:
            return
        family = _safe_name(family)
        now = time.monotonic()
        with self._lock:
            self._samples.setdefault(family, []).append(int(duration_ns))
            self._counts[family] = self._counts.get(family, 0) + 1
            if now - self._last_flush_s >= self._flush_interval_s:
                self._flush_locked(now_s=now)

    def set_stage_id(self, stage_id: str | int) -> None:
        with self._lock:
            self._stage_id = stage_id

    @property
    def path(self) -> Path | None:
        if self._timing_dir is None:
            return None
        filename = (
            f"{_safe_name(self._engine)}."
            f"{_safe_name(self._component)}."
            f"stage-{_safe_name(self._stage_id)}.pid-{os.getpid()}.timing.json"
        )
        return self._timing_dir / filename

    def flush(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._flush_locked(now_s=time.monotonic())

    def _flush_locked(self, *, now_s: float) -> None:
        destination = self.path
        if destination is None or not self._samples:
            return
        payload = {
            "schema_version": 1,
            "clock": "perf_counter_ns",
            "timestamp_s": time.time(),
            "engine": self._engine,
            "component": self._component,
            "stage_id": self._stage_id,
            "pid": os.getpid(),
            "families": self._samples,
            "counts": self._counts,
        }
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                    handle.write("\n")
                os.replace(temporary, destination)
                self._last_flush_s = now_s
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        except Exception:
            pass


def flush_all_runtime_timing_recorders() -> None:
    with _RECORDERS_LOCK:
        recorders = list(_RECORDERS)
    for recorder in recorders:
        recorder.flush()


atexit.register(flush_all_runtime_timing_recorders)
